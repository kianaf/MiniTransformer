import torch
import torch.nn as nn
import statsmodels.api as sm

import numpy as np

def calculate_bench1_loss(train_data, eval_data, predindex):
    """Calculate the loss for benchmark model 1 (global mean baseline).
    
    This benchmark model predicts future values by using the global mean across all
    timepoints and sequences in the training data. It serves as a simple baseline
    that doesn't consider temporal dependencies.
    
    Args:
        train_data (list[torch.Tensor]): List of training sequences, each of shape 
            (seq_len, num_features)
        eval_data (list[torch.Tensor]): List of evaluation sequences to test on
        predindex (int): Index of the feature to focus on for specific predictions
    
    Returns:
        tuple: Contains:
            - dimave (torch.Tensor): Global mean values for each feature
            - benchloss (float): Average MSE across all features
            - benchloss_predindex (float): MSE for the specific feature at predindex
    """
    p = train_data[0].shape[1]
    predn = len(eval_data)
    
    train_data_plain = torch.cat([train_data[i] for i in range(len(train_data))], dim=0)
    dimave = torch.mean(train_data_plain, dim=0)

    benchdelta = 0.0
    benchloss = 0.0
    benchloss_predindex = 0.0   

    for i in range(predn):
        benchdelta = eval_data[i][-1, :] - dimave
        benchloss += torch.mean(benchdelta * benchdelta).item()
        benchloss_predindex += torch.mean(benchdelta[predindex] * benchdelta[predindex]).item()

    benchloss /= predn
    benchloss_predindex /= predn

    return dimave, benchloss, benchloss_predindex


def calculate_bench2_loss(train_data, eval_data, dimave):
    """Calculate the loss for benchmark model 2 (conditional probability baseline).
    
    This benchmark predicts pos3 based on the conditional probability given pos2's value,
    and uses global means for other features. It captures simple conditional dependencies
    between specific positions.
    
    Args:
        train_data (list[torch.Tensor]): List of training sequences
        eval_data (list[torch.Tensor]): List of evaluation sequences
        dimave (torch.Tensor): Global mean values for each feature from bench1
    
    Returns:
        tuple: Contains:
            - bench2loss (float): Average MSE across all features
            - bench2loss_predindex (float): MSE for position 3 specifically
    """
    p = train_data[0].shape[1]
    predn = len(eval_data)
    n = len(train_data)

    pos1, pos2, pos3 = 0, 1, 2
    pos2count1, pos2count0 = 0, 0
    pos3ave1, pos3ave0 = 0.0, 0.0

    # Calculate conditional probabilities
    for i in range(n):
        curn = train_data[i].shape[0] 
        for j in range(curn-1):
            if (train_data[i][j, pos2] == 1.0):
                pos2count1 += 1
                pos3ave1 += train_data[i][j+1, pos3]
            else:
                pos2count0 += 1
                pos3ave0 += train_data[i][j+1, pos3]

    pos3ave1 = pos3ave1/pos2count1
    pos3ave0 = pos3ave0/pos2count0

    bench2pred = 0.0
    bench2delta = 0.0
    bench2loss = 0.0
    bench2loss_predindex = 0.0

    for i in range(predn):
        for j in range(p):
            if j == pos3:
                if eval_data[i][-2, pos3] == 1.0:   
                    bench2pred = 0.0
                else:
                    bench2pred = pos3ave1 if eval_data[i][-2, pos2] == 1.0 else pos3ave0
            else:
                bench2pred = dimave[j]

            bench2delta = eval_data[i][-1, j] - bench2pred
            bench2loss += bench2delta * bench2delta
            if j == pos3:
                bench2loss_predindex += (bench2delta * bench2delta)

    bench2loss /= predn*p
    bench2loss_predindex /= predn

    return bench2loss.item(), bench2loss_predindex.item()


def calculate_repeat_loss(eval_data, predindex):
    """Calculate the loss for benchmark model 3 (repeat last value baseline).
    
    This benchmark simply predicts that each feature will maintain its last observed value.
    It serves as a persistence baseline that assumes no change between timesteps.
    
    Args:
        eval_data (list[torch.Tensor]): List of evaluation sequences
        predindex (int): Index of the feature to focus on for specific predictions
    
    Returns:
        tuple: Contains:
            - bench3loss (float): Average MSE across all features
            - bench3_loss_predindex (float): MSE for the specific feature at predindex
    """
    p = eval_data[0].shape[1]
    predn = len(eval_data)
    size = predn

    bench3loss = 0.0
    bench3_loss_predindex = 0.0

    for i in range(predn):
        if eval_data[i].shape[1] < 3:
            size -= 1
            continue
        else:
            bench3delta = eval_data[i][-1, :] - eval_data[i][-2, :]
            bench3_loss_predindex += torch.mean(bench3delta[predindex]*bench3delta[predindex]).item()
            bench3loss += torch.mean(bench3delta * bench3delta).item()

    bench3loss /= predn 
    bench3_loss_predindex /= predn

    return bench3loss, bench3_loss_predindex


def calculate_regression_loss(train_data, eval_data, predindex, return_per_target=False):
    """Calculate the loss for a GLM regression baseline model.

    Fits separate Gaussian GLM models for each feature using all other features at t-1
    as predictors. This serves as a linear modeling baseline that captures basic
    temporal dependencies.

    Args:
        train_data (list[torch.Tensor]): List of training sequences
        eval_data (list[torch.Tensor]): List of evaluation sequences
        predindex (int): Index of the feature to focus on for specific predictions
        return_per_target (bool): If True, additionally return the full per-target
            evaluation-MSE array (length p). The first two return values are
            unchanged for backwards compatibility with the existing callers used
            to produce the paper's Tables 1 and 3.

    Returns:
        tuple: Contains:
            - pred_feature_mse (float): MSE for the specific feature at predindex
            - overall_mse (float): Average MSE across all features
            - (optional) per_target_mse (np.ndarray of shape (p,)): per-target
              evaluation-fold MSE, returned only when return_per_target=True.
    """
    num_features = train_data[0].shape[1]
    models = []
    train_mses = []
    eval_mses = []

    # Train separate regressions for each feature
    for feature_idx in range(num_features):
        train_features = []
        train_targets = []
        
        # Gather training data
        for seq in train_data:
            for t in range(2, seq.shape[0] - 1):
                train_features.append(seq[t - 1, :])
                train_targets.append(seq[t, feature_idx])

        train_features = np.stack(train_features)
        train_targets = np.stack(train_targets)

        # Add a constant term for the intercept
        train_features = sm.add_constant(train_features)

        # Fit the GLM for the current feature
        model = sm.GLM(train_targets, train_features, family=sm.families.Gaussian()).fit()
        models.append(model)

        # Calculate training MSE
        train_predictions = model.predict(train_features)
        train_mses.append(np.mean((train_predictions - train_targets) ** 2))

        # Prepare evaluation data (only for the last time point)
        eval_features = []
        eval_targets = []

        for seq in eval_data:
            if seq.shape[1] < 3:
                continue
            else:
                eval_features.append(seq[-2, :])  # Use all variables at t-1 for the last time point
                eval_targets.append(seq[-1, feature_idx])  # Target: feature_idx at the last time point

        # Convert to NumPy arrays
        eval_features = np.array(eval_features)
        eval_targets = np.array(eval_targets)

        # Add a constant term for the intercept
        eval_features = sm.add_constant(eval_features)

        # Predict on evaluation data
        eval_predictions = model.predict(eval_features)

        # Calculate evaluation MSE
        eval_mse = np.mean((eval_predictions - eval_targets) ** 2)
        eval_mses.append(eval_mse)

    # Print evaluation results
    # for i, (train_mse, eval_mse) in enumerate(zip(train_mses, eval_mses)):
    #     print(f"Feature {i+1}: Train MSE = {train_mse:.4f}, Eval MSE = {eval_mse:.4f}")

    if return_per_target:
        return np.mean(eval_mses[predindex]), np.mean(eval_mses), np.array(eval_mses)
    return np.mean(eval_mses[predindex]), np.mean(eval_mses)
    


def evaluate_mini_transformer(eval_data, model, predindex):
    """Evaluates a MiniTransformer model on evaluation data.
    
    Computes both the MSE for a specific prediction index and the overall MSE across
    all features. The evaluation is done on the last time point prediction only.

    Args:
        eval_data (DataLoader): DataLoader containing evaluation batches
        model (MiniTransformer): The transformer model to evaluate
        predindex (int): Index of the specific feature to evaluate

    Returns:
        tuple: (specific_mse, overall_mse)
            - specific_mse (float): MSE for the specified prediction index
            - overall_mse (float): Average MSE across all features
    """
    model.eval()
    loss = 0
    loss_predindex = 0
    size = len(eval_data)
    for batch in eval_data:

        # no padding here
        pred = model((batch[0][:, :-1, :], batch[1][:, :-1, :]))
        loss += nn.MSELoss()(pred[:, -1, :], batch[0][:, -1, :])
        loss_predindex += nn.MSELoss()(pred[:, -1, predindex], batch[0][:, -1, predindex])

    return loss_predindex/size, loss/size
