import torch
import torch.nn as nn
import statsmodels.api as sm

import numpy as np

def calculate_bench1_loss(train_data, eval_data):
    """
    Calculate the loss of the benchmark 1 model.
    averaging over all timepoints of all the sequences
    """

    p = train_data[0].shape[1]

    predn = len(eval_data)

    #This is for baseline1: Bench1 which averages over all timepoints of all the sequences
    
    train_data_plain = torch.cat([train_data[i][:-1, :] for i in range(len(train_data))], dim=0)
    dimave = torch.mean(train_data_plain, dim=0)
    


    benchdelta = 0.0
    benchloss = 0.0

    for i in range(predn):
        benchdelta = eval_data[i][-1, :] - dimave
        benchloss += torch.mean(benchdelta * benchdelta).item()

    benchloss /= predn
    

    return dimave, benchloss



def calculate_bench2_loss(train_data, eval_data, dimave):
    """
    Calculate the loss of the benchmark 2 model.
    Looking at pos2, and based on the probability of pos3 given pos2, predict pos3
    """

    p = train_data[0].shape[1]

    predn = len(eval_data)

    n = len(train_data)

    #This is for baseline2: Bench2 which averages over all timepoints of all the sequences
    



    pos1 = 0
    pos2 = 1
    pos3 = 2

    pos2count1 = 0
    pos2count0 = 0
    pos3ave1 = 0.0
    pos3ave0 = 0.0

    for i in range(n):

        curn = train_data[i].shape[0] 

        for j in range(curn-1):
            if (train_data[i][j, pos3] == 1.0):
                pos2count1+= 1
                pos3ave1 += train_data[i][j, pos3]
            else:
                pos2count0+= 1
                pos3ave0 += train_data[i][j, pos3]

    pos3ave1 = pos3ave1/pos2count1
    pos3ave0 = pos3ave0/pos2count0
        


    bench2pred = 0.0
    bench2delta = 0.0
    bench2loss = 0.0

    for i in range(predn):
        for j in range(p):
            if j == pos3:
                if eval_data[i][-2, pos3] == 1.0:   
                    bench2pred = 0.0
                else:
                    if eval_data[i][-2, pos2] == 1.0:
                        bench2pred = pos3ave1
                    else:
                        bench2pred = pos3ave0
            else:
                bench2pred = dimave[j]

            bench2delta = eval_data[i][-1, j] - bench2pred
            bench2loss += bench2delta * bench2delta


    bench2loss /= predn*p

    return bench2loss.item()


def calculate_repeat_loss(eval_data):
    """
    Calculate the loss of the benchmark 3 model.
    Repeat the last timepoint of each sequence
    """
    
    p = eval_data[0].shape[1]

    predn = len(eval_data)
    size = predn

    bench3loss = 0.0

    for i in range(predn):
        if eval_data[i].shape[1] < 3:
                size -= 1
                continue
        else:
            bench3delta = eval_data[i][-1, :] - eval_data[i][-2, :]
            bench3loss += torch.mean(bench3delta * bench3delta).item()

    bench3loss /= predn 

    return bench3loss


def calculate_regression_loss(train_data, eval_data, predindex):
    """
    Calculate the loss of the regression model.
    """
    
    # Example: train_data and eval_data have shapes (num_samples, num_time_points, num_features)

    num_features = train_data[0].shape[1]  # Number of variables
    models = []
    train_mses = []
    eval_mses = []

    # Train separate regressions for each feature
    for feature_idx in range(num_features):
        train_features = []
        train_targets = []
        
        # Loop over sequences to gather data for all time points except the first
        for seq in train_data:
            for t in range(2, seq.shape[0] - 1):  # Start from the second time point
                train_features.append(seq[t - 1, :])  # Use all variables at t-1
                train_targets.append(seq[t, feature_idx])  # Target: feature_idx at time t

        # Convert to NumPy arrays
        train_features = np.stack(train_features)
        train_targets = np.stack(train_targets)

        # Add a constant term for the intercept
        train_features = sm.add_constant(train_features)

        # Fit the GLM for the current feature
        model = sm.GLM(train_targets, train_features, family=sm.families.Gaussian()).fit()
        models.append(model)

        # Calculate training MSE
        train_predictions = model.predict(train_features)
        train_mse = np.mean((train_predictions - train_targets) ** 2)
        train_mses.append(train_mse)

        # Prepare evaluation data
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
        
    return np.mean(eval_mses[predindex]), np.mean(eval_mses)
    


def evaluate_mini_transformer(eval_data, model, predindex):
    """
    Evaluate the MiniTransformer model on the evaluation data.
    """
    model.eval()
    loss = 0
    loss_predindex = 0
    size = len(eval_data)
    for batch in eval_data:
        if batch[0].shape[1] < 3:
            size -= 1
            continue
        else:
            # no padding here
            pred = model(batch)
            loss += nn.MSELoss()(pred[:, -3, :], batch[0][:, -1, :])
            loss_predindex += nn.MSELoss()(pred[:, -3, predindex], batch[0][:, -1, predindex])

    
    return loss_predindex/size, loss/size
