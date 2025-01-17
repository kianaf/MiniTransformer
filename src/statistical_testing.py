import torch
from itertools import repeat
import multiprocessing as mp
from torch import nn
import math
import matplotlib.pyplot as plt
# Update the default font
plt.rcParams['font.family'] = 'Helvetica'
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def get_context_predindex_pair_effect(model, p, context, targetall, nrepp):

    context_predindex_pair_effect = torch.zeros((p, p))  # Initialize a 2D array with zeros

    for rep in range(nrepp):
        print(f"Repetition {rep + 1}/{nrepp}")
        for i in range(p):
            meansq, tsq = meansq_context(model, context, targetall, i)  # Call the function
            context_predindex_pair_effect[i, :] += meansq  # Accumulate the results

    context_predindex_pair_effect /= nrepp  # Average the results

    return context_predindex_pair_effect




def statistical_testing(model, train_dataset, p, predindex, nrepp, target_sample_size):
    # Create context and target as identity tensors of size p x p
    context = torch.eye(p)  # Identity matrix as tensor
    # target = context.clone()  # Clone to create an identical target tensor

    # Define avepval tensor to accumulate p-values
    pval_mat = torch.zeros(p, nrepp)

    targetall = all_comb(p)
    # targetall = get_the_existing_comb(train_dataset)

    for repetition in range(nrepp):

        # print repetition number
        print(f"Repetition {repetition + 1}/{nrepp}")

        # Randomly select indices
        selected_indexes = torch.randint(0, len(targetall), (target_sample_size,), dtype=torch.int32)
        target_selected_indexes = targetall[selected_indexes]
        target = target_selected_indexes

        # Compute the squared mean of differences for the context-target pair
        meansq, tsq = meansq_context(model, context, target, predindex)


        # print("shape of tsq: ", tsq.shape)

        
        # Compute empirical p-values for each value in meansq
        pval = calc_pval(meansq, permute_meansq(tsq).to('cpu')) 
        
        # collection = []
        # pval = calc_pval(meansq, torch.stack(permute_meansq(tsq, 0, 0.0, collection)))



        # Accumulate p-values
        pval_mat[:, repetition] = pval
    
    # Average the p-values over the number of repetitions
    avepval = pval_mat.mean(dim=1)
    
    stdpval = pval_mat.std(dim=1)

    return avepval, stdpval, context, targetall

   

def all_comb(p):
    # Generate all combinations of binary numbers from 0 to 2^p - 1
    # This will be the range of numbers represented in binary
    binary_range = torch.arange(2 ** p)
    
    # Convert each number to binary, with each row representing one combination
    # Unsqueeze and repeat to match binary length p
    combinations = (binary_range.unsqueeze(1).bitwise_and(2 ** torch.arange(p - 1, -1, -1)) > 0).float()
    
    return combinations[torch.randperm(combinations.shape[0])]  # Shuffle the combinations

def get_the_existing_comb(train_dataset):
    
    # Get unique feature vectors
    unique_combs = torch.unique(torch.cat(train_dataset), dim=0)
    
    return unique_combs[torch.randperm(unique_combs.shape[0])]


def calc_pval(meansq, collection):

    pvals = torch.zeros(meansq.shape[0])
    
    for i in range(meansq.shape[0]):
        abovecount = (collection >= meansq[i]).sum()
    
        pvals[i] = (abovecount/collection.shape[0])

    return pvals



def permute_meansq(tsq, device='mps'):
    """
    Computes the squared mean of sums over all possible combinations
    where, at each position (target), it selects one element from each column of tsq.

    Parameters:
    tsq (torch.Tensor): 2D tensor of shape (ncont, ntar)
    device (str): Device to perform computation on ('cpu' or 'cuda')

    Returns:
    torch.Tensor: 1D tensor containing the computed results for each combination.
    """
    tsq = tsq.to(device)

    ncont, ntar = tsq.shape

    # Generate all possible combinations of contributors per target
    indices = [torch.arange(ncont,  device=device) for _ in range(ntar)]
    combinations = torch.cartesian_prod(*indices)  # Shape: (ncomb, ntar), where ncomb = ncont ** ntar

    # Compute linear indices to select elements from tsq.flatten()
    # Linear index formula: index = row_index * ntar + column_index

    # Flatten tsq to a 1D tensor for advanced indexing
    tsq_flat = tsq.flatten()

    # Select the values corresponding to each combination
    # selected_values = tsq_flat[combinations * ntar +  torch.arange(ntar,  device=device).unsqueeze(0).expand(combinations.size(0), -1)]


    # Use combinations to index tsq directly, preserving 2D structure
    selected_values = torch.stack([tsq[combinations[:, i], i] for i in range(ntar)], dim=1)  # Shape: (ncomb, ntar)



    # Compute the squared mean for each combination
    collection = (selected_values.mean(dim=1) ** 2)
    # collection = selected_values.mean(dim=1)

    return collection


# def permute_meansq(tsq, device='mps'):
#     """
#     computation of the squared mean of sums over all possible combinations
#     where, at each position (target), it selects one element from each column of tsq.

#     Parameters:
#     tsq (torch.Tensor): 2D tensor of shape (ncont, ntar)
#     device (str): Device to perform computation on ('cpu', 'cuda', or 'mps')

#     Returns:
#     torch.Tensor: 1D tensor containing the computed results for each combination.
#     """
#     tsq = tsq.to(device)
#     ncont, ntar = tsq.shape

#     # Generate all possible combinations of row indices for each column
#     indices = torch.cartesian_prod(*[torch.arange(ncont, device=device) for _ in range(ntar)])  # Shape: (ncomb, ntar)

#     # Use combinations to index tsq directly, preserving 2D structure
#     selected_values = tsq[indices, torch.arange(ntar, device=device)]  # Shape: (ncomb, ntar)

#     # Compute the squared mean for each combination
#     collection = selected_values.mean(dim=1) ** 2  # Shape: (ncomb,)

#     return collection




def meansq_context(model, context, target, predindex):
    """
    Compute the squared mean of differences in prediction of feature predindex for each context-target pair.

    Parameters:
    model (ModelParams): Model parameters.
    context (list of tensors): Context sequences.
    target (list of tensors): Target sequences.
    predindex (int): Index of the prediction target.

    Returns:
    tuple of list of float: Tuple containing the squared mean of differences (meansq) and the differences (tsq).
    """

    model.eval()

    ncont = len(context)
    ntar = len(target)

    # initiate tensors for the meansq and tsq
    meansq = torch.zeros(ncont)
    tsq = torch.zeros(ncont , ntar)


    for i in range(ncont):
        curqueryother, curkeyother, curvalueother = model.multiheadattn.qkv(context[i].unsqueeze(0))
        for j in range(ntar):       
            curquery, curkeyself, curvalueself = model.multiheadattn.qkv(target[j].unsqueeze(0))
            curselfweight = torch.exp(curquery.matmul(curkeyself.transpose(2, 3)) / math.sqrt(model.dk))
            curotherweight =  torch.exp(curquery.matmul(curkeyother.transpose(2, 3)) / math.sqrt(model.dk))

            curwsum = curselfweight + curotherweight
            transval = ((curvalueself * curselfweight/curwsum + curvalueother * curotherweight/curwsum )).sum(dim = -1).unsqueeze(2).expand(1,model.num_heads, model.ncum,-1)
            pureval = (curvalueself.sum(dim = -1)).unsqueeze(2).expand(1,model.num_heads, model.ncum,-1)

            # Expand weights to broadcast
            weights_expanded = model.multiheadattn.cum_weights.unsqueeze(0).unsqueeze(3).expand(1, model.num_heads, model.ncum, 1) # 1 here we some over dv or it is one

            # weights_expanded = weights_expanded.expand_as(transval) 
            
            head_outputs_transval = (transval * weights_expanded).sum(dim=1)
            head_outputs_curvalueself = (pureval * weights_expanded).sum(dim=1)

            pred_transval = model.predict(head_outputs_transval)
            pred_curvalueself = model.predict(head_outputs_curvalueself)
    
            # curdelta = torch.abs(pred_transval - pred_curvalueself)[:, :, predindex].item()
            curdelta = (pred_transval - pred_curvalueself)[:, :, predindex].item()
            
            tsq[i, j] = curdelta #* curdelta
        
        meansq[i] = (tsq[i, :].mean())**2
        # meansq[i] = (tsq[i, :].mean())
    
    return meansq, tsq
            



#FIXME the model never has the possibility to learn what to predict as the second time point of sequence (seeing only one time point)


def print_p_values(avepval, stdpval):
    """
    Print the p-values in a readable format.

    Parameters:
    avepval (torch.Tensor): Average p-values to print.
    
    stdpval (torch.Tensor): Standard deviation of p-values to print.
    """
    for i in range(avepval.shape[0]):
        print(f"Variable {i+1}: {avepval[i]:.6f} ± {stdpval[i]:.6f}")


def plot_context_predindex_pair_effect(context_predindex_pair_effect, data_str, run_path):
    """
    Plot the heatmap for context-predindex pair effect.

    Parameters:
    context_predindex_pair_effect (torch.Tensor): Context-predindex pair effect to plot.
    """

    if data_str == "ghq_b_sum":
        variable_names = [
            "dh_10: Nightmares",
            "dh_35: Sleep problems",
            "dh_37: Paperwork",
            "dh_38: Housekeeping",
            "dh_45: Noise",
            "dh_53: Long work hours",
            "le_8: Financial problems",
            "le_17: Arguments with partner",
            "le_22: Serious illness",
            "ghq_b_sum: Anxiety & sleep issues"
        ]
    else: 
        if data_str == "ghq_sum":
            variable_names = [
                "dh_11: Commute to work/school",
                "dh_31: Unwanted visit",
                "dh_37: Paperwork",
                "dh_38: Housekeeping",
                "dh_42: Bad weather",
                "dh_46: Traffic",
                "le_1: Lost job",
                "le_16: Breakup",
                "le_17: Arguments with partner",
                "ghq_sum: Psychological distress"
            ]
        else:
            variable_names = [f"Feature {i+1}" for i in range(context_predindex_pair_effect.shape[0])]

    # Plot the context-predindex pair effect


    # red: #AE232F
    # blue: #1D4A91
    # Define your custom colors
    colors = ["#1D4A91", "#FFFFFF", "#AE232F"] 

    # Create the colormap
    custom_cmap = LinearSegmentedColormap.from_list("custom_palette", colors)

    plt.figure(figsize=(8, 8))
    im = plt.imshow(context_predindex_pair_effect, cmap=custom_cmap, interpolation='nearest')
    plt.colorbar(im)
    plt.xlabel(r"Context ($c_j$)")
    plt.ylabel(r"Target ($k$)")
    plt.title(r"Pairwise context-target effect $\overline{\Delta^2}$")

    # Rotate the x-axis labels by 90 degrees
    plt.xticks(ticks=range(len(variable_names)), labels=variable_names, rotation=90)
    plt.yticks(ticks=range(len(variable_names)), labels=variable_names)

    # save the plot
    plt.savefig(run_path + f"/context_target_effect_{data_str}.png", dpi=300, bbox_inches="tight")
    plt.show()