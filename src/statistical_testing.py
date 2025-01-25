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
import itertools
from torch import Tensor
from typing import List

def get_context_predindex_pair_effect(model, p, context, targetall, nrepp):

    context_predindex_pair_effect = torch.zeros((p, p))  # Initialize a 2D array with zeros

    torch.manual_seed(42)
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
    
    torch.manual_seed(42)

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

        
        
        
        if (target_sample_size == len(targetall)) and torch.backends.mps.is_available():
            pval = permute_meansq_pval_cal(tsq, meansq).to('cpu')
        else:
            # Compute empirical p-values for each value in meansq
            pval = calc_pval(meansq, permute_meansq(tsq).to('cpu')) 
            




        # Accumulate p-values
        pval_mat[:, repetition] = pval
    
    # Average the p-values over the number of repetitions
    avepval = pval_mat.mean(dim=1)
    
    if nrepp > 1:
        # Compute standard deviation of p-values
        stdpval = pval_mat.std(dim=1)
    else:
        stdpval = torch.zeros(p)

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


def permute_meansq_pval_cal(
    tsq: Tensor,
    meansq: Tensor,
    device: str = 'mps',
    batch_size: int = 10000000,
    verbose: bool = True
) -> Tensor:
    """
    Optimized computation of the squared mean of sums over all possible combinations
    where, at each position (target), it selects one element from each column of tsq.

    Parameters:
    ----------
    tsq : torch.Tensor
        2D tensor of shape (ncont, ntar), where each column represents a set of elements.
    meansq : torch.Tensor
        1D tensor of shape (ncont,) containing mean squared values for comparison.
    device : str, optional
        Device to perform computation on ('cpu', 'cuda', or 'mps'), by default 'mps'.
    batch_size : int, optional
        Number of combinations to process per batch, by default 100000.
    verbose : bool, optional
        If True, prints progress updates, by default True.

    Returns:
    -------
    torch.Tensor
        1D tensor of shape (ncont,) containing the computed p-values for each cont.
    """
    # Move tensors to the specified device
    tsq = tsq.to(device)
    meansq = meansq.to(device)
    ncont, ntar = tsq.shape

    # Extract columns as lists for itertools.product
    # Convert each column tensor to a list
    col_sets: List[List[float]] = [tsq[:, i].tolist() for i in range(ntar)]

    # Initialize pval tensor
    pval = torch.zeros(ncont, device=device)

    # Total number of combinations for normalization
    total_combinations = ncont ** ntar  # Adjust if sets have different sizes

    # Create an iterator for the Cartesian product
    cartesian_iter = itertools.product(*col_sets)

    # Initialize counter
    cnt = 0

    # Function to process a single batch
    def process_batch(batch: List[tuple]) -> Tensor:
        """
        Processes a batch of combinations.

        Parameters:
        ----------
        batch : List[tuple]
            List of combination tuples.

        Returns:
        -------
        torch.Tensor
            Tensor of shape (batch_size, ncont) representing the batch.
        """
        # Convert list of tuples to a 2D tensor (batch_size, ncont)
        # Use torch.float32 for precision; adjust dtype if necessary
        batch_tensor = torch.tensor(batch, dtype=tsq.dtype, device=device)  # Shape: (batch_size, ncont)
        return batch_tensor

    # Iterate over the Cartesian product in batches
    while True:
        # Fetch a batch of combinations
        batch = list(itertools.islice(cartesian_iter, batch_size))
        if not batch:
            break  # Exit loop if no more combinations

        # Convert batch to tensor
        try:
            batch_tensor = process_batch(batch)  # Shape: (batch_size, ncont)
        except Exception as e:
            print(f"Error processing batch at count {cnt}: {e}")
            break

        # Compute mean across targets (columns) for each combination
        # Shape: (batch_size,)
        mean_collection = batch_tensor.mean(dim=1)

        # Compute squared mean
        # Shape: (batch_size,)
        squared_mean_collection = mean_collection ** 2

        # Compare with meansq
        # meansq: (ncont,)
        # squared_mean_collection: (batch_size,)
        # We need to compare each squared mean with each meansq element
        # To do this, we can expand dimensions for broadcasting

        # Expand squared_mean_collection to (batch_size, 1)
        # meansq is (ncont,), we need to expand to (1, ncont)
        comparison = (squared_mean_collection.unsqueeze(1) >= meansq.unsqueeze(0)).float()  # Shape: (batch_size, ncont)

        # Sum over the batch dimension to get counts per cont
        pval += comparison.sum(dim=0)  # Shape: (ncont,)

        # Update counter
        cnt += len(batch)

        # Print progress if verbose
        if verbose and (cnt % (batch_size * 10) == 0):
            print(f"Processed {cnt} combinations out of approximately {total_combinations}")

    # Normalize pval to get p-values
    pval = pval / total_combinations

    return pval

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
    # torch.manual_seed(42)
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
        print(f"Variable {i+1}: {avepval[i]:.4f} ± {stdpval[i]:.2f}")


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