import torch
from itertools import repeat
import multiprocessing as mp
from torch import nn
import math

# def permute_meansq_sampled_gpu(tsq, num_samples, device='mps'):
#     """
#     Estimates the squared mean of sums over a random subset of combinations using PyTorch with GPU acceleration.

#     Parameters:
#     tsq (torch.Tensor): 2D tensor of shape (ncont, ntar) containing input data.
#     num_samples (int): Number of random combinations to sample.
#     device (str): Device to perform computation on ('cuda' for GPU or 'cpu').

#     Returns:
#     torch.Tensor: 1D tensor containing the computed results for sampled combinations on the specified device.
#     """
#     # Move tsq to the specified device
#     tsq = tsq.to(device)
#     ncont, ntar = tsq.shape

#     # Randomly select contributors for each sample and target
#     indices = torch.randint(0, ncont, (num_samples, ntar), device=device)


#     # Generate target indices
#     target_indices = torch.arange(ntar, device=device).unsqueeze(0)

#     # Select the values corresponding to each sampled combination
#     selected_values = tsq[indices, target_indices]

#     # Compute the sum of selected values for each sample
#     sums = selected_values.sum(dim=1)

#     # Compute the squared mean for each sample
#     collection = (sums ** 2) / (ntar ** 2)

#     return collection



# def permute_meansq_sampled_gpu_batch(tsq, num_samples, batch_size, device='mps'):
#     tsq = tsq.to(device)
#     ncont, ntar = tsq.shape
#     total_batches = (num_samples + batch_size - 1) // batch_size
#     results = []

#     for batch_idx in range(total_batches):
#         current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
#         indices = torch.randint(0, ncont, (current_batch_size, ntar), device=device)
#         target_indices = torch.arange(ntar, device=device).unsqueeze(0)
#         selected_values = tsq[indices, target_indices]
#         sums = selected_values.sum(dim=1)
#         collection = (sums ** 2) / (ntar ** 2)
#         results.append(collection)

#     # Concatenate all batches
#     return torch.cat(results)


def statistical_testing(model, p, predindex, nrepp, target_sample_size):
    # Create context and target as identity tensors of size p x p
    context = torch.eye(p)  # Identity matrix as tensor
    # target = context.clone()  # Clone to create an identical target tensor

    # Define avepval tensor to accumulate p-values
    avepval = torch.zeros(p)

    targetall = all_comb(p)

    for _ in range(nrepp):

        # print repetition number
        print(f"Repetition {_ + 1}/{nrepp}")

        # Randomly select indices
        selected_indexes = torch.randint(0, len(targetall), (target_sample_size,), dtype=torch.int32)
        target_selected_indexes = targetall[selected_indexes]
        # target = torch.cat((target, target_selected_indexes), dim=0)
        target = target_selected_indexes#torch.cat((target, target_selected_indexes), dim=0)

        # Compute the squared mean of differences for the context-target pair
        meansq, tsq = meansq_context(model, context, target, predindex)


        # print("shape of tsq: ", tsq.shape)

        
        # Compute empirical p-values for each value in meansq
        pval = calc_pval(meansq, permute_meansq(tsq).to('cpu')) 

        # collection = []
        # pval = calc_pval(meansq, torch.stack(permute_meansq(tsq, 0, 0.0, collection)))


        # Accumulate p-values
        avepval += pval
    
    # Average the p-values over the number of repetitions
    avepval /= nrepp

    return avepval

   

def all_comb(p):
    # Generate all combinations of binary numbers from 0 to 2^p - 1
    # This will be the range of numbers represented in binary
    binary_range = torch.arange(2 ** p)
    
    # Convert each number to binary, with each row representing one combination
    # Unsqueeze and repeat to match binary length p
    combinations = (binary_range.unsqueeze(1).bitwise_and(2 ** torch.arange(p - 1, -1, -1)) > 0).float()
    
    return combinations[torch.randperm(combinations.shape[0])]  # Shuffle the combinations


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

# def permute_meansq(tsq, curpos, curcum, collection):
#     """
#     Recursively computes the squared mean of sums over all possible combinations
#     where, at each position (target), it selects one element from each column of tsq.

#     Parameters:
#     tsq (numpy.ndarray): 2D array of shape (ncont, ntar)
#     curpos (int): Current position in the recursion (starting from 0)
#     curcum (float): Current cumulative sum (starting from 0.0)
#     collection (list): List to collect the computed results
#     """
#     ncont, ntar = tsq.shape
#     for i in range(ncont):

#         curval = curcum + tsq[i, curpos]
#         if curpos == ntar - 1:
#             result = (curval ** 2) / (ntar ** 2)
#             collection.append(result)
#         else:
#             permute_meansq(tsq, curpos + 1, curval, collection)

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

    

    # curqueryother, curkeyother, curvalueother = model.multiheadattn.qkv(context)
    # curquery, curkeyself, curvalueself = model.multiheadattn.qkv(target)
    # # attention_scores = curquery.matmul(curkeyself.transpose(2, 3)) / math.sqrt(d_model)
    # # head_output_self = attention_scores.matmul(curvalueself)

    
    # for i in range(ncont):
    #     for j in range(ntar):
    #         curselfweight = torch.exp(curquery[j, :, :, :].matmul(curkeyself[j, :, :, :].transpose(1, 2)))
    #         curotherweight =  torch.exp(curquery[j, :, :, :].matmul(curkeyother[i, :, :, :].transpose(1, 2)))

    #         curwsum = curselfweight + curotherweight
    #         transval = ((curselfweight.matmul(curvalueself[j, :, :, :]) + curotherweight.matmul(curvalueother[i, :, :, :]))/curwsum).unsqueeze(1).expand(-1,model.ncum,-1,-1)
    #         pureval = curvalueself[j, :, :, :].unsqueeze(1).expand(-1, model.ncum,-1,-1)

    #         # Expand weights to broadcast
    #         weights_expanded = model.multiheadattn.cum_weights.view(model.num_heads, model.ncum, 1, 1)
    #         weights_expanded = weights_expanded.expand_as(transval) 
            
    #         head_outputs_transval = (transval * weights_expanded).sum(dim=0)
    #         head_outputs_curvalueself = (pureval * weights_expanded).sum(dim=0)

    #         pred_transval = model.predict(head_outputs_transval.unsqueeze(0))
    #         pred_curvalueself = model.predict(head_outputs_curvalueself.unsqueeze(0))
    
    #         curdelta = (pred_transval - pred_curvalueself)[:, :, predindex].item()
    #         # curdelta = nn.MSELoss()(pred_transval[:, :, predindex], pred_curvalueself[:, :, predindex]).item()
    #         tsq[i, j] = curdelta #* curdelta
        
    #     meansq[i] = (tsq[i, :].mean())**2
    #     # meansq[i] = (tsq[i, :].mean())

    for i in range(ncont):
        curqueryother, curkeyother, curvalueother = model.multiheadattn.qkv(context[i].unsqueeze(0))
        for j in range(ntar):       
            curquery, curkeyself, curvalueself = model.multiheadattn.qkv(target[j].unsqueeze(0))
            curselfweight = torch.exp(curquery.matmul(curkeyself.transpose(2, 3)) / math.sqrt(model.d_model))
            curotherweight =  torch.exp(curquery.matmul(curkeyother.transpose(2, 3)) / math.sqrt(model.d_model))

            curwsum = curselfweight + curotherweight
            transval = ((curvalueself * curselfweight/curwsum + curvalueother * curotherweight/curwsum )).unsqueeze(2).expand(-1,-1, model.ncum,-1,-1)
            pureval = curvalueself.unsqueeze(2).expand(-1,-1, model.ncum,-1,-1)

            # Expand weights to broadcast
            weights_expanded = model.multiheadattn.cum_weights.view(1, model.num_heads, model.ncum, 1, 1)
            weights_expanded = weights_expanded.expand_as(transval) 
            
            head_outputs_transval = (transval * weights_expanded).sum(dim=1)
            head_outputs_curvalueself = (pureval * weights_expanded).sum(dim=1)

            pred_transval = model.predict(head_outputs_transval)
            pred_curvalueself = model.predict(head_outputs_curvalueself)
    
            # curdelta = torch.abs(pred_transval - pred_curvalueself)[:, :, predindex].item()
            curdelta = (pred_transval - pred_curvalueself)[:, :, predindex].item()
            
            # # curdelta = nn.MSELoss()(pred_transval[:, :, predindex], pred_curvalueself[:, :, predindex]).item()
            
            tsq[i, j] = curdelta #* curdelta
        
        meansq[i] = (tsq[i, :].mean())**2
        # meansq[i] = (tsq[i, :].mean())
    
    return meansq, tsq
            



#FIXME the model never has the possibility to learn what to predict as the second time point of sequence (seeing only one time point)


def print_p_values(avepval):
    """
    Print the p-values in a readable format.

    Parameters:
    avepval (torch.Tensor): Average p-values to print.
    """
    for i, pval in enumerate(avepval):
        print(f"Variable {i+1}: {pval:.6f}")