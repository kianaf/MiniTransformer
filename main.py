import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# torch.multiprocessing.set_start_method('spawn', force=True)
import time
import importlib
import torch.optim as optim
from torchsummary import summary


from src.data_preparation import SimulatedDataset, collate_function
from src.transformers import MiniTransformer
import src.transformers as transformerFunctions
from src.transformers import init_weights_recursive
from src.transformers import print_parameters
from src.transformers import create_custom_mask, create_distance_to_end_matrix, create_pairwise_distance_matrix
from src.evaluation import calculate_bench1_loss, calculate_bench2_loss, evaluate_mini_transformer
from src.statistical_testing import statistical_testing, print_p_values
import torch.autograd.profiler as profiler

# # Check if MPS is available on the current machine
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
#     print("Using MPS backend for GPU acceleration.")
# else:
#     device = torch.device("cpu")
#     print("MPS not available, using CPU instead.")

device = torch.device("cpu")




# Set the random seed for reproducibility
# seed = 42
# torch.manual_seed(seed)


if __name__ == '__main__':

    # Hyperparameters

    n = 200                   # sample size
    batch_size = 1          # Batch size for loading data
    p = 10                   # number of features
    mdim_head_dimension = 1 # dimension of the head
    nheads = 16             # number of heads
    ncum = 2                # number of cumulants
    maxlen = 10             # maximum length of the sequence
    learning_rate = 5e-4
    lambda_l2 = 1e-3
    EPOCHS = 150
    mask = create_custom_mask(maxlen, device)
    distance_to_end_matrix = create_distance_to_end_matrix(maxlen, device)
    pairwise_distance_matrix = create_pairwise_distance_matrix(maxlen, device)

    # Create Dataset and DataLoader
    train_dataset = SimulatedDataset(n, p, maxlen=maxlen, device = device)
    
    

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_function,  num_workers=0)
    
    model = MiniTransformer(p, nheads, mdim_head_dimension, ncum, mask, distance_to_end_matrix, pairwise_distance_matrix, device)

    model.apply(init_weights_recursive)
    model.to(device)

    model = torch.compile(model)
    # Start the timer
    start_time = time.time()

    # Define optimizer
    # optimizer = optim.Adam(model.parameters(), lr= learning_rate, weight_decay=lambda_l2)
    optimizer = optim.Adam(model.parameters(), lr= learning_rate, weight_decay=lambda_l2)
    # optimizer = optim.Adam(model.parameters(), lr= learning_rate)
    
    transformerFunctions.train_mini_transformer(model, dataloader, optimizer, lambda_l2, EPOCHS, device)


    # # Enable profiling
    # with profiler.profile() as prof:

    #     model.eval() 
    #     # Forward pass
    #     output = model(train_dataset.data)
    #     # Backward pass (this will profile the backward pass as well)
    #     output.backward(torch.ones_like(output))

    # # Print profiling results
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))



    # End the timer
    end_time = time.time()

    # Calculate and print the execution time
    execution_time = end_time - start_time 
    print(f"Execution time: {execution_time:.6f} seconds")

    # reload transformer.py
    importlib.reload(transformerFunctions)
    print(transformerFunctions.count_parameters(model))

    eval_dataset = SimulatedDataset(1000, p, maxlen=4, device = device)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=True, collate_fn=collate_function, num_workers=0)

    dimave, bench1loss = calculate_bench1_loss(train_dataset.data, eval_dataset.data)
    bench2loss = calculate_bench2_loss(train_dataset.data, eval_dataloader.dataset, dimave)
    model_loss = evaluate_mini_transformer(eval_dataloader, model)

    print("bench1loss ", bench1loss)
    print("bench2loss", bench2loss)
    print("model_loss", model_loss.item()) 


    # print_parameters(model)


    # # Set the specific values for each row in the second dimension
    # tensor[0, 0, :3] = torch.tensor([0, 1, 0])  # First row with specified values
    # tensor[0, 1, :3] = torch.tensor([1, 0, 0])  # Second row with specified values

    # print(model((tensor, torch.ones(1, 2, 10)))[0, 1, 2])

    # Initialize a tensor of shape 1 x 2 x 10 with zeros
    # tensor = torch.zeros(1, 2, 10)

    target_sample_size = 7
    nrepp = 10
    predindex = 2


    avepval = statistical_testing(model, p, predindex, nrepp, target_sample_size)
    print_p_values(avepval)






# 10 variables
# startvector = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]