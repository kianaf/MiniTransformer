import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# torch.multiprocessing.set_start_method('spawn', force=True)
import time
import importlib
import torch.optim as optim


from src.data_preparation import SimulatedDataset
from src.transformers import MiniTransformer
import src.transformers as transformerFunctions
from src.transformers import init_weights_recursive
from src.evaluation import calculate_bench1_loss, calculate_bench2_loss, evaluate_mini_transformer


# # Check if MPS is available on the current machine
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
#     print("Using MPS backend for GPU acceleration.")
# else:
#     device = torch.device("cpu")
#     print("MPS not available, using CPU instead.")

device = torch.device("cpu")

# Start the timer
start_time = time.time()


# Set the random seed for reproducibility
seed = 11
torch.manual_seed(seed)


# def linear_julia_style(input, weight, bias=None):
#     if input.dim() == 2 and bias is not None:
#         # fused op is marginally faster
#         return torch.addmm(bias, input, weight.t())
#     output = (weight.t().matmul(input.t())).t()
#     if bias is not None:
#         output += bias
#     return output




if __name__ == '__main__':

    # Hyperparameters
    n = 200  # sample size
    batch_size = 1  # Batch size for loading data
    p = 3 # number of features
    mdim_head_dimension = 1 # dimension of the head
    nheads = 16 # number of heads
    ncum = 2 # number of cumulants
    maxlen = 10 # maximum length of the sequence
    # cumveclen = ncum * (maxlen-2) # length of the cumulative vector

    learning_rate = 1e-2
    lambda_l2 = 0.001
    EPOCHS = 200


    # Create Dataset and DataLoader
    train_dataset = SimulatedDataset(n, maxlen=maxlen, device = device)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    
    model = MiniTransformer(p, nheads, mdim_head_dimension, ncum, device)

    # model.apply(init_weights_recursive)
    model.to(device)


    # Define optimizer
    # optimizer = optim.Adam(model.parameters(), lr= learning_rate)
    optimizer = optim.SGD(model.parameters(), lr= learning_rate, weight_decay=lambda_l2)
    # optimizer = optim.SGD(model.parameters(), lr= learning_rate)
    transformerFunctions.train_mini_transformer(model, dataloader, optimizer, lambda_l2, EPOCHS, device)


    # End the timer
    end_time = time.time()

    # Calculate and print the execution time
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.6f} seconds")

    # reload transformer.py
    # importlib.reload(transformerFunctions)
    print(transformerFunctions.count_parameters(model))

    eval_dataset = SimulatedDataset(1000, maxlen=4, device = device)

    dimave, bench1loss = calculate_bench1_loss(train_dataset.data, eval_dataset.data)
    bench2loss = calculate_bench2_loss(train_dataset.data, eval_dataset.data, dimave)
    model_loss = evaluate_mini_transformer(eval_dataset.data, model)

    print("bench1loss ", bench1loss)
    print("bench2loss", bench2loss)
    print("model_loss", model_loss.item()) 
