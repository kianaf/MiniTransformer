import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# torch.multiprocessing.set_start_method('spawn', force=True)
import time
import importlib
import torch.optim as optim
from torchsummary import summary


from src.data_preparation import SimulatedDataset, collate_function, load_real_data
from src.transformers import MiniTransformer
import src.transformers as transformerFunctions
from src.transformers import init_weights_recursive
from src.transformers import print_parameters
from src.transformers import create_custom_mask_pred, create_custom_mask_pair, create_distance_to_end_matrix, create_pairwise_distance_matrix
from src.evaluation import calculate_bench1_loss, calculate_bench2_loss, calculate_repeat_loss, calculate_regression_loss, evaluate_mini_transformer
from src.statistical_testing import statistical_testing, print_p_values, plot_context_predindex_pair_effect, get_context_predindex_pair_effect
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
    # data_str = "ghq_b_sum"
    # data_str = "ghq_sum"
    data_str = "simulation"
    batch_size = 1          # Batch size for loading data
    dk = 1                  # d_k
    dv = 1                  # d_v
    nheads = 16             # number of heads
    ncum = 2                 # number of cumulants
    maxlen = 10             # maximum length of the sequence
    learning_rate = 1e-3
    lambda_l2 = 1e-3
    EPOCHS = 100
    target_sample_size = 7
    nrepp = 10

    # Set the random seed for reproducibility
    # torch.manual_seed(42)
    
    if data_str == "simulation":
        n = 200
        p = 10
        maxlen = 10
        # Create Dataset and DataLoader
        train_dataset = SimulatedDataset(n, p, maxlen=maxlen, device = device).data
        eval_dataset = SimulatedDataset(1000, p, maxlen=4, device = device).data
        predindex = 2
        
    else:
        # load real data
        data, maxlen = load_real_data(data_str)
        
        n = int(0.75*len(data))
        p = data[0].shape[1]
        print("Sample size: ", len(data))
        
        train_dataset = data[:n]
        eval_dataset = data[n:]
        predindex = 9

    
    mask_pairwise = create_custom_mask_pair(maxlen, device)
    mask_pred = create_custom_mask_pred(maxlen, device)
    distance_to_end_matrix = create_distance_to_end_matrix(maxlen, device)
    pairwise_distance_matrix = create_pairwise_distance_matrix(maxlen, device)

   

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_function,  num_workers=0)
    
    model = MiniTransformer(p, nheads, dk, dv, ncum, mask_pairwise, mask_pred, pairwise_distance_matrix, distance_to_end_matrix,  device)

    model.apply(init_weights_recursive)
    model.to(device)

    # model = torch.compile(model)
    # Start the timer
    start_time = time.time()

    # Define optimizer
    # optimizer = optim.Adam(model.parameters(), lr= learning_rate, weight_decay=lambda_l2)
    optimizer = optim.Adam(model.parameters(), lr= learning_rate)
    
    print("Number of Parameters", transformerFunctions.count_parameters(model))
    
    run_path = transformerFunctions.train_mini_transformer(model, dataloader, optimizer, lambda_l2, EPOCHS, device)


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
    
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=True, collate_fn=collate_function, num_workers=0)

    
    dimave, bench1loss, benchloss_predindex = calculate_bench1_loss(train_dataset, eval_dataset, predindex)
    regression_loss_predindex, regression_loss_total = calculate_regression_loss(train_dataset, eval_dataset, predindex)
    model_loss_predindex, model_loss_total = evaluate_mini_transformer(eval_dataloader, model, predindex)
    
    
    print("baseline average loss: ", bench1loss)

    
    # Evaluate the model
    if data_str == "simulation":
        bench2loss, bench2loss_predindex = calculate_bench2_loss(train_dataset, eval_dataset, dimave)
        print("baseline informed loss: ", bench2loss)
        
    else:
        bench_repeat, bench_repeat_predindex = calculate_repeat_loss(eval_dataset, predindex)    
        print("baseline repeat: ", bench_repeat)
        
        
    print("baseline average loss predindex: ", benchloss_predindex)
      
    print("model loss total: ", model_loss_total.item(), "\n") 
    
    
    if data_str == "simulation":
        print("baseline informed loss predindex: ", bench2loss_predindex)
    else:
        print("baseline repeat predindex: ", bench_repeat_predindex)  
    print("regression loss predindex: ", regression_loss_predindex)
    print("model loss predindex: ", model_loss_predindex.item()) 


    # print_parameters(model)
    # # Set the specific values for each row in the second dimension
    # tensor[0, 0, :3] = torch.tensor([0, 1, 0])  # First row with specified values
    # tensor[0, 1, :3] = torch.tensor([1, 0, 0])  # Second row with specified values

    # print(model((tensor, torch.ones(1, 2, 10)))[0, 1, 2])

    # Initialize a tensor of shape 1 x 2 x 10 with zeros
    # tensor = torch.zeros(1, 2, 10)
    
    
    
    avepval, stdpval, context, targetall = statistical_testing(model, train_dataset, p, predindex, nrepp, target_sample_size)
    
    print_p_values(avepval, stdpval)

    context_predindex_pair_effect = get_context_predindex_pair_effect(model, p, context, targetall, nrepp)

    plot_context_predindex_pair_effect(context_predindex_pair_effect, data_str, run_path)
