import torch
import torch.nn as nn 
import math
from torch.nn.parallel import parallel_apply
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import wandb
from torchviz import make_dot
import matplotlib.pyplot as plt



# Create a triangular matrix as per the description
def create_custom_mask_pred(seq_len, device):
    matrix = torch.zeros((seq_len, seq_len), device=device)
    
    # Fill the diagonal after the main diagonal with zeros
    for i in range(seq_len - 1):
        matrix[i, i+1] = 0
    
    # Fill all the elements after that diagonal with ones
    for i in range(seq_len):
        for j in range(i + 2, seq_len):
            matrix[i, j] = 1
    
    return matrix * -1e9


def create_custom_mask_pair(seq_len, device):
    matrix = torch.zeros((seq_len, seq_len), device=device)
    
    # Fill the diagonal after the main diagonal with zeros
    for i in range(seq_len - 1):
        matrix[i, i+1] = 0
    
    # Fill all the elements after that diagonal with ones
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            matrix[i, j] = 1
    
    return matrix * -1e9



# def create_distance_to_end_matrix(seq_len, device):
#     matrix = torch.ones((seq_len, seq_len),  device=device) * 1e9
    
#     # Fill the diagonal after the main diagonal with zeros
#     for i in range(seq_len - 1):
#         for j in range(i+2):
#             matrix[i, j] = i +1 - j
    
#     return matrix



def create_distance_to_end_matrix(seq_len, device):
    matrix = torch.ones((seq_len, seq_len),  device=device) * 1e9
    
    # Fill the diagonal after the main diagonal with zeros
    for i in range(seq_len):
        for j in range(i+1):
            matrix[i, j] = i - j 
            # matrix[i, j] = 0
    
    
    return matrix



def new_weird_oh_my_god_pred_distance_matrix(seq_len, device):  
    matrix = torch.arange(seq_len, device=device).expand(seq_len, seq_len)
    
    # now mask
    
    # for i in range(seq_len):
    #     for j in range(i+1, seq_len):
    #         matrix[i, j] = 0
    
    return matrix
    


def create_pairwise_distance_matrix(seq_len, device):
    
    # dist_matrix = torch.zeros((seq_len, seq_len), device=device)
    # Create an index matrix using torch.arange
    indices = torch.arange(seq_len, device=device)

    # Create the matrix with |i - j|
    dist_matrix = torch.abs(indices[:, None] - indices[None, :])
    
    return dist_matrix



def init_weights_recursive(module, method="uniform", init_range=(-0.1, 0.1), initialized=set()):
    """
    Recursively initializes weights for all trainable parameters in a PyTorch module, avoiding duplicate initialization.
    
    - Uses `id(module)` to track initialized modules and prevent re-initialization.
    - Supports different initialization methods (`uniform`, `normal`, `xavier`, `kaiming`).
    
    Args:
        module (nn.Module): The module to initialize.
        method (str): Initialization method. Options: "uniform", "normal", "xavier", "kaiming".
        init_range (tuple): Range for uniform initialization (default: [-0.1, 0.1]).
        initialized (set): Set to track initialized module IDs.
    """
    if id(module) in initialized:
        return  # Prevent re-initializing the same module

    initialized.add(id(module))  # Mark this module as initialized

    for name, param in module.named_parameters(recurse = False):
        if param.requires_grad:
            with torch.no_grad():
                if method == "uniform":
                    torch.nn.init.uniform_(param, *init_range)
                    # if "bias" in name:
                    #     param.fill_(0.05)
                    # else:
                    #     param.fill_(-0.01)
                elif method == "normal":
                    torch.nn.init.normal_(param, mean=0, std=0.05)
                elif method == "xavier":
                    torch.nn.init.xavier_uniform_(param)
                elif method == "kaiming":
                    torch.nn.init.kaiming_uniform_(param, nonlinearity='relu')

            print(f"Initialized: {name} in {module.__class__.__name__} using {method}")

    # Recursively apply to child modules
    for child in module.children():
        init_weights_recursive(child, method, init_range, initialized)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dk, dv, ncum, mask_pairwise, pairwise_distance_matrix,  distance_to_end_matrix, device):
        super().__init__()
        
        self.device = device
        self.d_model = d_model
        self.num_heads = num_heads
        self.ncum = ncum
        self.dk = dk
        self.dv = dv
        # self.cum_weights = nn.Parameter(torch.randn(num_heads, ncum))
        self.cum_weights = nn.Linear(num_heads, ncum, bias = False)
        self.mask_pairwise = mask_pairwise

        self.pairwise_distance_matrix = pairwise_distance_matrix
        self.distance_to_end_matrix = distance_to_end_matrix


        # This is the weight that is used to change the slope of the effect of the distance to the end
        self.distance_to_end_weight = nn.Parameter(torch.randn(1, 1))


        # This is the weight that is used to change the slope of the effect of the distance between two positions
        self.distance_between_two_positions_weight = nn.Parameter(torch.randn(1, 1))
        
        # Create linear layers for all heads then we can split if we want
        self.W_b_q = nn.Linear(d_model, self.num_heads * self.dk)
        self.W_b_k = nn.Linear(d_model, self.num_heads * self.dk)
        self.W_b_v = nn.Linear(d_model, self.num_heads * self.dv)

        
        # self.act_function = nn.ReLU()
        self.act_function = nn.Identity()

    
    def exponential_decay_pred(self, dist, weight):
        # return torch.softmax((-dist * torch.exp(weight)**5), dim = -1)
        return torch.exp((-dist * torch.exp(weight))**5)
        # return torch.exp((-dist * torch.exp(torch.tensor([0.5])))**5)


    def exponential_decay_pair(self, dist, weight):
        return (-dist * torch.exp(weight))**5


    def activation(self, projection_layer, x):
        return self.act_function(projection_layer(x))

    def split_heads(self, x, seq_len, dim_head):
        # Reshape the input to have num_heads for multi-head attention
        return x.view(x.shape[0], seq_len, self.num_heads, dim_head).transpose(1, 2)

    def qkv(self, x):
        
        seq_len = x.shape[1]

        Q = self.activation(self.W_b_q, x)
        K = self.activation(self.W_b_k, x)
        V = self.activation(self.W_b_v, x)


        # split heads
        Q = self.split_heads(Q, seq_len, self.dk)
        K = self.split_heads(K, seq_len, self.dk)
        V = self.split_heads(V, seq_len, self.dv)

        return Q, K, V
    
        
    def get_attention(self, x, dist_weight, mask=None, padding_mask=None):
        
        batch_size, seq_len, _ = x.shape

        Q, K, V = self.qkv(x)
        
        scores = Q.matmul(K.transpose(2, 3)) / math.sqrt(self.dk)

        scores = scores.masked_fill((padding_mask[:, :, 0].unsqueeze(1).unsqueeze(-1).expand(batch_size, self.num_heads, seq_len, seq_len)) !=True, -1e9)

        attention_scores = torch.nn.functional.softmax((scores + dist_weight + mask)[:, :, 1:seq_len, :seq_len], dim=-1)
        
        return attention_scores, V
    
    
    def forward(self, data):

        # print("X shape ", x.shape)
        x = data[0]
        
        padding_mask = data[1]
        
        batch_size, seq_len, _ = x.size()
        
        mask_pairwise = self.mask_pairwise[:seq_len, :seq_len].expand(batch_size, self.num_heads, seq_len, seq_len)

        attention_scores, V = self.get_attention(x, self.exponential_decay_pair(self.pairwise_distance_matrix[:seq_len, :seq_len], self.distance_between_two_positions_weight[0,0]).expand(batch_size, self.num_heads, seq_len, seq_len), mask_pairwise, padding_mask)
        
        head_output = (attention_scores.matmul(V)).transpose(1,2).squeeze(dim=-1)
        
        pooling_weights = self.exponential_decay_pred(torch.flip(self.distance_to_end_matrix[:seq_len-1, :seq_len-1], dims = [1]) , self.distance_to_end_weight[0,0])* torch.exp(self.mask_pairwise[:seq_len-1, :seq_len-1])
        
        head_output_weighted_sum_pool = pooling_weights.matmul(head_output)
        
        head_outputs_cum = self.cum_weights(head_output_weighted_sum_pool)
        
       
       
       
       ###something###
        # pooling_weights = self.exponential_decay_pred(torch.flip(self.distance_to_end_matrix[:seq_len-1], dims = [0]), self.distance_to_end_weight[0,0])
   
        # head_output = ((attention_scores.matmul(V))*pooling_weights.unsqueeze(-1)).transpose(1,2).squeeze(dim=-1)
         
        # head_outputs_cum = self.cum_weights(head_output)
         
        return head_outputs_cum
 

def l2_penalty_params_except_bias(model, lambda_l2):
    # Get only the weight parameters (exclude biases)
    penalty = 0.0
    for name, param in model.named_parameters(): 
        
        # if "bias" not in name and "distance" not in name:
        if "bias" not in name:
            penalty += torch.sum(torch.pow(param, 2))
            
    return lambda_l2 * penalty


def mini_transformer_loss(output, target, padded_masks):    
    
    # running_loss = torch.sum(((output[:, :-1, :] - (target[:, 2:, :])) * padded_masks[:, 2:, :]) **2) / output.shape[0]

    running_loss = torch.sum(((output - (target[:, 2:, :])) * padded_masks[:, 2:, :]) **2) / output.shape[0]


    return running_loss
 

class MiniTransformer(nn.Module):
    def __init__(self, d_model, num_heads, dk, dv, ncum, mask_pairwise, pairwise_distance_matrix,  distance_to_end_matrix, device):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.ncum = ncum
        self.dk = dk
        self.dv = dv
        self.device = device
        self.multiheadattn = MultiHeadAttention(d_model, num_heads, dk, dv, ncum, mask_pairwise, pairwise_distance_matrix,  distance_to_end_matrix,  self.device)
        
        self.prediction = nn.Linear(self.ncum, self.d_model)

    def predict(self, out):

        pred = self.prediction(out)
        
        return pred

        
    def forward(self, data):
        x = data[0]
        padded_masks = data[1]
        out = self.multiheadattn(data)  # The last row is the label

        pred = self.predict(out) #* padded_masks

        return pred



# define training function for MiniTransformer with tensorboard logging
def train_mini_transformer_one_epoch(model, train_loader, optimizer, lambda_l2, device):
    
    running_loss = 0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_loader, 0):
        optimizer.zero_grad()

        # Move inputs to the device (MPS or CPU)
        output = model((data[0][:,:-1,:], data[1][:,:-1,:]))

        # dot = make_dot(output, params=dict(model.named_parameters()))
        # dot.graph_attr.update({'size': '100,100!'})  # Increase canvas size
        # dot.graph_attr.update({'ratio': 'compress'})  # Compress the graph
        # dot.render("model_graph", format="pdf")
        # del dot 


        # tb_writer.add_graph(model, data)
        # tb_writer.close()

        
        loss = mini_transformer_loss(output, data[0], data[1]) + l2_penalty_params_except_bias(model, lambda_l2)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    return running_loss/len(train_loader)

def train_mini_transformer(model, train_loader, eval_loader, optimizer, lambda_l2, EPOCHS, device):
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_path = 'runs/trainer_{}'.format(timestamp)
    tb_writer = SummaryWriter('runs/trainer_{}'.format(timestamp))


    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    best_vloss = 1_000_000.
    for epoch_number in range(EPOCHS):
        
        
        avg_loss = train_mini_transformer_one_epoch(model, train_loader, optimizer, lambda_l2, device)

        
        if eval_loader is not None:
            
            model.eval()
            
            eval_data = next(iter(eval_loader))
            
            output_eval = model(eval_data)
            
            loss_eval = mini_transformer_loss(output_eval, eval_data[0][:,:-1,:], eval_data[1][:,:-1,:]).item() + l2_penalty_params_except_bias(model, lambda_l2)
            
            model.train(True)
        
        
        if epoch_number % 10 == 0:
            print('EPOCH {}:'.format(epoch_number + 1))
            print('avg_loss: {:.5f}'.format(avg_loss))
            if eval_loader is not None:
                print('avg_loss_val: {:.5f}'.format(loss_eval))

        # # Log model parameters (weights and biases) after every epoch
        # for name, param in model.named_parameters():
        #     if "W_b" in name:
        #         try:
        #             tb_writer.add_histogram(f"{name}[0]", param[:, 0], epoch_number)
        #             tb_writer.add_histogram(f"{name}[1]", param[:, 1], epoch_number)
        #             tb_writer.add_histogram(f"{name}[2]", param[:, 2], epoch_number)
        #             if param.grad is not None: # Log gradients
        #                 tb_writer.add_histogram(f"{name}[0].grads", param.grad[:,0], epoch_number)  
        #                 tb_writer.add_histogram(f"{name}[1].grads", param.grad[:,1], epoch_number)
        #                 tb_writer.add_histogram(f"{name}[2].grads", param.grad[:,2], epoch_number)
        #         except:
        #             tb_writer.add_histogram(f"{name}", param, epoch_number)
        #             if param.grad is not None:
        #                 tb_writer.add_histogram(f"{name}.grads", param.grad, epoch_number)  
        #     else:
        #         tb_writer.add_histogram(f"{name}", param, epoch_number)         # Log weights
        #         if param.grad is not None:
        #             tb_writer.add_histogram(f"{name}.grads", param.grad, epoch_number)

        

        # # Log average loss for the epoch
        # tb_writer.add_scalar('avg_loss', avg_loss, epoch_number)
    return run_path

        
        

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def print_parameters(model):
    # print multihead parameters
    for head in range(model.num_heads):
        print("Head ", head+1, ":\n")
        for dim in range(model.dk):
            print("\t Q weights: ", model.multiheadattn.W_b_q.weight[head*model.dk + dim, :].data, ",\t b: ", model.multiheadattn.W_b_q.bias[head*model.dk + dim].data, "\n")

            print("\t K weights: ", model.multiheadattn.W_b_k.weight[head*model.dk + dim, :].data, "\t b: ", model.multiheadattn.W_b_k.bias[head*model.dk + dim].data, "\n")

    for dim in range(model.dv):
            print("\t V weights: ", model.multiheadattn.W_b_v.weight[head*model.dv + dim, :].data, "\t b: ", model.multiheadattn.W_b_v.bias[head*model.dv + dim].data, "\n\n")
            print("\t cum weights: ", model.multiheadattn.cum_weights.weight[:, head].data, "\n\n")

    print("Prediction weights: \n", model.prediction.weight.data, "\n")

            


#FIXME for performance on CPU
# performance viewer on google chrome. 




def plot_matrix(matrix):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Value')
    plt.title('Heatmap')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.show()