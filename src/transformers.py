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
import time


def create_custom_mask_pred(seq_len, device):
    """Creates a custom prediction mask for transformer attention.
    
    Generates a triangular mask matrix where elements after the first diagonal are masked
    with a large negative value (-1e9) to prevent attention to future positions.
    
    Args:
        seq_len (int): Length of the sequence.
        device (torch.device): Device to create the mask on (CPU/GPU).
    
    Returns:
        torch.Tensor: A (seq_len, seq_len) mask matrix where masked positions contain -1e9.
    """
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
    """Creates a custom pairwise mask for transformer attention.
    
    Generates an upper triangular mask matrix where elements above the diagonal are masked
    with a large negative value (-1e9) to prevent attention to future positions.
    
    Args:
        seq_len (int): Length of the sequence.
        device (torch.device): Device to create the mask on (CPU/GPU).
    
    Returns:
        torch.Tensor: A (seq_len, seq_len) mask matrix where masked positions contain -1e9.
    """
    matrix = torch.zeros((seq_len, seq_len), device=device)
    
    # Fill all the elements after that diagonal with ones
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            matrix[i, j] = 1
    
    return matrix * -1e9


def create_distance_to_end_matrix(seq_len, device):
    """Creates a matrix representing distances to the end of sequences.
    
    For each position (i,j) in the matrix, calculates the distance from position i
    to position j, with special handling for positions beyond i+1.
    
    Args:
        seq_len (int): Length of the sequence.
        device (torch.device): Device to create the matrix on (CPU/GPU).
    
    Returns:
        torch.Tensor: A (seq_len, seq_len) matrix containing distances, with positions
                     beyond i+1 filled with 1e9.
    """
    matrix = torch.ones((seq_len, seq_len),  device=device) * 1e9
    
    # Fill the diagonal after the main diagonal with zeros
    for i in range(seq_len):
        for j in range(i+1):
            matrix[i, j] = i - j +1
    
    
    return matrix



def new_weird_oh_my_god_pred_distance_matrix(seq_len, device):  
    matrix = torch.arange(seq_len, device=device).expand(seq_len, seq_len)
    
    return matrix
    


def create_pairwise_distance_matrix(seq_len, device):
    """Creates a matrix of pairwise distances between sequence positions.
    
    Computes the absolute difference |i - j| between all pairs of positions in the sequence.
    
    Args:
        seq_len (int): Length of the sequence.
        device (torch.device): Device to create the matrix on (CPU/GPU).
    
    Returns:
        torch.Tensor: A (seq_len, seq_len) matrix containing absolute distances between positions.
    """
    indices = torch.arange(seq_len, device=device)
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

    for name, param in module.named_parameters():
        if param.requires_grad:
            with torch.no_grad():
                if method == "uniform":
                    # torch.random.manual_seed(time.time())
                    torch.nn.init.uniform_(param, *init_range)
                    
                    # param.copy_(torch.rand(param.size()) * 0.2 - 0.1)
                    
                    # if "bias" in name:
                    #     param.fill_(0.06)
                    # else:
                    #     param.fill_(0.1)
                    # if ("W_b_v" in name) & ("weight" in name):
                    #     param[:, 0].fill_(0.5)
                    #     param[:, 3].fill_(-0.05)
                        
                    
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
    """Multi-head attention mechanism with custom distance-based attention weights.
    
    This implementation extends the standard transformer multi-head attention by incorporating
    distance-based attention weights and cumulative attention mechanisms. It includes both
    pairwise distance considerations and distance-to-end weighting in the attention computation.
    
    Args:
        d_model (int): The dimension of the model's hidden state
        num_heads (int): Number of attention heads
        dk (int): Dimension of the key vectors
        dv (int): Dimension of the value vectors
        ncum (int): Number of cumulative attention outputs
        mask_pairwise (torch.Tensor): Mask for pairwise attention
        pairwise_distance_matrix (torch.Tensor): Matrix of pairwise distances between positions
        distance_to_end_matrix (torch.Tensor): Matrix of distances to sequence end
        device (torch.device): Device to use for computations
    """
    def __init__(self, d_model, num_heads, dk, dv, ncum, mask_pairwise, pairwise_distance_matrix,  distance_to_end_matrix, device):
        super().__init__()
        
        self.device = device
        self.d_model = d_model
        self.num_heads = num_heads
        self.ncum = ncum
        self.dk = dk
        self.dv = dv
        self.cum_weights = nn.Linear(num_heads, ncum, bias = False)
        self.mask_pairwise = mask_pairwise

        self.pairwise_distance_matrix = pairwise_distance_matrix
        self.distance_to_end_matrix = distance_to_end_matrix

        # Weight for adjusting the effect of distance to the end
        self.distance_to_end_weight = nn.Parameter(torch.randn(1, 1))
        torch.nn.init.uniform_(self.distance_to_end_weight, -0.1, 0.1)

        # Weight for adjusting the effect of pairwise distances
        self.distance_between_two_positions_weight = nn.Parameter(torch.randn(1, 1))
        torch.nn.init.uniform_(self.distance_between_two_positions_weight, -1, 1)
        
        # Linear projections for Q, K, V
        self.W_b_q = nn.Linear(d_model, self.num_heads * self.dk)
        self.W_b_k = nn.Linear(d_model, self.num_heads * self.dk)
        self.W_b_v = nn.Linear(d_model, self.num_heads * self.dv)

        
        # self.act_function = nn.ReLU()
        self.act_function = nn.Identity()

    
    def exponential_decay_pred(self, dist, weight):
        """Applies exponential decay to prediction distances.
        
        Args:
            dist (torch.Tensor): Distance matrix
            weight (float): Weight parameter for scaling the decay
            
        Returns:
            torch.Tensor: Exponentially decayed distances
        """
        # return torch.softmax((-dist * torch.exp(weight)**5), dim = -1)
        return torch.exp((-dist * torch.exp(weight))**5)

    def exponential_decay_pair(self, dist, weight):
        """Applies exponential decay to pairwise distances.
        
        Args:
            dist (torch.Tensor): Distance matrix
            weight (float): Weight parameter for scaling the decay
            
        Returns:
            torch.Tensor: Exponentially decayed distances
        """
        return (-dist * torch.exp(weight))**5

    def activation(self, projection_layer, x):
        """Applies activation function to projected values.
        
        Args:
            projection_layer (nn.Module): Linear projection layer
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Activated projection
        """
        return self.act_function(projection_layer(x))

    def split_heads(self, x, seq_len, dim_head):
        """Splits the input tensor into multiple attention heads.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, num_heads * dim_head]
            seq_len (int): Length of the sequence
            dim_head (int): Dimension of each head
            
        Returns:
            torch.Tensor: Reshaped tensor of shape [batch_size, num_heads, seq_len, dim_head]
        """
        return x.view(x.shape[0], seq_len, self.num_heads, dim_head).transpose(1, 2)

    def qkv(self, x):
        """Computes query, key, and value projections and splits them into heads.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            tuple: (Q, K, V) tensors after projection and head splitting
        """
        seq_len = x.shape[1]

        Q = self.activation(self.W_b_q, x)
        K = self.activation(self.W_b_k, x)
        V = self.activation(self.W_b_v, x)

        Q = self.split_heads(Q, seq_len, self.dk)
        K = self.split_heads(K, seq_len, self.dk)
        V = self.split_heads(V, seq_len, self.dv)

        return Q, K, V

    def get_attention(self, x, dist_weight, mask=None, padding_mask=None):
        """Computes attention scores with distance weighting and masking.
        
        Args:
            x (torch.Tensor): Input tensor
            dist_weight (torch.Tensor): Distance-based attention weights
            mask (torch.Tensor, optional): Attention mask
            padding_mask (torch.Tensor, optional): Mask for padded positions
            
        Returns:
            tuple: (attention_scores, V) - Computed attention scores and value tensors
        """
        batch_size, seq_len, _ = x.shape

        Q, K, V = self.qkv(x)
        
        scores = Q.matmul(K.transpose(2, 3)) / math.sqrt(self.dk)
        scores = scores.masked_fill((padding_mask[:, :, 0].unsqueeze(1).unsqueeze(-1).expand(batch_size, self.num_heads, seq_len, seq_len)) !=True, -1e9)
        attention_scores = torch.nn.functional.softmax((scores + dist_weight + mask)[:, :, 1:seq_len, :seq_len], dim=-1)
        
        return attention_scores, V

    def forward(self, data):
        """Forward pass of the multi-head attention module.
        
        Args:
            data (tuple): Tuple containing:
                - x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model]
                - padding_mask (torch.Tensor): Mask for padded positions
                
        Returns:
            torch.Tensor: Processed attention outputs after cumulative weighting
        """
        x = data[0]
        padding_mask = data[1]
        
        batch_size, seq_len, _ = x.size()
        
        mask_pairwise = self.mask_pairwise[:seq_len, :seq_len].expand(batch_size, self.num_heads, seq_len, seq_len)
        attention_scores, V = self.get_attention(
            x, 
            self.exponential_decay_pair(
                self.pairwise_distance_matrix[:seq_len, :seq_len], 
                self.distance_between_two_positions_weight[0,0]
            ).expand(batch_size, self.num_heads, seq_len, seq_len),
            mask_pairwise, 
            padding_mask
        )
        
        head_output = (attention_scores.matmul(V)).transpose(1,2).squeeze(dim=-1)
        
        # pooling_weights = self.exponential_decay_pred(torch.flip(self.distance_to_end_matrix[:seq_len-1, :seq_len-1], dims = [1]) , self.distance_to_end_weight[0,0])* torch.exp(self.mask_pairwise[:seq_len-1, :seq_len-1])
        
        pooling_weights = self.exponential_decay_pred(
            self.distance_to_end_matrix[:seq_len-1, :seq_len-1],
            self.distance_to_end_weight[0,0]
        )
        head_output_weighted_sum_pool = pooling_weights.matmul(head_output)
        head_outputs_cum = self.cum_weights(head_output_weighted_sum_pool)
    
        return head_outputs_cum


def l2_penalty_params_except_bias(model, lambda_l2):
    """Computes L2 regularization penalty for all model parameters except biases.
    
    Args:
        model (nn.Module): The model whose parameters to compute L2 penalty for
        lambda_l2 (float): L2 regularization coefficient
        
    Returns:
        torch.Tensor: The computed L2 penalty
    """
    # Get only the weight parameters (exclude biases)
    penalty = 0.0
    for name, param in model.named_parameters(): 
        
        # if "bias" not in name and "distance" not in name:
        if "bias" not in name:
            penalty += torch.sum(torch.pow(param, 2))
            
    return lambda_l2 * penalty


def mini_transformer_loss(output, target, padded_masks):    
    
    # loss = torch.sum(((output[:, :-1, :] - (target[:, 2:, :])) * padded_masks[:, 2:, :]) **2) / output.shape[0]

    loss = torch.sum(((output - target[:, 2:, :]) * padded_masks[:, 2:, :]) **2) / output.shape[0]


    return loss
 

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
        
        # self.prediction = nn.Sequential(nn.Linear(self.ncum, self.d_model), nn.Sigmoid())
        self.prediction = nn.Linear(self.ncum, self.d_model)

        
    def forward(self, data):
        x = data[0]
        padded_masks = data[1]
        out = self.multiheadattn(data)  # The last row is the label

        pred = self.prediction(out) #* padded_masks

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

        
        sum_square_error = mini_transformer_loss(output, data[0], data[1])
        penalty = l2_penalty_params_except_bias(model, lambda_l2)
        loss = sum_square_error + penalty
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().cpu().item() # without .cpu Might crash if loss is on GPU/MPS

        
    return running_loss/len(train_loader), penalty

def train_mini_transformer(model, train_loader, eval_loader, optimizer, lambda_l2, EPOCHS, device):
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_path = 'runs/trainer_{}'.format(timestamp)
    tb_writer = SummaryWriter('runs/trainer_{}'.format(timestamp))


    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    best_vloss = 1_000_000.
    for epoch_number in range(EPOCHS):
        
        
        avg_loss, penalty  = train_mini_transformer_one_epoch(model, train_loader, optimizer, lambda_l2, device)

        
        if eval_loader is not None:
            
            model.eval()
            
            eval_data = next(iter(eval_loader))
            
            output_eval = model((eval_data[0][:,:-1,:] , eval_data[1][:,:-1,:]))    
            
            sum_square_error = mini_transformer_loss(output_eval, eval_data[0], eval_data[1]).detach().cpu().item()
            penalty_eval = l2_penalty_params_except_bias(model, lambda_l2)
            loss_eval = sum_square_error #+ penalty_eval
            
            model.train(True)
        
        
        if epoch_number % 10 == 0:
            print('EPOCH {}:'.format(epoch_number + 1))
            print('avg_loss:     {:.5f}'.format(avg_loss), 'penalty    : {:.5f}'.format(penalty))
            if eval_loader is not None:
                print('avg_loss_val: {:.5f}'.format(loss_eval), 'penalty_val: {:.5f}'.format(penalty_eval))

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