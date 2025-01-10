import torch
import torch.nn as nn 
import math
from torch.nn.parallel import parallel_apply
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import wandb
from torchviz import make_dot


# Create a triangular matrix as per the description
def create_custom_mask(seq_len, device):
    matrix = torch.zeros((seq_len, seq_len), device=device)
    
    # Fill the diagonal after the main diagonal with zeros
    for i in range(seq_len - 1):
        matrix[i, i+1] = 0
    
    # Fill all the elements after that diagonal with ones
    for i in range(seq_len):
        for j in range(i + 2, seq_len):
            matrix[i, j] = 1
    
    return matrix * -1e9



def create_distance_to_end_matrix(seq_len, device):
    matrix = torch.zeros((seq_len, seq_len),  device=device)
    
    # Fill the diagonal after the main diagonal with zeros
    for i in range(seq_len - 1):
        for j in range(i+2):
            matrix[i, j] = i + 2 - j
    
    return matrix


def create_pairwise_distance_matrix(seq_len, device):
    # Create an index matrix using torch.arange
    indices = torch.arange(seq_len, device=device)

    # Create the matrix with |i - j|
    dist_matrix = torch.abs(indices[:, None] - indices[None, :])
    
    return dist_matrix

#FIXME: first difference is that all the embedding size is being used for all the heads
def init_weights_recursive(module):
    # Apply to all parameters in the module
    for param in module.parameters():
        with torch.no_grad():
            # Initialize using the same logic as the C++ code
            param.copy_(torch.rand(param.size()) * 0.2 - 0.1)
    
    # Recursively apply to all sub-modules
    for child in module.children():
        init_weights_recursive(child)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, mdim_head_dimension, ncum, mask, pairwise_distance_matrix,  distance_to_end_matrix, device):
        super(MultiHeadAttention, self).__init__()
        
        self.device = device
        self.d_model = d_model
        self.num_heads = num_heads
        self.ncum = ncum
        self.mdim_head_dimension = mdim_head_dimension
        self.cum_weights = nn.Parameter(torch.randn(num_heads, ncum))
        self.mask = mask
        self.pairwise_distance_matrix = pairwise_distance_matrix
        self.distance_to_end_matrix = distance_to_end_matrix


        # This is the weight that is used to change the slope of the effect of the distance to the end
        self.distance_to_end_weight = nn.Parameter(torch.randn(1, 1))
        # nn.init.normal_(self.distance_to_end_weight, mean=0, std=0.03)


        # This is the weight that is used to change the slope of the effect of the distance between two positions
        self.distance_between_two_positions_weight = nn.Parameter(torch.randn(1, 1))
        # nn.init.normal_(self.distance_between_two_positions_weight, mean=0, std=0.03)
        
        # Create linear layers for all heads then we can split if we want
        self.W_b_q = nn.Linear(d_model, self.num_heads * self.mdim_head_dimension)
        # nn.init.normal_(self.W_b_q.weight, mean=0, std=0.03)
        self.W_b_k = nn.Linear(d_model, self.num_heads * self.mdim_head_dimension)
        # nn.init.normal_(self.W_b_k.weight, mean=0, std=0.03)
        self.W_b_v = nn.Linear(d_model, self.num_heads * self.mdim_head_dimension)
        # nn.init.normal_(self.W_b_v.weight, mean=0, std=0.03)
        
        # self.act_function = nn.ReLU()
        self.act_function = nn.Identity()

    
    def exponential_decay(self, dist, weight):
        return torch.exp((-dist * torch.exp(weight))**5)
        # return torch.exp(-dist * (torch.exp(weight)**5))



    def activation(self, projection_layer, x):
        return self.act_function(x.matmul(projection_layer.weight.t()) + projection_layer.bias)


    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        return x.view(x.shape[0], -1, self.num_heads, self.mdim_head_dimension).transpose(1, 2)

    def qkv(self, x):

        Q = self.activation(self.W_b_q, x)
        K = self.activation(self.W_b_k, x)
        V = self.activation(self.W_b_v, x)


        # Q = self.W_b_q(x)
        # K = self.W_b_k(x)
        # V = self.W_b_v(x)

        # Q = self.act_function(self.W_b_q(x))
        # K = self.act_function(self.W_b_k(x))
        # V = self.act_function(self.W_b_v(x))

        # split heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        return Q, K, V
    
        
    def get_attention(self, x, dist_weight, mask=None):

        Q, K, V = self.qkv(x)
        
        scores = Q.matmul(K.transpose(2, 3)) / math.sqrt(self.mdim_head_dimension)

        attention_scores = torch.nn.functional.softmax(scores + dist_weight + mask , dim=-1)
        
        return attention_scores, V
    
    
    def forward(self, x):

        # print("X shape ", x.shape)

        batch_size, seq_len, _ = x.size()
        
        mask = self.mask[:seq_len, :seq_len] 

        attention_scores, V = self.get_attention(x, self.exponential_decay(self.pairwise_distance_matrix[:seq_len, :seq_len], self.distance_between_two_positions_weight[0,0]).expand(batch_size, self.num_heads, seq_len, seq_len), mask.expand(batch_size, self.num_heads, seq_len, seq_len))
        
        head_output = ((attention_scores * self.exponential_decay(self.distance_to_end_matrix[:seq_len, :seq_len], self.distance_to_end_weight[0,0])).matmul(V)).sum(dim=-1) 
        head_output = head_output.unsqueeze(2).expand(-1,-1,self.ncum,-1)

        # Expand weights to broadcast
        weights_expanded = self.cum_weights.view(-1, self.num_heads, self.ncum, 1)
        weights_expanded = weights_expanded.expand_as(head_output) 
         
        head_outputs_cum = (head_output * weights_expanded).sum(dim=1)

        return head_outputs_cum
 

def l2_penalty_params_except_bias(model, lambda_l2):
    # Get only the weight parameters (exclude biases)
    penalty = lambda_l2 * torch.sum(torch.tensor([torch.sum(torch.pow(param, 2))  for name, param in model.named_parameters() if "bias" not in name]))
    # print("Penalty: ", penalty)

    return penalty


def mini_transformer_loss(output, target, padded_masks):    
    
    running_loss = torch.sum(((output[:, :-1, :] - (target[:, 2:, :])) * padded_masks[:, 2:, :]) **2) / output.shape[0]

    return running_loss
 

class MiniTransformer(nn.Module):
    def __init__(self, d_model, num_heads, mdim_head_dimension, ncum, mask, pairwise_distance_matrix,  distance_to_end_matrix, device):
        super(MiniTransformer, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.ncum = ncum
        self.mdim_head_dimension = mdim_head_dimension
        self.device = device
        self.multiheadattn = MultiHeadAttention(d_model, num_heads, mdim_head_dimension, ncum, mask, pairwise_distance_matrix,  distance_to_end_matrix,  self.device)
        self.prediction_weights = nn.Parameter(torch.randn(self.ncum, self.d_model))
        self.prediction_biases = nn.Parameter(torch.randn(d_model))

    def predict(self, out):
        batch_size = out.shape[0]

        pred = (out.transpose(1,2)).matmul(self.prediction_weights.unsqueeze(0).expand(batch_size,self.ncum,self.d_model)) + self.prediction_biases
        return pred

        
    def forward(self, data):
        x = data[0][:, :-1, :]
        padded_masks = data[1][:, :-1, :]
        out = self.multiheadattn(x)  # The last row is the label

        pred = self.predict(out) * padded_masks

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
        output = model(data)

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

def train_mini_transformer(model, train_loader, optimizer, lambda_l2, EPOCHS, device):
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_path = 'runs/trainer_{}'.format(timestamp)
    tb_writer = SummaryWriter('runs/trainer_{}'.format(timestamp))
    # epoch_number = 0

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    best_vloss = 1_000_000.
    for epoch_number in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))



        avg_loss = train_mini_transformer_one_epoch(model, train_loader, optimizer, lambda_l2, device)

        print('avg_loss: {:.5f}'.format(avg_loss))

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
        for dim in range(model.mdim_head_dimension):
            print("\t Q weights: ", model.multiheadattn.W_b_q.weight[head*model.mdim_head_dimension + dim, :].data, ",\t b: ", model.multiheadattn.W_b_q.bias[head*model.mdim_head_dimension + dim].data, "\n")
    

            print("\t K weights: ", model.multiheadattn.W_b_k.weight[head*model.mdim_head_dimension + dim, :].data, "\t b: ", model.multiheadattn.W_b_k.bias[head*model.mdim_head_dimension + dim].data, "\n")


            print("\t V weights: ", model.multiheadattn.W_b_v.weight[head*model.mdim_head_dimension + dim, :].data, "\t b: ", model.multiheadattn.W_b_v.bias[head*model.mdim_head_dimension + dim].data, "\n\n")
            print("\t cum weights: ", model.multiheadattn.cum_weights[head, :].data, "\n\n")

        print("Prediction weights: \n", model.prediction_weights.data, "\n")

            


#FIXME for performance on CPU
# performance viewer on google chrome. 