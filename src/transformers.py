import torch
import torch.nn as nn 
import math
from torch.nn.parallel import parallel_apply
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import wandb
from torchviz import make_dot

#FIXME: first difference is that all the embedding size is being used for all the heads

def init_weights_recursive(module, param_low=0, param_high=0.03):
    # Apply to all parameters in the module
    for param in module.parameters():
        # Initialize using normal distribution (mean=param_low, std=param_high)
        nn.init.normal_(param, mean=param_low, std=param_high)
    
    # Recursively apply to all sub-modules
    for child in module.children():
        init_weights_recursive(child, param_low, param_high)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, mdim_head_dimension, ncum, device):
        super(MultiHeadAttention, self).__init__()
        # assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.device = device
        self.d_model = d_model
        self.num_heads = num_heads
        self.ncum = ncum
        # self.mdims = d_model // num_heads
        self.mdim_head_dimension = mdim_head_dimension

        self.cum_weights = nn.Parameter(torch.randn(ncum * num_heads, 1))


        # This is the weight that is used to change the slope of the effect of the distance to the end
        self.distance_to_end_weight = nn.Parameter(torch.randn(1, 1))
        nn.init.normal_(self.distance_to_end_weight, mean=0, std=0.03)


        # This is the weight that is used to change the slope of the effect of the distance between two positions
        self.distance_between_two_positions_weight = nn.Parameter(torch.randn(1, 1))
        nn.init.normal_(self.distance_between_two_positions_weight, mean=0, std=0.03)
        
        # Create separate linear layers for each head
        # self.W_b_q = nn.ModuleList([nn.Linear(d_model, self.mdim_head_dimension) for _ in range(num_heads)])
        # self.W_b_k = nn.ModuleList([nn.Linear(d_model, self.mdim_head_dimension) for _ in range(num_heads)])
        # self.W_b_v = nn.ModuleList([nn.Linear(d_model, self.mdim_head_dimension) for _ in range(num_heads)])
        

        # Create linear layers for all heads then we can split if we want
        self.W_b_q = nn.Linear(d_model, self.num_heads * self.mdim_head_dimension)
        self.W_b_k = nn.Linear(d_model, self.num_heads * self.mdim_head_dimension)
        self.W_b_v = nn.Linear(d_model, self.num_heads * self.mdim_head_dimension)
        
        self.act_function = nn.ReLU()

        
        # Final output projection
        # self.W_o = nn.Linear(d_model, d_model)

    def distance_to_end_function(self, seq_len, weight):

        dist_list = torch.arange(-2, seq_len - 2, device = self.device).flip(0)

        distweight_list = torch.exp(-torch.pow(dist_list * torch.exp(weight),2))

        return distweight_list
    
    def distance_between_two_positions_function(self, seq_len, weight):

        # Create an index matrix using torch.arange
        indices = torch.arange(seq_len, device=self.device)

        # Create the matrix with |i - j|
        dist_matrix = torch.abs(indices[:, None] - indices[None, :])
        
        distweight_matrix = torch.exp(-torch.pow(dist_matrix * torch.exp(weight),2))

        return distweight_matrix

    #FIXME: Harald's activation function
    def activation(self, projection_layer, x):
        res = self.act_function(torch.matmul(x, projection_layer.weight.t()) + projection_layer.bias) + projection_layer.bias

        return res

    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        
        return x.view(x.shape[0], -1, self.num_heads, self.mdim_head_dimension).transpose(1, 2)

    def qkv(self, x):

        # Q = self.activation(self.W_b_q[head_number], x)
        # K = self.activation(self.W_b_k[head_number], x)
        # V = self.activation(self.W_b_v[head_number], x)

        # Q = self.act_function(self.W_b_q(x))
        # K = self.act_function(self.W_b_k(x))
        # V = self.act_function(self.W_b_v(x))

        Q = self.activation(self.W_b_q, x)
        K = self.activation(self.W_b_k, x)
        V = self.activation(self.W_b_v, x)

        # split heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        return Q, K, V
    
        
    def get_attention(self, x, dist_weight, mask=None):

        Q, K, V = self.qkv(x)
        
        scores = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.d_model)

        attention_scores = torch.softmax(scores + dist_weight + mask , dim=-1)
        
        return attention_scores, V
    
    def attention_layer_output(self, attention_scores, V,):

        output = torch.matmul(attention_scores, V)

        return output
    
    def forward(self, x, mask = None):
        batch_size, seq_len, _ = x.size()
        
        # Process each head separately
        head_outputs_cum = []
        
        mask = (torch.triu(torch.ones(batch_size, self.num_heads, seq_len, seq_len, device= self.device), diagonal=1) * -1e9)


        # for i in range(self.num_heads):

        attention_scores, V = self.get_attention(x, self.distance_between_two_positions_function(seq_len, self.distance_between_two_positions_weight[0,0]).expand(batch_size, self.num_heads, seq_len, seq_len), mask)
            # self.cum_weights[i * , 0] * 
            # head_output = torch.sum(self.attention_layer_output(attention_scores, V) * self.distance_to_end_function(seq_len, self.distance_to_end_weight[0,0]), dim = -1, keepdim=True)

        head_output = self.attention_layer_output(attention_scores, V)
        head_output = (head_output * self.distance_to_end_function(seq_len, self.distance_to_end_weight[0,0]).view(1, 1, -1, 1)).expand_as(head_output).unsqueeze(2)
        head_output = torch.cat([head_output, head_output], dim = 2)

        # Expand weights to broadcast
        weights_expanded = self.cum_weights.view(-1, self.num_heads, self.ncum, 1, 1)
        weights_expanded = weights_expanded.expand_as(head_output) 

                
        # for j in range(self.ncum):
        #     if i == 0:
        #         head_outputs_cum.append(torch.sum(head_output * self.cum_weights[i * self.ncum + j, 0], dim = -1, keepdim=True))
        #     else:
        #         head_outputs_cum[j] += torch.sum(head_output * self.cum_weights[i * self.ncum + j, 0], dim = -1, keepdim=True)
            
        head_outputs_cum = (head_output * weights_expanded).sum(dim=1)

        return head_outputs_cum


def l2_penalty_params_except_bias(model, lambda_l2):
    # Get only the weight parameters (exclude biases)
    penalty = lambda_l2 * torch.sum(torch.tensor([torch.sum(torch.pow(param, 2))  for name, param in model.named_parameters() if "bias" not in name]))
    return penalty


def mini_transformer_loss(output, target, model, lambda_l2):

    running_loss = nn.MSELoss()(output[:, 1:, :], target[:, 2:, :]) #+ l2_penalty_params_except_bias(model, lambda_l2)

    return running_loss
 

class MiniTransformer(nn.Module):
    def __init__(self, d_model, num_heads, mdim_head_dimension, ncum, device):
        super(MiniTransformer, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.ncum = ncum
        # self.mdims = d_model // num_heads
        self.mdim_head_dimension = mdim_head_dimension
        self.device = device
        self.multiheadattn = MultiHeadAttention(d_model, num_heads, mdim_head_dimension, ncum, self.device)

        
        # self.prediction_weights = nn.ModuleList([nn.Linear(mdim_head_dimension, d_model , bias = False) for _ in range(self.ncum)])
        self.prediction_weights = nn.Parameter(torch.randn(self.ncum, self.mdim_head_dimension, self.d_model))
        self.prediction_biases = nn.Parameter(torch.randn(d_model))

    def predict(self, out):
        
        pred = torch.matmul(out, self.prediction_weights).sum(dim = 1) + self.prediction_biases
        return pred


        # for i in range(self.ncum):
        #     pred = self.prediction_weights[i](out[:, i, :, :])
        #     if i == 0:
        #         preds = pred
        #     else:
        #         preds += pred
        # return preds + self.prediction_biases
        
    def forward(self, x):
        out = self.multiheadattn(x[:, :-1, :])  # The last row is the label

        pred = self.predict(out)

        return pred



# define training function for MiniTransformer with tensorboard logging
def train_mini_transformer_one_epoch(model, train_loader, optimizer, lambda_l2, device):
    
    running_loss = 0.0

    last_loss = 0.0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_loader, 0):
        optimizer.zero_grad()
        # Move inputs to the device (MPS or CPU)
        # data = data.to(device)
        output = model(data)

        # dot = make_dot(output, params=dict(model.named_parameters()))
        # dot.graph_attr.update({'size': '100,100!'})  # Increase canvas size
        # dot.graph_attr.update({'ratio': 'compress'})  # Compress the graph
        # dot.render("model_graph", format="pdf")
        # del dot 


        # tb_writer.add_graph(model, data)
        # tb_writer.close()


        loss = mini_transformer_loss(output, data, model, lambda_l2)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()


    return running_loss/len(train_loader)

def train_mini_transformer(model, train_loader, optimizer, lambda_l2, EPOCHS, device):
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tb_writer = SummaryWriter('runs/trainer_{}'.format(timestamp))
    # epoch_number = 0


    best_vloss = 1_000_000.
    for epoch_number in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_mini_transformer_one_epoch(model, train_loader, optimizer, lambda_l2, device)

        print('avg_loss: {}'.format(avg_loss))

        # Log average loss for the epoch
        tb_writer.add_scalar('avg_loss', avg_loss, epoch_number)

        # Log model parameters (weights and biases) after every epoch
        for name, param in model.named_parameters():
            tb_writer.add_histogram(f"{name}.weights", param, epoch_number)         # Log weights
            if param.grad is not None:
                tb_writer.add_histogram(f"{name}.grads", param.grad, epoch_number)  # Log gradients
        

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)