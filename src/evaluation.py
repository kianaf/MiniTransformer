import torch
import torch.nn as nn

def calculate_bench1_loss(train_data, eval_data):
    """
    Calculate the loss of the benchmark 1 model.
    averaging over all timepoints of all the sequences
    """

    p = train_data.shape[2]
    n = train_data.shape[0]
    predn = eval_data.shape[0]

    #This is for baseline1: Bench1 which averages over all timepoints of all the sequences
    
    # dimave = [0.0] * p 

    # train_data_plain = torch.cat(train_data, dim=0)
    dimave = torch.mean(train_data.view(-1, 3), dim=0)


    benchdelta = 0.0
    benchloss = 0.0

    for i in range(predn):
        benchdelta = eval_data[i, 2:, :] - dimave
        benchloss += torch.mean(benchdelta * benchdelta).item()

    benchloss /= predn
    

    return dimave, benchloss



def calculate_bench2_loss(train_data, eval_data, dimave):
    """
    Calculate the loss of the benchmark 2 model.
    Looking at pos2, and based on the probability of pos3 given pos2, predict pos3
    """


    p = train_data.shape[2]
    n = train_data.shape[0]
    predn = eval_data.shape[0]

    #This is for baseline2: Bench2 which averages over all timepoints of all the sequences
    



    pos1 = 0
    pos2 = 1
    pos3 = 2

    pos2count1 = 0
    pos2count0 = 0
    pos3ave1 = 0.0
    pos3ave0 = 0.0

    for i in range(n):

        curn = train_data.shape[1] 

        for j in range(curn):
            if (train_data[i, j, pos2] == 1.0):
                pos2count1+= 1
                pos3ave1 += train_data[i, j, pos3]
            else:
                pos2count0+= 1
                pos3ave0 += train_data[i, j, pos3]

    pos3ave1 = pos3ave1/pos2count1
    pos3ave0 = pos3ave0/pos2count0
        


    bench2pred = 0.0
    bench2delta = 0.0
    bench2loss = 0.0

    for i in range(predn):
        for j in range(p):
            if j == pos3:
                if eval_data[i, -1, pos2] == 1.0:
                    bench2pred = pos3ave1
                else:
                    bench2pred = pos3ave0
            else:
                bench2pred = dimave[j]

            bench2delta = eval_data[i, 2:, j] - bench2pred
            bench2loss += torch.mean(bench2delta * bench2delta)


    bench2loss /= predn*p

    return bench2loss.item()


def evaluate_mini_transformer(eval_data, model):
    """
    Evaluate the MiniTransformer model on the evaluation data.
    """

    pred = model(eval_data)
    loss = nn.MSELoss()(pred[:, 1:, :], eval_data[:, 2:, :])
    # loss = torch.sum((pred[:, 1:, :]- eval_data[:, 2:, :]) ** 2)
    return loss
