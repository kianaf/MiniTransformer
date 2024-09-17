import torch

def relu(x):
    return torch.maximum(x, torch.tensor(0.0))

def activation(W, b, obs):
    res = b[0]
    res += torch.sum(relu(obs * W + b[0]))
    return res

def query(W, b, obs, p):
    return activation(W[0:p], b[0:1], obs)

def key(W, b, obs, p):
    return activation(W[p:2*p], b[1:2], obs)

def value(W, b, obs, p):
    return activation(W[2*p:3*p], b[2:3], obs)

def lossFun(W, b, input, n, p):
    queryvec = []
    keyvec = []
    valuevec = []

    for i in range(n - 1):
        obs = input[i*p : (i+1)*p]
        queryvec.append(query(W, b, obs, p))
        keyvec.append(key(W, b, obs, p))
        valuevec.append(value(W, b, obs, p))

    queryvec = torch.stack(queryvec)
    keyvec = torch.stack(keyvec)
    valuevec = torch.stack(valuevec)

    repcum = 0.0
    for i in range(n - 1):
        indices = torch.arange(n - 1)
        curw = torch.exp(queryvec[i] * keyvec) / (torch.abs(i - indices) + 1)
        cursum = torch.sum(curw * valuevec)
        curwsum = torch.sum(curw)
        repcum += cursum / curwsum

    loss = 0.0
    for i in range(p):
        pred = W[3*p + i] * repcum + b[3 + i]
        delta = input[p*(n - 1) + i] - pred
        loss += delta * delta
    return loss

def main():
    n = 4
    p = 3

    # Initialize weights and biases
    W = (torch.rand(p*4, requires_grad=True) * 0.2 - 0.1).clone().detach().requires_grad_(True)
    b = (torch.rand(3 + p, requires_grad=True) * 0.2 - 0.1).clone().detach().requires_grad_(True)

    # Define constants
    aaa = 1.0
    bbb = 0.0
    ccc = 0.0
    ddd = 1.0

    # Input data
    input_list = [
        aaa, 0.0, bbb,    # obs 1
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.0,
        ccc, 0.0, ddd,
        aaa, 0.0, bbb,    # obs 2
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        ccc, 0.0, ddd,
        0.0, 0.0, 0.0,    # obs 3
        aaa, 0.0, bbb,
        0.0, 1.0, 0.0,
        ccc, 0.0, ddd,
        aaa, 0.0, bbb,    # obs 4
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        aaa, 0.0, bbb,    # obs 5
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.0,
        ccc, 0.0, ddd,
        0.0, 0.0, 0.0,    # obs 6
        aaa, 0.0, bbb,
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.0,    # obs 7
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 1.0, 0.0
    ]

    input = torch.tensor(input_list)
    startvec = [0, 4, 8, 12, 16, 20, 24]
    nu = 0.01  # Learning rate

    for iter in range(100):
        for j in range(len(startvec)):
            if j < len(startvec) - 1:
                curn = startvec[j+1] - startvec[j]
            else:
                curn = int(len(input) / p) - startvec[j]
            curinput_start = startvec[j] * p
            curinput_end = curinput_start + curn * p
            curinput = input[curinput_start : curinput_end]

            for l in range(curn - 2):
                n_l = curn - l
                curinput_slice_start = l * p
                curinput_slice_end = curinput_slice_start + n_l * p
                curinput_slice = curinput[curinput_slice_start : curinput_slice_end]
                loss = lossFun(W, b, curinput_slice, n_l, p)
                loss.backward()

                with torch.no_grad():
                    W -= nu * W.grad
                    b -= nu * b.grad

                    W.grad.zero_()
                    b.grad.zero_()

    print('W=', W.detach().numpy())
    print('b=', b.detach().numpy())

if __name__ == '__main__':
    main()