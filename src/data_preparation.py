import torch
from torch.utils.data import Dataset

import torch
from torch.utils.data import Dataset
import random

class SimulatedDataset(Dataset):
    def __init__(self, n, p=3, pos1=0, pos2=1, pos3=2, maxlen=4, device=None):
        self.n = n        # Number of data points
        self.p = p        # Number of features (dimensions)
        self.pos1 = pos1  # Index for pos1 condition
        self.pos2 = pos2  # Index for pos2 condition
        self.pos3 = pos3  # Index for pos3 condition
        self.maxlen = maxlen  # Maximum sequence length

        # Specify the device (CPU or GPU)
        self.device = device if device is not None else torch.device('cpu')

        # Generate data directly on the desired device
        self.data =  self.generate_data()

    def generate_data(self):
        input_data = []
        start_indices = []

        # Generate all data points
        for i in range(self.n):
            cur_matrix = torch.zeros((self.maxlen, self.p), dtype=torch.float32, device=self.device)  # Pre-allocate a matrix for each sample
            justseenfirst = False
            seenfirst = False
            justseensecond = False
            seensecond = False

            # start_indices.append(i)  # Store the index of the current individual
            
            # Generate the sequences for each sample
            for j in range(self.maxlen):
                curran = torch.rand(self.p, device=self.device)  # Generate random values for the sequence, vectorized

                # Apply the conditions based on the pos1, pos2, pos3
                for k in range(self.p):
                    if (k != self.pos3 and curran[k] > 0.3) or (k == self.pos3 and seenfirst and seensecond and curran[k] > 0.1):
                        cur_matrix[j, k] = 1.0
                        if k == self.pos3:
                            justseenfirst = False
                            justseensecond = False
                        if k == self.pos1:
                            justseenfirst = True
                        if seenfirst and k == self.pos2:
                            justseensecond = True
                    else:
                        cur_matrix[j, k] = 0.0

                # Update the flags for seenfirst and seensecond
                seenfirst = justseenfirst
                seensecond = justseensecond

                if j > 1 and torch.rand(1) > 0.8:
                    break

            input_data.append(cur_matrix)  # Store the generated matrix

        # Convert the start_indices to a tensor
        # start_indices_tensor = torch.tensor(start_indices, dtype=torch.int64, device=self.device)

        return torch.stack(input_data)

    def __len__(self):
        return len(self.data)  # Number of data points (n)

    def __getitem__(self, idx):
        return self.data[idx]  # Return the matrix corresponding to the idx-th sample