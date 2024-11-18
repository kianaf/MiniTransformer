import torch
from torch.utils.data import Dataset

import torch
from torch.utils.data import Dataset
import random
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence


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
        self.data = self.generate_data()
   
    def torch_cpp_rand(self):
        # Define RAND_MAX based on typical C++ value (e.g., 32767)
        RAND_MAX = 32767
        # Generate an integer in the range [0, RAND_MAX]
        random_int = torch.randint(0, RAND_MAX + 1, (1,), dtype=torch.int32).item()
        # Normalize by RAND_MAX to get a float in the range [0, 1]
        return random_int / RAND_MAX

    def generate_data(self):


        
        input_data = []

        # Generate all data points
        for i in range(self.n):
            cur_matrix = []  # Dynamically create rows for each sample
            justseenfirst = False
            seenfirst = False
            justseensecond = False
            seensecond = False

            # Generate the sequences for each sample (no pre-allocation)
            for j in range(self.maxlen):
                # curran = torch.rand(self.p, device=self.device)  # Generate random values for the sequence, vectorized

                row = torch.zeros(self.p, dtype=torch.float32, device=self.device)  # Initialize the current row

                # Apply the conditions based on the pos1, pos2, pos3
                for k in range(self.p):
                    # curran = torch.rand(1).item()
                    curran = self.torch_cpp_rand()

                    if (k != self.pos3 and curran > 0.3) or (k == self.pos3 and seenfirst and seensecond and curran > 0.1):
                        row[k] = 1.0
                        if k == self.pos3:
                            justseenfirst = False
                            justseensecond = False
                        if k == self.pos1:
                            justseenfirst = True
                        if seenfirst and k == self.pos2:
                            justseensecond = True
                    else:
                        row[k] = 0.0

                # Append the row to cur_matrix
                cur_matrix.append(row)

                # Update the flags for seenfirst and seensecond
                seenfirst = justseenfirst
                seensecond = justseensecond

                # Stop the sequence generation randomly after j > 1 (for variable sequence lengths)
                if j > 1 and self.torch_cpp_rand() > 0.8:
                    break

            # Convert cur_matrix (which is a list of tensors) into a tensor and append to input_data
            cur_matrix_tensor = torch.stack(cur_matrix)
            input_data.append(cur_matrix_tensor)  # Store the generated matrix

        # Convert the list of tensors into a single tensor by stacking them (if needed)
        # or return them as a list if the row numbers are different
        return input_data

    def __len__(self):
        return len(self.data)  # Number of data points (n)

    def __getitem__(self, idx):
        return self.data[idx]  # Return the matrix corresponding to the idx-th sample
    


# Custom collate function to pad sequences and create a padding mask
def collate_function(batch):
        # Step 1: Pad sequences with a temporary value
    padded_sequences = pad_sequence(batch, batch_first=True, padding_value=-1)

    # Step 2: Create a mask (True where data is present, False where padding is present)
    mask = (padded_sequences != -1)
    
    # Step 3: Convert the temporary padding value to zero
    padded_sequences = torch.where(mask, padded_sequences, torch.tensor(0, dtype=padded_sequences.dtype))

    return padded_sequences, mask