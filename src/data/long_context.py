import numpy as np
import torch

def copy_task(batch_size, seq_len, input_dim=1):
    """
    Generates a synthetic copy task dataset.
    X: random binary sequences of shape (batch_size, seq_len, input_dim)
    Y: identical to X
    """
    # Use 0/1 if input_dim == 1, otherwise generate random float sequences
    if input_dim == 1:
        X = np.random.randint(0, 2, size=(batch_size, seq_len, input_dim)).astype(np.float32)
    else:
        X = np.random.rand(batch_size, seq_len, input_dim).astype(np.float32)
    
    Y = X.copy()
    return torch.tensor(X), torch.tensor(Y)
