import torch
from torch import nn, optim
import numpy as np
from src.models.gated_ssm import GatedSSM
from src.data.long_context import copy_task

# Hyperparameters
input_dim = 1
hidden_dim = 32
batch_size = 8
epochs = 100
lr = 0.01
sequence_lengths = [20, 50, 100, 200, 500]  # variable lengths to test memory

# Model, loss, optimizer
max_seq_len = max(sequence_lengths)
model = GatedSSM(input_dim, hidden_dim, max_seq_len)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

results = {}

# Training across max sequence length
for epoch in range(epochs):
    X, Y = copy_task(max_seq_len, batch_size)
    X = torch.tensor(X).unsqueeze(-1)
    Y = torch.tensor(Y).unsqueeze(-1)
    
    optimizer.zero_grad()
    out = model(X)
    loss = criterion(out, Y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

# Evaluate memory retention for each sequence length
with torch.no_grad():
    for L in sequence_lengths:
        X_test, Y_test = copy_task(L, 4)
        X_test = torch.tensor(X_test).unsqueeze(-1)
        Y_test = torch.tensor(Y_test).unsqueeze(-1)
        
        out_test = model(X_test[:, :L, :])
        mse = nn.MSELoss()(out_test, Y_test).item()
        results[L] = mse
        print(f"Sequence length {L}, Test MSE: {mse:.6f}")

# Summary
print("\nMemory performance across sequence lengths:")
for L, mse in results.items():
    print(f"Length {L}: MSE {mse:.6f}")
