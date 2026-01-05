import torch
from torch import nn, optim
from src.models.gated_ssm import GatedSSM
from src.data.long_context import copy_task

# Hyperparameters
seq_len = 20
batch_size = 32
input_dim = 1
hidden_dim = 32
epochs = 200
lr = 0.01

# Model, loss, optimizer
model = GatedSSM(input_dim, hidden_dim, seq_len)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    X, Y = copy_task(seq_len, batch_size)
    X = torch.tensor(X).unsqueeze(-1)
    Y = torch.tensor(Y).unsqueeze(-1)
    
    optimizer.zero_grad()
    out = model(X)
    loss = criterion(out, Y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

# Test
X_test, Y_test = copy_task(seq_len, 4)
X_test = torch.tensor(X_test).unsqueeze(-1)
Y_test = torch.tensor(Y_test).unsqueeze(-1)
with torch.no_grad():
    out_test = model(X_test)
    print("Test output sample:", out_test[0, :, 0])
    print("Test target sample:", Y_test[0, :, 0])
