import torch
from src.models.ssm_block import StateSpaceBlock

T = 50
batch_size = 1
state_dim = 16
input_dim = 1

# Impulse input
u = torch.zeros(T, batch_size, input_dim)
u[10, 0, 0] = 1.0

ssm = StateSpaceBlock(state_dim=state_dim, input_dim=input_dim)
states = ssm(u)

print(states[10:30, 0, 0])
