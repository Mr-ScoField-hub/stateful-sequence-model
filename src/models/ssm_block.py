import torch
from torch import nn
from .diagonal_ssm import DiagonalSSM

class SSMBlock(nn.Module):
    def __init__(self, state_dim, input_dim, output_dim):
        super().__init__()
        self.ssm = DiagonalSSM(state_dim, input_dim)
        self.linear_out = nn.Linear(state_dim, output_dim)

    def forward(self, x):
        h = self.ssm(x)
        return self.linear_out(h)
