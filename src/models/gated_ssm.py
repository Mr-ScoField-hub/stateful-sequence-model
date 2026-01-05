import torch
from torch import nn
from .ssm_block import SSMBlock

class GatedSSM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.ssm = SSMBlock(hidden_dim, input_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.ssm(x)
        gated = torch.sigmoid(self.gate(h)) * h
        return self.output_layer(gated)
