import torch
from torch import nn

class DiagonalSSM(nn.Module):
    """
    Diagonal State Space Model (SSM) block.
    Implements a simplified SSM with diagonal state matrix for long sequences.
    """

    def __init__(self, state_dim, input_dim, output_dim=None):
        super(DiagonalSSM, self).__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim or state_dim

        # learnable diagonal state matrix A
        self.A = nn.Parameter(torch.rand(state_dim) * 0.1)  # small init for stability

        # input-to-state linear transform
        self.B = nn.Linear(input_dim, state_dim, bias=False)
        nn.init.uniform_(self.B.weight, -0.01, 0.01)  # small init

        # state-to-output linear transform
        self.linear_out = nn.Linear(state_dim, self.output_dim)

        # initial state
        self.register_buffer("state", torch.zeros(state_dim))

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        returns: (batch, seq_len, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, seq_len, self.state_dim, device=x.device)
        state = self.state.unsqueeze(0).repeat(batch_size, 1)

        # clamp A to prevent numerical explosion
        exp_decay = torch.exp(-torch.clamp(self.A, max=10.0))  # stable decay

        for t in range(seq_len):
            # state update
            state = exp_decay * state + self.B(x[:, t, :])
            state = torch.tanh(state)  # keep values bounded
            h[:, t, :] = state

        out = self.linear_out(h)
        return out
