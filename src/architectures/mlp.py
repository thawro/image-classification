import torch
from torch import nn


class FeedForwardBlock(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, use_batch_norm: bool = True, dropout: float = 0
    ):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.use_dropout = dropout > 0

        self.linear = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_dim)

        if self.use_dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = self.relu(out)
        if self.use_batch_norm:
            out = self.batch_norm(out)
        if self.use_dropout:
            out = self.dropout(out)
        return out


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims):
        super().__init__()
        in_dims = [in_dim] + hidden_dims[:-1]
        n_layers = len(hidden_dims)
        layers: list[nn.Module] = [
            FeedForwardBlock(in_dims[i], hidden_dims[i]) for i in range(n_layers)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
