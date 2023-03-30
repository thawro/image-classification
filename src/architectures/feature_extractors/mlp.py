import torch
from torch import nn
from .base import FeatureExtractor


class FeedForwardBlock(nn.Module):
    """Single FeedForward block constructed of combination of Linear, Activation, Batch Normalization and Dropout."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_batch_norm: bool = True,
        dropout: float = 0,
        activation: str = "ReLU",
    ):
        """
        Args:
            in_dim (int): Input dimension of `torch.nn.Linear` layer.
            out_dim (int): Output dimension of `torch.nn.Linear` layer.
            use_batch_norm (bool, optional): Whether to use Batch Normalization (BN) after activation. Defaults to True.
            dropout (float, optional): Dropout probability (used after BN). Defaults to 0.
            activation (str, optional): Type of activation function used before BN. Defaults to 0.
        """
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.use_dropout = dropout > 0

        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = getattr(nn, activation)()
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_dim)

        if self.use_dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = self.activation(out)
        if self.use_batch_norm:
            out = self.batch_norm(out)
        if self.use_dropout:
            out = self.dropout(out)
        return out


class MLP(FeatureExtractor):
    """Multi Layer Perceptron (MLP) constructed of many FeedForward blocks."""

    name: str = "MLP"

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        use_batch_norm: bool,
        dropout: float = 0,
        activation: str = "ReLU",
    ):
        """
        Args:
            in_dim (int): Input dimension.
            hidden_dims (list[int]): Dimensionalities of hidden layers.
            use_batch_norm (bool, optional): Whether to use Batch Normalization (BN) after activation. Defaults to True.
            dropout (float, optional): Dropout probability (used after BN). Defaults to 0.
            activation (str, optional): Type of activation function used before BN. Defaults to 0.
        """
        super().__init__()
        self.hidden_dims = hidden_dims
        in_dims = [in_dim] + hidden_dims[:-1]
        n_layers = len(hidden_dims)
        layers: list[nn.Module] = [
            nn.Flatten(start_dim=1, end_dim=-1),
            *[
                FeedForwardBlock(in_dims[i], hidden_dims[i], use_batch_norm, dropout, activation)
                for i in range(n_layers)
            ],
        ]
        self.net = nn.Sequential(*layers)

    @property
    def out_shape(self):
        return self.hidden_dims[-1]
