from torch import nn

from src.architectures.feature_extractors.base import FeatureExtractor
from src.architectures.helpers import FeedForwardBlock
from src.utils.types import Any


class MLP(FeatureExtractor):
    """Multi Layer Perceptron (MLP) constructed of many FeedForward blocks."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        use_batch_norm: bool = True,
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
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must have atleast one element")
        self.hidden_dims = hidden_dims
        self.use_batch_norm = use_batch_norm
        self.dropout = dropout
        self.activation = activation
        in_dims = [in_dim] + hidden_dims[:-1]
        n_layers = len(hidden_dims)
        layers: list[nn.Module] = [
            nn.Flatten(start_dim=1, end_dim=-1),
            *[
                FeedForwardBlock(in_dims[i], hidden_dims[i], use_batch_norm, dropout, activation)
                for i in range(n_layers)
            ],
        ]
        net = nn.Sequential(*layers)
        super().__init__(net)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "activation": self.activation,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "batch_norm": self.use_batch_norm,
        }

    @property
    def out_dim(self) -> int:
        return self.hidden_dims[-1]
