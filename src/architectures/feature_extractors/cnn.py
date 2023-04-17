from typing import Literal

from torch import nn

from src.architectures.helpers import CNNBlock
from src.utils.types import Any, _size_2_t_list

from .base import FeatureExtractor


class DeepCNN(FeatureExtractor):
    """Deep Convolutional Neural Network (CNN) constructed of many CNN blocks and ended with Global Average Pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: list[int],
        kernels: _size_2_t_list,
        pool_kernels: _size_2_t_list,
        pool_type: Literal["Max", "Avg"] = "Max",
        use_batch_norm: bool = True,
        dropout: float = 0,
        activation: str = "ReLU",
    ):
        """
        Args:
            in_channels (int): Number of image channels.
            out_channels (list[int]): Number of channels used in CNN blocks.
            kernels (int | list[int]): Kernels of Conv2d in CNN blocks.
                If int or tuple[int, int] is passed, then all layers use same kernel size.
            pool_kernels (int | list[int]): Kernels of Pooling in CNN blocks.
                If int is passed, then all layers use same pool kernel size.
            pool_type (Literal["Max", "Avg"], optional): Pooling type in CNN blocks. Defaults to "Max".
            use_batch_norm (bool, optional): Whether to use BN in CNN blocks. Defaults to True.
            dropout (float, optional): Dropout probability used in CNN blocks. Defaults to 0.
            activation (str, optional): Type of activation function used in CNN blocks. Defaults to 0.
        """
        super().__init__()
        self.out_channels = out_channels
        self.kernels = kernels
        self.pool_kernels = pool_kernels
        self.pool_type = pool_type
        self.use_batch_norm = use_batch_norm
        self.dropout = dropout
        self.activation = activation
        n_blocks = len(out_channels)
        fixed_params = dict(
            pool_type=pool_type,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            activation=activation,
        )
        if isinstance(kernels, int) or isinstance(kernels, tuple):
            kernels = [kernels] * n_blocks
        if isinstance(pool_kernels, int) or isinstance(pool_kernels, tuple):
            pool_kernels = [pool_kernels] * n_blocks
        layers = [
            CNNBlock(
                in_channels if i == 0 else out_channels[i - 1],
                out_channels[i],
                kernels[i],
                pool_kernel_size=pool_kernels[i],
                **fixed_params,
            )
            for i in range(n_blocks)
        ] + [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()]
        self.net = nn.Sequential(*layers)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "out_channels": self.out_channels,
            "kernels": self.kernels,
            "pool_kernels": self.pool_kernels,
            "pool_type": self.pool_type,
            "batch_norm": self.use_batch_norm,
            "dropout": self.dropout,
            "activation": self.activation,
        }

    @property
    def out_dim(self) -> int:
        return self.out_channels[-1]
