from torch import nn
from .base import FeatureExtractor
from typing import Literal
from src.utils.types import _size_2_t, _size_2_t_list
from torchtyping import TensorType


class CNNBlock(nn.Module):
    """Single CNN block constructed of combination of Conv2d, Activation, Pooling, Batch Normalization and Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: str | _size_2_t = 0,
        pool_kernel_size: _size_2_t = 1,
        pool_type: Literal["Max", "Avg"] = "Max",
        use_batch_norm: bool = True,
        dropout: float = 0,
        activation: str = "ReLU",
    ):
        """
        Args:
            in_channels (int): Number of Conv2d input channels.
            out_channels (int): Number of Conv2d out channels.
            kernel_size (int): Conv2d kernel equal to `(kernel_size, kernel_size)`.
            stride (int, optional): Conv2d stride equal to `(stride, stride)`.
                Defaults to 1.
            padding (int | str, optional): Conv2d padding equal to `(padding, padding)`.
                Defaults to 1.. Defaults to 0.
            pool_kernel_size (int, optional): Pooling kernel equal to `(pool_kernel_size, pool_kernel_size)`.
                 Defaults to 1.
            pool_type (Literal["Max", "Avg"], optional): Pooling type. Defaults to "Max".
            use_batch_norm (bool, optional): Whether to use Batch Normalization (BN) after activation. Defaults to True.
            dropout (float, optional): Dropout probability (used after BN). Defaults to 0.
            activation (str, optional): Type of activation function used before BN. Defaults to 0.
        """
        super().__init__()
        if isinstance(pool_kernel_size, int):
            self.use_pool = pool_kernel_size > 1
        else:
            self.use_pool = all(dim == 1 for dim in pool_kernel_size)
        self.use_batch_norm = use_batch_norm
        self.use_dropout = dropout > 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = getattr(nn, activation)()
        if self.use_pool:
            self.pool = getattr(nn, f"{pool_type}Pool2d")(pool_kernel_size)

        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels)

        if self.use_dropout:
            self.dropout = nn.Dropout2d(dropout)

    def forward(
        self, x: TensorType["batch", "in_channels", "in_height", "in_width"]
    ) -> TensorType["batch", "out_channels", "out_height", "out_width"]:
        out = self.conv(x)
        out = self.activation(out)
        if self.use_pool:
            out = self.pool(out)
        if self.use_batch_norm:
            out = self.batch_norm(out)
        if self.use_dropout:
            out = self.dropout(out)
        return out


class DeepCNN(FeatureExtractor):
    """Deep Convolutional Neural Network (CNN) constructed of many CNN blocks and ended with Global Average Pooling."""

    name: str = "DeepCNN"

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
    def out_dim(self) -> int:
        return self.out_channels[-1]
