from abc import abstractmethod
from typing import Literal

from torch import nn
from torchvision.models._utils import _make_divisible

from src.utils.types import Optional, Tensor, TensorType, _size_2_t


class OutChannelsModule(nn.Module):
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def out_channels(self):
        raise NotImplementedError()


class ResidualBlock(nn.Module):
    def __init__(self, net: nn.Module):
        self.net = net

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class CNNBlock(nn.Module):
    """Single CNN block constructed of combination of Conv2d, Activation, Pooling, Batch Normalization and Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: str | _size_2_t = 0,
        groups: int = 1,
        pool_kernel_size: _size_2_t = 1,
        pool_type: Literal["Max", "Avg"] = "Max",
        use_batch_norm: bool = True,
        dropout: float = 0,
        activation: Optional[str] = "ReLU",
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
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        use_bias = not use_batch_norm
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=use_bias,
        )
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels)

        self.linear = activation is None
        if not self.linear:
            self.activation_fn = getattr(nn, activation)()
        if self.use_pool:
            self.pool = getattr(nn, f"{pool_type}Pool2d")(pool_kernel_size, stride=2)

        if self.use_dropout:
            self.dropout = nn.Dropout2d(dropout)

    def forward(
        self, x: TensorType["batch", "in_channels", "in_height", "in_width"]
    ) -> TensorType["batch", "out_channels", "out_height", "out_width"]:
        out = self.conv(x)
        if self.use_batch_norm:
            out = self.batch_norm(out)
        if not self.linear:
            out = self.activation_fn(out)
        if self.use_pool:
            out = self.pool(out)
        if self.use_dropout:
            out = self.dropout(out)
        return out


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

    def forward(self, x: TensorType["batch", "in_dim"]) -> TensorType["batch", "out_dim"]:
        out = self.linear(x)
        out = self.activation(out)
        if self.use_batch_norm:
            out = self.batch_norm(out)
        if self.use_dropout:
            out = self.dropout(out)
        return out


class DepthwiseSeparableConvolution(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t,
        padding: str | _size_2_t,
    ):
        self.depthwise = CNNBlock(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = CNNBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class SEBlock(nn.Module):
    """Squeeze and Excitation (SE) Block based on https://arxiv.org/pdf/1709.01507.pdf
    Use it as a wrapper for any function (nn.Module block) F_tr, which transforms input to CxHxW space.
    """

    def __init__(
        self,
        block: CNNBlock,
        reduction_ratio: int = 16,
        reduce_activation: str = "ReLU",
        expand_activation: str = "Sigmoid",
        make_divisible_by_n: int = 8,
    ):
        super().__init__()
        channels = block.out_channels
        self.reduction_ratio = reduction_ratio
        self.reduce_activation = reduce_activation
        self.expand_activation = expand_activation
        mid_channels = channels // reduction_ratio
        mid_channels = _make_divisible(mid_channels, make_divisible_by_n)
        self.block = block  # C' x H' x W' -> C x H x W
        self.mid_channels = mid_channels
        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # C x H x W -> C x 1 x 1
            nn.Flatten(1, -1),  # C x 1 x 1 -> C
        )
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid_channels),  # C -> C/r
            getattr(nn, reduce_activation)(),
            nn.Linear(mid_channels, channels),  # C/r -> C
            getattr(nn, expand_activation)(),
        )

    def forward(self, x: Tensor) -> Tensor:
        U = self.block(x)  # C x H x W
        z = self.squeeze(U)  # C
        s = self.excitation(z)  # C
        s = s.unsqueeze(-1).unsqueeze(-1)  # C x 1 x 1
        x_out = s * U  # C x H x W
        return x_out
