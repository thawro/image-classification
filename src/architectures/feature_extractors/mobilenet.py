"""MobiletNet architecture based on:
Main idea, depthwise-separable convolutions (v1): https://arxiv.org/pdf/1704.04861v1.pdf
+ inverted linear bottleneck (v2): https://arxiv.org/pdf/1801.04381.pdf
+ squeeze and excite (v3): https://arxiv.org/pdf/1905.02244v5.pdf
"""

from torch import nn
from torchvision.models._utils import _make_divisible

from src.architectures.feature_extractors.base import FeatureExtractor
from src.architectures.helpers import CNNBlock
from src.architectures.utils import make_named_sequential
from src.utils.types import Tensor, _any_dict, _size_2_t


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        expansion_ratio: int,
        out_channels: int,
        stride: int,
        kernel_size: _size_2_t = 3,
    ):
        super().__init__()
        self.expansion_ratio = expansion_ratio
        mid_channels = in_channels * expansion_ratio
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels

        if expansion_ratio != 1:
            self.pointwise_1 = CNNBlock(in_channels, mid_channels, kernel_size=1, activation="ReLU6")
        else:
            self.pointwise_1 = nn.Identity()
        self.depthwise = CNNBlock(
            mid_channels,
            mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            activation="ReLU6",
            groups=mid_channels,
        )
        self.use_residual = stride == 1 and in_channels == out_channels
        self.pointiwise_2 = CNNBlock(mid_channels, out_channels, kernel_size=1, activation=None)

    def forward(self, x: Tensor) -> Tensor:
        out = self.pointwise_1(x)
        out = self.depthwise(out)
        out = self.pointiwise_2(out)
        if self.use_residual:
            return x + out
        return out


class MobilenetV2(FeatureExtractor):
    def __init__(self, in_channels: int, width_mul: float):
        round_nearest = 8
        conv_0_out_channels = 32
        conv_8_channels = 1280
        self.width_mul = width_mul
        conv_0_out_channels = _make_divisible(conv_0_out_channels * width_mul, round_nearest)
        self.conv_8_channels = _make_divisible(conv_8_channels * max(1, width_mul), round_nearest)

        conv_0 = CNNBlock(in_channels, conv_0_out_channels, kernel_size=3, stride=2, padding=1, activation="ReLU6")
        # t - expansion ratio, c - output channels, n - number of layers in block, s - stride
        net_config = [
            # t  c  n  s
            [1, 16, 1, 1],  # 1
            [6, 24, 2, 2],  # 2
            [6, 32, 3, 2],  # 3
            [6, 64, 4, 2],  # 4
            [6, 96, 3, 1],  # 5
            [6, 160, 3, 2],  # 6
            [6, 320, 1, 1],  # 7
        ]
        layers: list[tuple[str, nn.Module]] = [("conv_0", conv_0)]
        in_channels = conv_0_out_channels
        for i, (expansion_ratio, out_channels, n, stride) in enumerate(net_config):
            out_channels = _make_divisible(out_channels * width_mul, round_nearest)
            for j in range(n):
                stride = stride if j == 0 else 1  # first layer has stride specified in config table
                block = Bottleneck(in_channels, expansion_ratio, out_channels, stride)
                layers.append((f"bottleneck_{i}_{j}", block))
                in_channels = out_channels

        conv_8 = CNNBlock(in_channels, conv_8_channels, kernel_size=1, stride=1, activation="ReLU6")
        layers.append(("conv_8", conv_8))
        net = make_named_sequential(layers)
        super().__init__(net)

    @property
    def params(self) -> _any_dict:
        return {"width_multiplier": self.width_mul}

    @property
    def out_dim(self) -> int:
        return self.conv_8_channels
