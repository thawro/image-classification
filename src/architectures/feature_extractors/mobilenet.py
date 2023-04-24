"""MobiletNet architecture based on:
Main idea, depthwise-separable convolutions (v1): https://arxiv.org/pdf/1704.04861v1.pdf
+ inverted linear bottleneck (v2): https://arxiv.org/pdf/1801.04381.pdf
+ squeeze and excite (v3): https://arxiv.org/pdf/1905.02244v5.pdf

TODO: check why pytorch version removes first pointwise convolution and SEBlock for last bottlenecks
"""

from collections import namedtuple

from torch import nn
from torchvision.models import (
    MobileNet_V2_Weights,
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    mobilenet_v2,
    mobilenet_v3_large,
    mobilenet_v3_small,
)
from torchvision.models._utils import _make_divisible

from src.architectures.feature_extractors.base import (
    ExternalFeatureExtractor,
    FeatureExtractor,
)
from src.architectures.helpers import CNNBlock, OutChannelsModule, SEBlock
from src.architectures.utils import get_padding, make_named_sequential
from src.utils.types import Literal, Tensor, _any_dict, _size_2_t

# kernel - kernel size of depthwise convolution
# exp_size - expansion size
# out - out channels
# SE - whether to use Squeeze and Excitation
# NL - type of nonlinearity after first convolutions
# s - stride of depthwise convolution
BottleneckConfig = namedtuple("BottleneckConfig", ["kernel", "exp_size", "out", "SE", "NL", "s"])


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        expansion_size: int,
        out_channels: int,
        stride: int,
        kernel_size: _size_2_t = 3,
        use_SE: bool = False,
        activation: str = "ReLU6",
    ):
        super().__init__()
        self.use_SE = use_SE
        self.use_residual = stride == 1 and in_channels == out_channels

        self.in_channels = in_channels
        self.expansion_size = expansion_size
        self.out_channels = out_channels

        self.pointwise_1 = CNNBlock(in_channels, expansion_size, kernel_size=1, activation=activation)
        depthwise_block = CNNBlock(
            expansion_size,
            expansion_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=get_padding(kernel_size, stride),
            activation=activation,
            groups=expansion_size,
        )
        if use_SE:
            self.depthwise = SEBlock(
                depthwise_block,
                reduce_activation="ReLU",
                expand_activation="Hardsigmoid",
                reduction_ratio=4,
            )
        else:
            self.depthwise = depthwise_block
        self.pointiwise_2 = CNNBlock(expansion_size, out_channels, kernel_size=1, activation=None)

    def forward(self, x: Tensor) -> Tensor:
        out = self.pointwise_1(x)
        out = self.depthwise(out)
        out = self.pointiwise_2(out)
        if self.use_residual:
            return x + out
        return out


class OriginalLastStage(OutChannelsModule):
    """Last stage used in MobilenetV2"""

    def __init__(self, in_channels: int, expansion_size: int, mid_channels: int, out_channels: int):
        super().__init__()
        self.bottleneck_1 = Bottleneck(
            in_channels,
            expansion_size,
            mid_channels,
            stride=1,
            kernel_size=3,
            activation="Hardswish",
        )
        self.conv_2 = CNNBlock(mid_channels, out_channels, kernel_size=1, stride=1, activation="Hardswish")
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1, -1)

    @property
    def out_channels(self):
        return self.conv_2.out_channels

    def forward(self, x: Tensor) -> Tensor:
        out = self.bottleneck_1(x)
        out = self.conv_2(out)
        out = self.global_pool(out)
        out = self.flatten(out)
        return out


class EfficientLastStage(OutChannelsModule):
    """Efficient Last stage used in MobilenetV3"""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        use_SE: bool,
    ):
        super().__init__()
        conv_1 = CNNBlock(in_channels, mid_channels, kernel_size=1, activation="Hardswish")
        if use_SE:
            self.conv_1 = SEBlock(
                conv_1,
                reduce_activation="ReLU",
                expand_activation="Hardsigmoid",
                reduction_ratio=4,
            )
        else:
            self.conv_1 = conv_1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_2 = CNNBlock(mid_channels, out_channels, kernel_size=1, activation="Hardswish", use_batch_norm=False)

    @property
    def out_channels(self):
        return self.conv_2.out_channels

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv_1(x)
        out = self.global_pool(out)
        out = self.conv_2(out)
        return out


class BaseMobileNet(FeatureExtractor):
    round_nearest: int = 8

    def __init__(
        self,
        conv_0: CNNBlock,
        width_mul: float,
        bottlenecks_config: list[BottleneckConfig],
        last_stage: OutChannelsModule,
    ):
        self.width_mul = width_mul
        self.stem_channels = conv_0.out_channels
        bottleneck_layers: list[tuple[str, nn.Module]] = []
        in_channels = conv_0.out_channels
        for i, cfg in enumerate(bottlenecks_config):
            kernel_size, expansion_size, out_channels, use_SE, activation, stride = cfg
            out_channels = _make_divisible(out_channels * width_mul, self.round_nearest)
            block = Bottleneck(in_channels, expansion_size, out_channels, stride, kernel_size, use_SE, activation)
            bottleneck_layers.append((f"bottleneck_{i}", block))
            in_channels = out_channels

        self.out_channels = last_stage.out_channels
        layers = [("conv_0", conv_0), *bottleneck_layers, ("last_stage", last_stage)]
        net = make_named_sequential(layers)
        super().__init__(net)

    @property
    def out_dim(self) -> int:
        return self.out_channels

    @property
    def params(self) -> _any_dict:
        return {"width_multiplier": self.width_mul, "stem_channels": self.stem_channels}


class MobileNetV2(BaseMobileNet):
    def __init__(self, in_channels: int, width_mul: float):
        # flattened version of:
        # net_config = [
        # t  c  n  s
        # [1, 16, 1, 1],  # 0
        # [6, 24, 2, 2],  # 1
        # [6, 32, 3, 2],  # 2
        # [6, 64, 4, 2],  # 3
        # [6, 96, 3, 1],  # 4
        # [6, 160, 3, 2],  # 5
        # [6, 320, 1, 1],  # 6 - used for last_stage
        # ]
        # kernel, exp_size, out, SE, NL, s
        bottlenecks_config = [
            (3, 32, 16, False, "ReLU6", 1),  # 0 (0)
            (3, 96, 24, False, "ReLU6", 2),  # 1 (1)
            (3, 144, 24, False, "ReLU6", 1),  # 2 (1)
            (3, 144, 32, False, "ReLU6", 2),  # 3 (2)
            (3, 192, 32, False, "ReLU6", 1),  # 4 (2)
            (3, 192, 32, False, "ReLU6", 1),  # 5 (2)
            (3, 192, 64, False, "ReLU6", 2),  # 6 (3)
            (3, 384, 64, False, "ReLU6", 1),  # 7 (3)
            (3, 384, 64, False, "ReLU6", 1),  # 8 (3)
            (3, 384, 64, False, "ReLU6", 1),  # 9 (3)
            (3, 384, 96, False, "ReLU6", 1),  # 10 (4)
            (3, 576, 96, False, "ReLU6", 1),  # 11 (4)
            (3, 576, 96, False, "ReLU6", 1),  # 12 (4)
            (3, 576, 160, False, "ReLU6", 2),  # 13 (5)
            (3, 960, 160, False, "ReLU6", 1),  # 14 (5)
            (3, 960, 160, False, "ReLU6", 1),  # 15 (5)
            (3, 960, 320, False, "ReLU6", 1),  # 16 (6) - used in last stage
        ]
        bottlenecks_cfg = [BottleneckConfig(*cfg) for cfg in bottlenecks_config[:-1]]
        last_btnck_cfg = BottleneckConfig(*bottlenecks_config[-1])

        conv_0 = CNNBlock(in_channels, 32, kernel_size=3, stride=2, padding=1, activation="ReLU6")
        last_stage = OriginalLastStage(
            in_channels=_make_divisible(bottlenecks_cfg[-1].out * width_mul, self.round_nearest),
            expansion_size=last_btnck_cfg.exp_size,
            mid_channels=_make_divisible(last_btnck_cfg.out * width_mul, self.round_nearest),
            out_channels=_make_divisible(1280 * max(1, width_mul), self.round_nearest),
        )
        super().__init__(
            conv_0=conv_0,
            width_mul=width_mul,
            bottlenecks_config=bottlenecks_cfg,
            last_stage=last_stage,
        )


class MobileNetV3Small(BaseMobileNet):
    def __init__(self, in_channels: int, width_mul: float):
        bottlenecks_config = [
            (3, 16, 16, True, "ReLU6", 2),  # 0
            (3, 72, 24, False, "ReLU6", 2),  # 1
            (3, 88, 24, False, "ReLU6", 1),  # 2
            (5, 96, 40, True, "Hardswish", 2),  # 3
            (5, 240, 40, True, "Hardswish", 1),  # 4
            (5, 240, 40, True, "Hardswish", 1),  # 5
            (5, 120, 48, True, "Hardswish", 1),  # 6
            (5, 144, 48, True, "Hardswish", 1),  # 7
            (5, 288, 96, True, "Hardswish", 2),  # 8
            (5, 576, 96, True, "Hardswish", 1),  # 9
            (5, 576, 96, True, "Hardswish", 1),  # 10
        ]
        bottlenecks_cfg = [BottleneckConfig(*cfg) for cfg in bottlenecks_config]

        conv_0 = CNNBlock(in_channels, 16, kernel_size=3, stride=2, padding=1, activation="Hardswish")
        last_stage = EfficientLastStage(
            in_channels=_make_divisible(bottlenecks_cfg[-1].out * width_mul, self.round_nearest),
            mid_channels=_make_divisible(576 * width_mul, self.round_nearest),
            out_channels=_make_divisible(1024 * max(1, width_mul), self.round_nearest),
            use_SE=True,
        )

        super().__init__(
            conv_0=conv_0,
            width_mul=width_mul,
            bottlenecks_config=bottlenecks_cfg,
            last_stage=last_stage,
        )


class MobileNetV3Large(BaseMobileNet):
    def __init__(self, in_channels: int, width_mul: float):
        bottlenecks_config = [
            (3, 16, 16, False, "ReLU6", 1),  # 1
            (3, 64, 24, False, "ReLU6", 2),  # 2
            (3, 72, 24, False, "ReLU6", 1),  # 3
            (5, 72, 40, True, "ReLU6", 2),  # 4
            (5, 120, 40, True, "ReLU6", 1),  # 5
            (5, 120, 40, True, "ReLU6", 1),  # 6
            (3, 240, 80, False, "Hardswish", 2),  # 7
            (3, 200, 80, False, "Hardswish", 1),  # 8
            (3, 184, 80, False, "Hardswish", 1),  # 9
            (3, 184, 80, False, "Hardswish", 1),  # 10
            (3, 480, 112, True, "Hardswish", 1),  # 12
            (3, 672, 112, True, "Hardswish", 1),  # 13
            (5, 672, 160, True, "Hardswish", 2),  # 14
            (5, 960, 160, True, "Hardswish", 1),  # 15
            (5, 960, 160, True, "Hardswish", 1),  # 16
        ]
        bottlenecks_cfg = [BottleneckConfig(*cfg) for cfg in bottlenecks_config]

        conv_0 = CNNBlock(in_channels, 16, kernel_size=3, stride=2, padding=1, activation="Hardswish")
        last_stage = EfficientLastStage(
            in_channels=_make_divisible(bottlenecks_cfg[-1].out * width_mul, self.round_nearest),
            mid_channels=_make_divisible(960 * width_mul, self.round_nearest),
            out_channels=_make_divisible(1280 * max(1, width_mul), self.round_nearest),
            use_SE=False,
        )

        super().__init__(
            conv_0=conv_0,
            width_mul=width_mul,
            bottlenecks_config=bottlenecks_cfg,
            last_stage=last_stage,
        )


class MobileNet:
    def __new__(
        cls,
        in_channels: int,
        width_mul: float,
        version: Literal["v2", "v3_small", "v3_large"],
        load_from_torch: bool = False,
        pretrained: bool = False,
        freeze_extractor: bool = False,
    ):
        if load_from_torch:
            all_params = {
                # load_fn, weights, out_channels
                "v2": (mobilenet_v2, MobileNet_V2_Weights, 1280),
                "v3_small": (mobilenet_v3_small, MobileNet_V3_Small_Weights, 1024),
                "v3_large": (mobilenet_v3_large, MobileNet_V3_Large_Weights, 1280),
            }

            last_layers = [nn.AdaptiveAvgPool2d(1), nn.Flatten(1, -1)]
            load_fn, weights, out_channels = all_params[version]
            if pretrained:
                _net = load_fn(weights=weights)
            else:
                if version in ["v3_small", "v3_large"]:  # out_channels are scaled in V3 versions
                    out_channels = _make_divisible(out_channels * width_mul, 8)
                _net = load_fn(width_mult=width_mul)
            if version in ["v3_small", "v3_large"]:
                last_layers.extend([_net.classifier[0], _net.classifier[1]])
            net = nn.Sequential(_net.features, *last_layers)
            mobilenet = ExternalFeatureExtractor(net, out_channels=out_channels)
            if freeze_extractor:
                mobilenet.freeze()
        else:
            if version == "v2":
                mobilenet = MobileNetV2(in_channels, width_mul)
            elif version == "v3_small":
                mobilenet = MobileNetV3Small(in_channels, width_mul)
            elif version == "v3_large":
                mobilenet = MobileNetV3Large(in_channels, width_mul)
        return mobilenet
