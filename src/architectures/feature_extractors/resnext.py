"""ResNext architecture based on https://arxiv.org/pdf/1611.05431.pdf
Implementation differs from the one present in torchvision models
in a sense that ResNext101 models have smaller filter depths
"""

import torchvision
from torch import nn

from src.architectures.feature_extractors.base import (
    ExternalFeatureExtractor,
    FeatureExtractor,
)
from src.architectures.feature_extractors.resnet import (
    BottleneckBlock,
    BottleneckResNetCore,
)
from src.architectures.helpers import CNNBlock
from src.utils.types import Any, Literal, Tensor, _size_2_t


class ResNextBottleneckBlock(BottleneckBlock):
    expansion = 2


class BaseResNext(FeatureExtractor):
    """Convolutional Neural Network with residual connextions"""

    def __init__(
        self,
        in_channels: int,
        stages_n_blocks: list[int],
        groups: int,
        stem_channels: int = 64,
        first_conv_out_channels: int = 128,
        stem_kernel_size: _size_2_t = 7,
        pool_kernel_size: _size_2_t = 3,
    ):
        """
        Args:
            in_channels (int): Number of image channels.
            stem_channels (int): Number of channels used in first Conv2d layer (before the residual layers).
            first_conv_out_channels (int): TODO
            stem_kernel_size (int): Kernel used in first Conv2d layer
                (before the residual layers) as '(stem_kernel_size, stem_kernel_size)`.
            pool_kernel_size (int): Kernel used in Pooling layer after the first Conv2d layer
                (before the residual layers) as '(pool_kernel_size, pool_kernel_size)`.
            stages_n_blocks (list[int]): Number of bottleneck blocks used per stage.
            block_type (Literal["basic", "bottleneck"], optional): Whether to use Basic block or Bottleneck block.
                Defaults to "basic".
        """
        self.stem_channels = stem_channels
        self.stem_kernel_size = stem_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.groups = groups
        self.stages_n_blocks = stages_n_blocks
        layers = [
            CNNBlock(
                in_channels,
                stem_channels,
                stem_kernel_size,
                stride=2,
                padding=3,
                pool_kernel_size=pool_kernel_size,
            ),
            BottleneckResNetCore(
                in_channels=stem_channels,
                first_conv_out_channels=first_conv_out_channels,
                stages_n_blocks=stages_n_blocks,
                groups=groups,
                block_class=ResNextBottleneckBlock,
            ),
        ]
        layers.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten(1, -1)])
        net = nn.Sequential(*layers)
        super().__init__(net)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "stem_channels": self.stem_channels,
            "stem_kernel_size": self.stem_kernel_size,
            "pool_kernel_size": self.pool_kernel_size,
            "block_type": self.block_type,
            "stages_n_blocks": self.stages_n_blocks,
            "groups": self.groups,
        }

    @property
    def out_dim(self) -> int:
        n_stages = len(self.stages_n_blocks)
        return self.stem_channels * 2 ** (n_stages - 1) * BottleneckBlock.expansion


class ResNext50_32x4d(BaseResNext):
    def __init__(self, in_channels: int):
        super().__init__(
            in_channels=in_channels,
            stem_channels=64,
            first_conv_out_channels=128,
            stages_n_blocks=[3, 4, 6, 3],
            groups=32,
        )


class ResNext101_32x8d(BaseResNext):
    def __init__(self, in_channels: int):
        super().__init__(
            in_channels=in_channels,
            stem_channels=64,
            first_conv_out_channels=128,
            stages_n_blocks=[3, 4, 23, 3],
            groups=32,
        )


class ResNext101_64x4d(BaseResNext):
    def __init__(self, in_channels: int):
        super().__init__(
            in_channels=in_channels,
            stem_channels=64,
            first_conv_out_channels=128,
            stages_n_blocks=[3, 4, 23, 3],
            groups=64,
        )


class ResNext:
    def __new__(
        cls,
        in_channels: int,
        version: Literal["resnext50_32x4d", "resnext101_32x8d", "resnext101_64x4d"],
        load_from_torch: bool = False,
        pretrained: bool = False,
        freeze_extractor: bool = False,
    ):
        version2cfg = {
            # local_class, torch_load_fn, out_channels
            "resnext50_32x4d": (ResNext50_32x4d, torchvision.models.resnext50_32x4d, 2048),
            "resnext101_32x8d": (ResNext101_32x8d, torchvision.models.resnext101_32x8d, 2048),
            "resnext101_64x4d": (ResNext101_64x4d, torchvision.models.resnext101_64x4d, 2048),
        }
        ModelClass, torch_load_fn, out_channels = version2cfg[version]
        if load_from_torch:
            params = dict(pretrained=pretrained) if pretrained else dict()
            _net = torch_load_fn(**params)
            _net.fc = nn.Identity()
            net = nn.Sequential(_net, nn.Flatten(1, -1))
            net = ExternalFeatureExtractor(net, out_channels=out_channels)
            if freeze_extractor:
                net.freeze()
        else:
            net = ModelClass(in_channels=in_channels)
        return net
