"""VGG architecture based on
https://arxiv.org/pdf/1409.1556.pdf
"""

from abc import abstractmethod
from collections import namedtuple
from typing import Literal

import torch
import torchvision
from torch import nn

from src.architectures.feature_extractors.base import (
    ExternalFeatureExtractor,
    FeatureExtractor,
)
from src.architectures.helpers import CNNBlock
from src.utils.types import _any_dict

CNNConfig = namedtuple("CNNConfig", ["kernel", "out", "use_maxpool"])


class BaseVGG(FeatureExtractor):
    def __init__(
        self,
        in_channels: int,
        use_batch_norm: bool = False,
        use_global_pool: bool = False,
    ):
        self.use_batch_norm = use_batch_norm
        self.use_global_pool = use_global_pool
        layers = []
        for cfg in self.layers_config:
            pool_kernel_size = 2 if cfg.use_maxpool else 1
            layer = CNNBlock(
                in_channels,
                cfg.out,
                cfg.kernel,
                stride=1,
                padding="same",
                pool_kernel_size=pool_kernel_size,
                use_batch_norm=use_batch_norm,
            )
            layers.append(layer)
            in_channels = cfg.out
        if use_global_pool:
            layers.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten(1, -1)])
        self.out_channels = in_channels
        net = nn.Sequential(*layers)
        super().__init__(net)

    @property
    @abstractmethod
    def layers_config(self) -> list[CNNConfig]:
        raise NotImplementedError()

    @property
    def params(self) -> _any_dict:
        return {"use_batch_norm": self.use_batch_norm, "use_global_pool": self.use_global_pool}

    @property
    def out_dim(self) -> int:
        return self.out_channels


class VGG11(BaseVGG):
    """Table 1.A"""

    @property
    def layers_config(self) -> list[CNNConfig]:
        cnn_config = [
            # kernel, out, maxpool
            (3, 64, True),
            (3, 128, True),
            (3, 256, False),
            (3, 256, True),
            (3, 512, False),
            (3, 512, True),
            (3, 512, False),
            (3, 512, True),
        ]
        return [CNNConfig(*cfg) for cfg in cnn_config]


class VGG13(BaseVGG):
    """Table 1.B"""

    @property
    def layers_config(self) -> list[CNNConfig]:
        cnn_config = [
            # kernel, out, maxpool
            (3, 64, False),
            (3, 64, True),
            (3, 128, False),
            (3, 128, True),
            (3, 256, False),
            (3, 256, True),
            (3, 512, False),
            (3, 512, True),
            (3, 512, False),
            (3, 512, True),
        ]
        return [CNNConfig(*cfg) for cfg in cnn_config]


class VGG16(BaseVGG):
    """Table 1.D"""

    @property
    def layers_config(self) -> list[CNNConfig]:
        cnn_config = [
            # kernel, out, maxpool
            (3, 64, False),
            (3, 64, True),
            (3, 128, False),
            (3, 128, True),
            (3, 256, False),
            (3, 256, False),
            (3, 256, True),
            (3, 512, False),
            (3, 512, False),
            (3, 512, True),
            (3, 512, False),
            (3, 512, False),
            (3, 512, True),
        ]
        return [CNNConfig(*cfg) for cfg in cnn_config]


class VGG19(BaseVGG):
    """Table 1.E"""

    @property
    def layers_config(self) -> list[CNNConfig]:
        cnn_config = [
            # kernel, out, maxpool
            (3, 64, False),
            (3, 64, True),
            (3, 128, False),
            (3, 128, True),
            (3, 256, False),
            (3, 256, False),
            (3, 256, False),
            (3, 256, True),
            (3, 512, False),
            (3, 512, False),
            (3, 512, False),
            (3, 512, True),
            (3, 512, False),
            (3, 512, False),
            (3, 512, False),
            (3, 512, True),
        ]
        return [CNNConfig(*cfg) for cfg in cnn_config]


class VGG:
    def __new__(
        cls,
        in_channels: int,
        version: Literal["vgg11", "vgg13", "vgg16", "vgg19"],
        use_batch_norm: bool = False,
        use_global_pool: bool = False,
        load_from_torch: bool = False,
        pretrained: bool = False,
        freeze_extractor: bool = False,
    ):
        if load_from_torch:
            if use_global_pool:
                last_layers = [nn.AdaptiveAvgPool2d(1), nn.Flatten(1, -1)]
            else:
                last_layers = []
            suffix = "_bn" if use_batch_norm else ""
            load_fn = getattr(torchvision.models, f"{version}{suffix}")
            params = dict(pretrained=pretrained) if pretrained else dict()
            _net = load_fn(**params)
            net = nn.Sequential(_net.features, *last_layers)
            net = ExternalFeatureExtractor(net, out_channels=512)
            if freeze_extractor:
                net.freeze()
        else:
            version2class = {"vgg11": VGG11, "vgg13": VGG13, "vgg16": VGG16, "vgg19": VGG19}
            VGGClass: BaseVGG = version2class[version]
            net = VGGClass(
                in_channels=in_channels,
                use_batch_norm=use_batch_norm,
                use_global_pool=use_global_pool,
            )
        return net
