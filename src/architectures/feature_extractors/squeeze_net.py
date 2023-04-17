from collections import OrderedDict

import torch
from torch import nn

from src.architectures.feature_extractors.base import FeatureExtractor
from src.utils.types import Any, Tensor


class FireBlock(nn.Module):
    def __init__(self, in_channels: int, squeeze_ratio: float, expand_filters: int, pct_3x3: float):
        super().__init__()
        s_1x1 = int(squeeze_ratio * expand_filters)
        e_3x3 = int(expand_filters * pct_3x3)
        e_1x1 = int(expand_filters * (1 - pct_3x3))
        self.squeeze_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=s_1x1, kernel_size=1)
        self.expand_1x1 = nn.Conv2d(in_channels=s_1x1, out_channels=e_1x1, kernel_size=1)
        self.expand_3x3 = nn.Conv2d(in_channels=s_1x1, out_channels=e_3x3, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        squeeze_out = self.relu(self.squeeze_1x1(x))
        expand_1x1_out = self.expand_1x1(squeeze_out)
        expand_3x3_out = self.expand_3x3(squeeze_out)
        out = torch.concat([expand_1x1_out, expand_3x3_out], dim=1)  # concat over channels
        return self.relu(out)


class SqueezeNet(FeatureExtractor):
    def __init__(
        self,
        in_channels: int = 3,
        base_e: int = 128,
        incr_e: int = 128,
        pct_3x3: float = 0.5,
        freq: int = 2,
        SR: float = 0.125,
    ):
        self.in_channels = in_channels
        self.base_e = base_e
        self.incr_e = incr_e
        self.pct_3x3 = pct_3x3
        self.freq = freq
        self.SR = SR

        # architecture, fb - fire block
        out_channels = 96

        fb_expand_filters = [base_e + (incr_e * i // freq) for i in range(8)]
        fb_in_channels = [out_channels] + fb_expand_filters
        self.out_channels = fb_expand_filters[-1]
        net = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2)),
                    ("maxpool1", nn.MaxPool2d(kernel_size=3, stride=2)),
                    ("fire2", FireBlock(fb_in_channels[0], SR, fb_expand_filters[0], pct_3x3)),
                    ("fire3", FireBlock(fb_in_channels[1], SR, fb_expand_filters[1], pct_3x3)),
                    ("fire4", FireBlock(fb_in_channels[2], SR, fb_expand_filters[2], pct_3x3)),
                    ("maxpool4", nn.MaxPool2d(kernel_size=3, stride=2)),
                    ("fire5", FireBlock(fb_in_channels[3], SR, fb_expand_filters[3], pct_3x3)),
                    ("fire6", FireBlock(fb_in_channels[4], SR, fb_expand_filters[4], pct_3x3)),
                    ("fire7", FireBlock(fb_in_channels[5], SR, fb_expand_filters[5], pct_3x3)),
                    ("fire8", FireBlock(fb_in_channels[6], SR, fb_expand_filters[6], pct_3x3)),
                    ("maxpool8", nn.MaxPool2d(kernel_size=3, stride=2)),
                    ("fire9", FireBlock(fb_in_channels[7], SR, fb_expand_filters[7], pct_3x3)),
                    ("dropout9", nn.Dropout2d(p=0.5)),
                ]
            )
        )
        super().__init__(net)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "in_channels": self.in_channels,
            "base_e": self.base_e,
            "incr_e": self.incr_e,
            "pct_3x3": self.pct_3x3,
            "freq": self.freq,
            "SR": self.SR,
        }

    @property
    def out_dim(self) -> int:
        return self.out_channels
