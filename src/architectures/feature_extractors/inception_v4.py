"""Inception architecture based on:
Main idea (v1): https://arxiv.org/abs/1409.4842.abs
+ Batch Normalization (v2): https://arxiv.org/pdf/1502.03167v3.pdf
+ factorization ideas (v3): https://arxiv.org/abs/1512.00567
+ Residual connections (Inception v4): https://arxiv.org/pdf/1602.07261.pdf
+ Number of channels based on: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/inception_resnet_v2.py
The Inception version implemented in this module is the Inception-ResNet-v2
"""

from collections import OrderedDict

import torch
from torch import nn

from src.architectures.feature_extractors.base import FeatureExtractor
from src.architectures.helpers import CNNBlock
from src.utils.types import Tensor


class StemConv(nn.Module):
    """Figure 3 from the https://arxiv.org/pdf/1602.07261.pdf paper"""

    out_channels: int = 192 + 192

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # layers 1 - 3 from fig. 3
        self.conv_1 = nn.Sequential(
            CNNBlock(in_channels, 32, kernel_size=3, stride=2, padding=0, use_batch_norm=True),
            CNNBlock(32, 32, kernel_size=3, stride=1, padding=0, use_batch_norm=True),
            CNNBlock(32, 64, kernel_size=3, stride=1, padding="same", use_batch_norm=True),
        )
        # layers 4_1 (left) and 4_2 (right) from fig. 3
        self.maxpool_4_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv_4_2 = CNNBlock(64, 96, kernel_size=3, stride=2, padding=0)

        # layers 5 - 8: 5_1 (left) and 5_2 (right) from fig. 3
        self.conv_5_1 = nn.Sequential(
            CNNBlock(96 + 64, 64, kernel_size=1, stride=1, padding="same"),
            CNNBlock(64, 96, kernel_size=3, stride=1, padding=0),
        )
        self.conv_5_2 = nn.Sequential(
            CNNBlock(96 + 64, 64, kernel_size=1, stride=1, padding="same"),
            CNNBlock(64, 64, kernel_size=(7, 1), stride=1, padding="same"),
            CNNBlock(64, 64, kernel_size=(1, 7), stride=1, padding="same"),
            CNNBlock(64, 96, kernel_size=3, stride=1, padding=0),
        )

        # layers 9_1 (left) and 9_2 (right) from fig. 3:
        self.maxpool_9_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv_9_2 = CNNBlock(96 + 96, 192, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv_1(x)
        maxpool_4_1_out = self.maxpool_4_1(out)
        conv_4_2_out = self.conv_4_2(out)
        out = torch.concat([maxpool_4_1_out, conv_4_2_out], dim=1)  # filter concat
        conv_5_1_out = self.conv_5_1(out)
        conv_5_2_out = self.conv_5_2(out)
        out = torch.concat([conv_5_1_out, conv_5_2_out], dim=1)  # filter concat
        maxpool_9_1_out = self.maxpool_9_1(out)
        conv_9_2_out = self.conv_9_2(out)
        out = torch.concat([maxpool_9_1_out, conv_9_2_out], dim=1)  # filter concat
        return out


class StemConv2(nn.Module):
    out_channels: int = 96 + 64 + 96 + 64

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv_1 = CNNBlock(in_channels, 32, kernel_size=3, stride=2, padding=0)
        self.conv_2 = CNNBlock(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv_3 = CNNBlock(32, 64, kernel_size=3, stride=1, padding="same")
        self.maxpool_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv_4 = CNNBlock(64, 80, kernel_size=1, stride=1, padding="same")
        self.conv_5 = CNNBlock(80, 192, kernel_size=3, stride=1, padding=0)
        self.maxpool_5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)  # in paper: conv with S=2

        self.branch_1 = CNNBlock(192, 96, kernel_size=1, stride=1)

        self.branch_2 = nn.Sequential(
            CNNBlock(192, 48, kernel_size=1, stride=1, padding=0),
            CNNBlock(48, 64, kernel_size=5, stride=1, padding=2),
        )

        self.branch_3 = nn.Sequential(
            CNNBlock(192, 64, kernel_size=1, stride=1, padding=0),
            CNNBlock(64, 96, kernel_size=3, stride=1, padding=1),
            CNNBlock(96, 96, kernel_size=3, stride=1, padding=1),
        )

        self.branch_4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            CNNBlock(192, 64, kernel_size=1, stride=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.maxpool_3(out)
        out = self.conv_4(out)
        out = self.conv_5(out)
        out = self.maxpool_5(out)

        out_1 = self.branch_1(out)
        out_2 = self.branch_2(out)
        out_3 = self.branch_3(out)
        out_4 = self.branch_4(out)
        out = torch.concat([out_1, out_2, out_3, out_4], dim=1)
        return out


class InceptionResNetA(nn.Module):
    """Figure 16 from the https://arxiv.org/pdf/1602.07261.pdf paper"""

    out_channels: int = 320

    def __init__(self, in_channels: int, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.branch_1 = CNNBlock(in_channels, 32, kernel_size=1, stride=1, padding="same")
        self.branch_2 = nn.Sequential(
            CNNBlock(in_channels, 32, kernel_size=1, stride=1, padding="same"),
            CNNBlock(32, 32, kernel_size=3, stride=1, padding="same"),
        )
        self.branch_3 = nn.Sequential(
            CNNBlock(in_channels, 32, kernel_size=1, stride=1, padding="same"),
            CNNBlock(32, 48, kernel_size=3, stride=1, padding="same"),
            CNNBlock(48, 64, kernel_size=3, stride=1, padding="same"),
        )
        self.out_conv = nn.Conv2d(32 + 32 + 64, self.out_channels, kernel_size=1, stride=1, padding="same")
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch_1_out = self.branch_1(x)
        branch_2_out = self.branch_2(x)
        branch_3_out = self.branch_3(x)
        out = torch.concat([branch_1_out, branch_2_out, branch_3_out], dim=1)
        out = self.out_conv(out)
        return self.relu(x + self.scale * out)


class InceptionResNetB(nn.Module):
    """Figure 17 from the https://arxiv.org/pdf/1602.07261.pdf paper"""

    out_channels: int = 1088

    def __init__(self, in_channels: int, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.branch_1 = CNNBlock(in_channels, 192, kernel_size=1, stride=1, padding="same")
        self.branch_2 = nn.Sequential(
            CNNBlock(in_channels, 128, kernel_size=1, stride=1, padding="same"),
            CNNBlock(128, 160, kernel_size=(1, 7), stride=1, padding="same"),
            CNNBlock(160, 192, kernel_size=(7, 1), stride=1, padding="same"),
        )
        self.out_conv = nn.Conv2d(192 + 192, self.out_channels, kernel_size=1, stride=1, padding="same")
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch_1_out = self.branch_1(x)
        branch_2_out = self.branch_2(x)
        out = torch.concat([branch_1_out, branch_2_out], dim=1)
        out = self.out_conv(out)
        return self.relu(x + self.scale * out)


class InceptionResNetC(nn.Module):
    """Figure 19 from the https://arxiv.org/pdf/1602.07261.pdf paper"""

    out_channels: int = 2080

    def __init__(self, in_channels: int, scale: float = 1.0, last_relu: bool = True):
        super().__init__()
        self.scale = scale
        self.branch_1 = CNNBlock(in_channels, 192, kernel_size=1, stride=1, padding="same")
        self.branch_2 = nn.Sequential(
            CNNBlock(in_channels, 192, kernel_size=1, stride=1, padding="same"),
            CNNBlock(192, 224, kernel_size=(1, 3), stride=1, padding="same"),
            CNNBlock(224, 256, kernel_size=(3, 1), stride=1, padding="same"),
        )
        self.out_conv = nn.Conv2d(192 + 256, self.out_channels, kernel_size=1, stride=1, padding="same")
        self.relu = nn.ReLU(inplace=True) if last_relu else None

    def forward(self, x):
        out_1 = self.branch_1(x)
        out_2 = self.branch_2(x)
        out = torch.concat([out_1, out_2], dim=1)
        out = self.out_conv(out)
        out = x + self.scale * out
        if self.relu is not None:
            return self.relu(out)
        return out


class ReductionA(nn.Module):
    """Reduction from 35x35 to 17x17
    Figure 7 from the https://arxiv.org/pdf/1602.07261.pdf paper"""

    def __init__(self, in_channels: int, k: int = 256, l: int = 256, m: int = 384, n: int = 384):
        super().__init__()
        self.in_channels = in_channels
        self.n = n
        self.m = m
        self.maxpool_1_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv_1_2 = CNNBlock(in_channels, n, kernel_size=3, stride=2, padding=0)
        self.conv_1_3 = nn.Sequential(
            CNNBlock(in_channels, k, kernel_size=1, stride=1, padding="same"),
            CNNBlock(k, l, kernel_size=3, stride=1, padding="same"),
            CNNBlock(l, m, kernel_size=3, stride=2, padding=0),
        )

    @property
    def out_channels(self):
        return self.in_channels + self.n + self.m

    def forward(self, x):
        maxpool_1_1_out = self.maxpool_1_1(x)
        conv_1_2_out = self.conv_1_2(x)
        conv_1_3_out = self.conv_1_3(x)
        out = torch.concat([maxpool_1_1_out, conv_1_2_out, conv_1_3_out], dim=1)  # filter concat
        return out


class ReductionB(nn.Module):
    """Reduction from 17x17 to 8x8
    Figure 18 from the https://arxiv.org/pdf/1602.07261.pdf paper"""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.branch_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.branch_2 = nn.Sequential(
            CNNBlock(in_channels, 256, kernel_size=1, stride=1, padding="same"),
            CNNBlock(256, 384, kernel_size=3, stride=2, padding=0),
        )
        self.branch_3 = nn.Sequential(
            CNNBlock(in_channels, 256, kernel_size=1, stride=1, padding="same"),
            CNNBlock(256, 288, kernel_size=3, stride=2, padding=0),
        )
        self.branch_4 = nn.Sequential(
            CNNBlock(in_channels, 256, kernel_size=1, stride=1, padding="same"),
            CNNBlock(256, 288, kernel_size=3, stride=1, padding="same"),
            CNNBlock(288, 320, kernel_size=3, stride=2, padding=0),
        )

    def forward(self, x):
        out_1 = self.branch_1(x)
        out_2 = self.branch_2(x)
        out_3 = self.branch_3(x)
        out_4 = self.branch_4(x)
        out = torch.concat([out_1, out_2, out_3, out_4], dim=1)  # filter concat
        return out

    @property
    def out_channels(self):
        return self.in_channels + 384 + 288 + 320


class InceptionResNetV2(FeatureExtractor):
    def __init__(self, in_channels: int, scales: list[float] = [0.17, 0.10, 0.20]):
        Stem = StemConv2
        A_in_channels = [Stem.out_channels] + [InceptionResNetA.out_channels] * 4
        reduction_A = ReductionA(in_channels=A_in_channels[-1])
        B_in_channels = [reduction_A.out_channels] + [InceptionResNetB.out_channels] * 9
        reduction_B = ReductionB(B_in_channels[-1])
        C_in_channels = [reduction_B.out_channels] + [InceptionResNetC.out_channels] * 4

        self.scales = scales
        self.A_in_channels = A_in_channels
        self.B_in_channels = B_in_channels
        self.C_in_channels = C_in_channels
        self.out_channels = 1536

        net = nn.Sequential(
            OrderedDict(
                [("stem", Stem(in_channels=in_channels))]
                + [
                    (
                        f"block_A_{i}",
                        InceptionResNetA(in_channels=A_in_channels[i], scale=scales[0]),
                    )
                    for i in range(len(A_in_channels))
                ]
                + [("reduction_A", reduction_A)]
                + [
                    (
                        f"block_B_{i}",
                        InceptionResNetB(in_channels=B_in_channels[i], scale=scales[1]),
                    )
                    for i in range(len(B_in_channels))
                ]
                + [("reduction_B", reduction_B)]
                + [
                    (
                        f"block_C_{i}",
                        InceptionResNetC(in_channels=C_in_channels[i], scale=scales[2]),
                    )
                    for i in range(len(C_in_channels) - 1)
                ]
                + [
                    (
                        f"block_C_{len(C_in_channels) - 1}",
                        InceptionResNetC(in_channels=C_in_channels[-1], scale=1.0, last_relu=False),
                    ),
                    (
                        "last_conv",
                        CNNBlock(
                            InceptionResNetC.out_channels,
                            self.out_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                        ),
                    ),
                ]
            )
        )
        super().__init__(net)

    @property
    def params(self):
        return {"scales": self.scales, "num_features": self.out_channels}

    @property
    def out_dim(self) -> int:
        return self.out_channels
