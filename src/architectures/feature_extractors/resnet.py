import torch
from torch import nn
from typing import Literal
from collections import OrderedDict
from .base import FeatureExtractor


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        downsample: bool = False,
    ):
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            out_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.downsample:
            identity = self.shortcut(x)
        else:
            identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out + identity)


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        kernel_size: int,
        stride: int,
        downsample: bool = False,
    ):
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            out_channels = mid_channels * self.expansion
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            out_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=1, stride=stride)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.downsample:
            identity = self.shortcut(x)
        else:
            identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        return self.relu(out + identity)


class BottleneckResNetCore(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stages_n_blocks: list[int],
    ):
        super().__init__()
        mid_channels = in_channels
        layers = []
        for stage, n_blocks in enumerate(stages_n_blocks):
            is_first_stage = stage == 0
            blocks_layers = []
            for i in range(n_blocks):
                is_first_block = i == 0
                stride = 2 if not is_first_stage and is_first_block else 1
                block = BottleneckBlock(
                    in_channels=in_channels,
                    mid_channels=mid_channels,
                    kernel_size=3,
                    stride=stride,
                    downsample=is_first_block,
                )
                in_channels = mid_channels * block.expansion
                blocks_layers.append(block)
            mid_channels *= 2
            blocks_layers = nn.Sequential(*blocks_layers)
            layers.append(blocks_layers)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BasicResNetCore(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stages_n_blocks: list[int],
    ):
        super().__init__()
        out_channels = in_channels
        layers = []
        for stage, n_blocks in enumerate(stages_n_blocks):
            is_first_stage = stage == 0
            blocks_layers = []
            for i in range(n_blocks):
                is_first_block = i == 0
                downsample = not is_first_stage and is_first_block
                stride = 2 if downsample else 1
                block = BasicBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    kernel_size=3,
                    downsample=downsample,
                )
                if not is_first_stage:
                    in_channels = out_channels
                blocks_layers.append(block)
            in_channels, out_channels = out_channels, in_channels * 2
            blocks_layers = nn.Sequential(*blocks_layers)
            layers.append(blocks_layers)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResNet(FeatureExtractor):
    def __init__(
        self,
        in_channels: int,
        stem_channels: int,
        stem_kernel_size: int,
        pool_kernel_size: int,
        stages_n_blocks: list[int],
        block_type: Literal["basic", "bottleneck"] = "basic",
    ):
        super().__init__()
        self.stem_channels = stem_channels
        self.block_type = block_type
        self.stages_n_blocks = stages_n_blocks
        ResnetCoreBlocks = BasicResNetCore if block_type == "basic" else BottleneckResNetCore
        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("stem_conv", nn.Conv2d(in_channels, stem_channels, stem_kernel_size, stride=2)),
                    ("maxpool", nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)),
                    ("residual_layers", ResnetCoreBlocks(in_channels=stem_channels, stages_n_blocks=stages_n_blocks)),
                    ("global_pool", nn.AdaptiveAvgPool2d((1, 1))),
                    ("flatten", nn.Flatten()),
                ]
            )
        )

    @property
    def out_shape(self):
        n_stages = len(self.stages_n_blocks)
        if self.block_type == "basic":
            return self.stem_channels * 2 ** (n_stages - 1)
        else:
            return self.stem_channels * 2 ** (n_stages - 1) * BottleneckBlock.expansion
