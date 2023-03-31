"""Implementation based on https://arxiv.org/pdf/1512.03385.pdf 
with modifications suggested in https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py"""
from torch import nn
from typing import Literal
from collections import OrderedDict
from .base import FeatureExtractor
from src.utils.types import _size_2_t
from torchtyping import TensorType


class BasicBlock(nn.Module):
    """Basic Residual Block.
    Pipeline without downsampling:
    x -> Conv -> BN -> ReLU -> Conv -> BN -> ReLU(out + x)

    If downsampling is present, then shortcut connection is added:
    Shortcut = Conv1x1 -> BN
    x -> Conv -> BN -> ReLU -> Conv -> BN -> ReLU(out + Shortcut(x))
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t,
        downsample: bool = False,
    ):
        """
        Args:
            in_channels (int): Number of block input channels.
            out_channels (int): Number of block output channels.
            kernel_size (_size_2_t): Kernel used for both Conv2d layers.
            stride (_size_2_t): Stride used in first Conv2d layer and in optional shortcut.
            downsample (bool, optional): Whether to apply downsampling. Defaults to False.
        """
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

    def forward(
        self, x: TensorType["batch", "in_channels", "in_height", "in_width"]
    ) -> TensorType["batch", "out_channels", "out_height", "out_width"]:
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
    """Bottleneck Residual Block.
    Pipeline without downsampling:
    x -> Conv1x1 -> BN -> ReLU -> Conv -> BN -> ReLU -> Conv1x1 -> BN -> ReLU(out + x)

    If downsampling is present, then shortcut connection is added:
    Shortcut = Conv1x1 -> BN
    x -> Conv1x1 -> BN -> ReLU -> Conv -> BN -> ReLU -> Conv1x1 -> BN -> ReLU(out + Shortcut(x))
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t,
        downsample: bool = False,
    ):
        """
        Args:
            in_channels (int): Number of block input channels.
            mid_channels (int): Number of block bottleneck channels.
            kernel_size (_size_2_t): Kernel used for middle Conv2d layer.
            stride (_size_2_t): Stride used in middle Conv2d layer and in optional shortcut.
            downsample (bool, optional): Whether to apply downsampling. Defaults to False.
        """

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

    def forward(
        self, x: TensorType["batch", "in_channels", "in_height", "in_width"]
    ) -> TensorType["batch", "out_channels", "out_height", "out_width"]:
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


class BasicResNetCore(nn.Module):
    """Core Basic Residual layers"""

    def __init__(
        self,
        in_channels: int,
        stages_n_blocks: list[int],
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            stages_n_blocks (list[int]): Number of bottleneck blocks used per stage.
        """
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

    def forward(
        self, x: TensorType["batch", "in_channels", "in_height", "in_width"]
    ) -> TensorType["batch", "out_channels", "out_height", "out_width"]:
        return self.net(x)


class BottleneckResNetCore(nn.Module):
    """Core Bottleneck Residual layers"""

    def __init__(
        self,
        in_channels: int,
        stages_n_blocks: list[int],
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            stages_n_blocks (list[int]): Number of bottleneck blocks used per stage.
        """
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

    def forward(
        self, x: TensorType["batch", "in_channels", "in_height", "in_width"]
    ) -> TensorType["batch", "out_channels", "out_height", "out_width"]:
        return self.net(x)


class ResNet(FeatureExtractor):
    """Convolutional Neural Network with residual connextions"""

    name: str = "ResNet"

    def __init__(
        self,
        in_channels: int,
        stem_channels: int,
        stem_kernel_size: _size_2_t,
        pool_kernel_size: _size_2_t,
        stages_n_blocks: list[int],
        block_type: Literal["basic", "bottleneck"] = "basic",
    ):
        """
        Args:
            in_channels (int): Number of image channels.
            stem_channels (int): Number of channels used in first Conv2d layer (before the residual layers).
            stem_kernel_size (int): Kernel used in first Conv2d layer
                (before the residual layers) as '(stem_kernel_size, stem_kernel_size)`.
            pool_kernel_size (int): Kernel used in Pooling layer after the first Conv2d layer
                (before the residual layers) as '(pool_kernel_size, pool_kernel_size)`.
            stages_n_blocks (list[int]): Number of bottleneck blocks used per stage.
            block_type (Literal["basic", "bottleneck"], optional): Whether to use Basic block or Bottleneck block.
                Defaults to "basic".
        """
        super().__init__()
        self.stem_channels = stem_channels
        self.block_type = block_type
        self.stages_n_blocks = stages_n_blocks
        ResnetCoreBlocks = BasicResNetCore if block_type == "basic" else BottleneckResNetCore
        self.net = nn.Sequential(
            OrderedDict(
                [
                    (
                        "stem_conv",
                        nn.Conv2d(in_channels, stem_channels, stem_kernel_size, stride=2),
                    ),
                    ("maxpool", nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)),
                    (
                        "residual_layers",
                        ResnetCoreBlocks(
                            in_channels=stem_channels, stages_n_blocks=stages_n_blocks
                        ),
                    ),
                    ("global_pool", nn.AdaptiveAvgPool2d((1, 1))),
                    ("flatten", nn.Flatten()),
                ]
            )
        )

    @property
    def out_dim(self) -> int:
        n_stages = len(self.stages_n_blocks)
        if self.block_type == "basic":
            return self.stem_channels * 2 ** (n_stages - 1)
        else:
            return self.stem_channels * 2 ** (n_stages - 1) * BottleneckBlock.expansion
