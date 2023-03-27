import torch
from torch import nn


class CNNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | str = 0,
        pool_kernel_size: int = 1,
        use_batch_norm: bool = True,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.use_pool = pool_kernel_size > 1
        self.use_batch_norm = use_batch_norm
        self.use_dropout = dropout > 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()

        if self.use_pool:
            self.max_pool = nn.MaxPool2d(pool_kernel_size)

        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels)

        if self.use_dropout:
            self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.relu(out)
        if self.use_pool:
            out = self.max_pool(out)
        if self.use_batch_norm:
            out = self.batch_norm(out)
        if self.use_dropout:
            out = self.dropout(out)
        return out


class DeepCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: list[int],
        kernels: list[int],
        pool_kernels: list[int],
    ):
        super().__init__()
        layers = [
            CNNBlock(
                in_channels if i == 0 else out_channels[i - 1],
                out_channels[i],
                kernels[i],
                pool_kernel_size=pool_kernels[i],
            )
            for i in range(len(out_channels))
        ]
        self.net = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        out = self.net(x)
        out = self.global_pool(out)
        return out
