import itertools
from typing import Literal

import pytest
import torch

from src.architectures.feature_extractors.resnet import BaseResNet, ResNet
from src.utils.types import _size_2_t


@pytest.mark.parametrize(
    "batch_X, stem_channels, stem_kernel_size, pool_kernel_size, stages_n_blocks, block_type",
    itertools.product(
        ["resnet_data_batch"],
        [16, 64],
        [1, 5, 9],
        [1, 5, 9],
        [[1], [1, 2], [2, 2, 2]],
        ["basic", "bottleneck"],
    ),
)
def test_out_dim(
    batch_X: torch.Tensor,
    stem_channels: int,
    stem_kernel_size: _size_2_t,
    pool_kernel_size: _size_2_t,
    stages_n_blocks: list[int],
    block_type: Literal["basic", "bottleneck"],
    request,
):
    batch_X = request.getfixturevalue(batch_X)
    in_channels = batch_X[0].shape[0]
    model = BaseResNet(
        in_channels=in_channels,
        stem_channels=stem_channels,
        stem_kernel_size=stem_kernel_size,
        pool_kernel_size=pool_kernel_size,
        stages_n_blocks=stages_n_blocks,
        block_type=block_type,
    )
    out = model(batch_X)
    assert out[0].shape[0] == model.out_dim


@pytest.mark.parametrize(
    "batch_X, version, load_from_torch, freeze_extractor",
    itertools.product(
        ["resnet_data_batch"],
        ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],  # version
        [True, False],  # load_from_torch
        [True, False],  # freeze_extractor
    ),
)
def test_out_dim_versioned(
    batch_X: torch.Tensor,
    version: Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
    load_from_torch: bool,
    freeze_extractor: bool,
    request,
):
    batch_X = request.getfixturevalue(batch_X)
    in_channels = batch_X[0].shape[0]
    model = ResNet(
        in_channels=in_channels,
        version=version,
        load_from_torch=load_from_torch,
        freeze_extractor=freeze_extractor,
    )
    out = model(batch_X)
    assert out[0].shape[0] == model.out_dim
