import pytest
from src.architectures.feature_extractors.resnet import ResNet
import itertools
import torch
from src.tests.conftest import TRANSFORMED_BATCH_DATA
from typing import Literal
from src.utils.types import _size_2_t


@pytest.mark.parametrize(
    "batch_X, stem_channels, stem_kernel_size, pool_kernel_size, stages_n_blocks, block_type",
    itertools.product(
        TRANSFORMED_BATCH_DATA,
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
    model = ResNet(
        in_channels=in_channels,
        stem_channels=stem_channels,
        stem_kernel_size=stem_kernel_size,
        pool_kernel_size=pool_kernel_size,
        stages_n_blocks=stages_n_blocks,
        block_type=block_type,
    )
    out = model(batch_X)
    assert out[0].shape[0] == model.out_dim
