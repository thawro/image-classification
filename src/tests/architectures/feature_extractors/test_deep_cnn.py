import pytest
from src.architectures.feature_extractors.cnn import DeepCNN
from src.utils.types import _size_2_t_list
import itertools
import torch
from src.tests.conftest import TRANSFORMED_BATCH_DATA
from typing import Literal


@pytest.mark.parametrize(
    "batch_X, out_channels, kernels, pool_kernels, pool_type, use_batch_norm, dropout, activation",
    itertools.product(
        TRANSFORMED_BATCH_DATA,
        [[16], [64, 128]],
        [1, 3],
        [1, 2],
        ["Max", "Avg"],
        [True, False],
        [0, 0.5, 1.0],
        ["ReLU", "LeakyReLU", "Tanh"],
    ),
)
def test_out_dim(
    batch_X: torch.Tensor,
    out_channels: list[int],
    kernels: _size_2_t_list,
    pool_kernels: _size_2_t_list,
    pool_type: Literal["Max", "Avg"],
    use_batch_norm: bool,
    dropout: float,
    activation: str,
    request,
):
    batch_X = request.getfixturevalue(batch_X)
    in_channels = batch_X[0].shape[0]
    model = DeepCNN(
        in_channels=in_channels,
        out_channels=out_channels,
        kernels=kernels,
        pool_kernels=pool_kernels,
        pool_type=pool_type,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
        activation=activation,
    )
    out = model(batch_X)
    assert out[0].shape[0] == model.out_dim
