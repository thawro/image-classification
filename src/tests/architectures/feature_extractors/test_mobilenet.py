import itertools

import pytest
import torch

from src.architectures.feature_extractors.mobilenet import MobilenetV2


@pytest.mark.parametrize(
    "batch_X, width_mul",
    itertools.product(
        ["inception_v4_data_batch"],
        [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],  # width_mul
    ),
)
def test_out_dim_mobilenet_v2(
    batch_X: torch.Tensor,
    width_mul: float,
    request,
):
    batch_X = request.getfixturevalue(batch_X)
    in_channels = batch_X[0].shape[0]
    model = MobilenetV2(in_channels=in_channels, width_mul=width_mul)
    out = model(batch_X)
    assert out[0].shape[0] == model.out_dim
