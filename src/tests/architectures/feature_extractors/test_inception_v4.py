import itertools

import pytest
import torch

from src.architectures.feature_extractors.inception_v4 import InceptionResNetV2


@pytest.mark.parametrize(
    "batch_X, scale_a, scale_b, scale_c",
    itertools.product(
        ["inception_v4_data_batch"],
        [0, 0.5, 1.0],  # scale_a
        [0, 0.5, 1.0],  # scale_b
        [0, 0.5, 1.0],  # scale_c
    ),
)
def test_out_dim(
    batch_X: torch.Tensor,
    scale_a: int,
    scale_b: int,
    scale_c: float,
    request,
):
    batch_X = request.getfixturevalue(batch_X)
    in_channels = batch_X[0].shape[0]
    scales = [scale_a, scale_b, scale_c]
    model = InceptionResNetV2(
        in_channels=in_channels,
        scales=scales,
    )
    out = model(batch_X)
    assert out[0].shape[0] == model.out_dim
