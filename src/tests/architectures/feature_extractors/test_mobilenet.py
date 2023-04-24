import itertools

import pytest
import torch

from src.architectures.feature_extractors.mobilenet import MobileNet
from src.utils.types import Literal


@pytest.mark.parametrize(
    "batch_X, version, width_mul",
    itertools.product(
        ["mobilenet_data_batch"],
        ["v2", "v3_small", "v3_large"],  # version
        [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],  # width_mul
    ),
)
def test_out_dim(
    batch_X: torch.Tensor,
    width_mul: float,
    version: Literal["v2", "v3_small", "v3_large"],
    request,
):
    batch_X = request.getfixturevalue(batch_X)
    in_channels = batch_X[0].shape[0]
    model = MobileNet(in_channels=in_channels, width_mul=width_mul, version=version)
    out = model(batch_X)
    assert out[0].shape[0] == model.out_dim
