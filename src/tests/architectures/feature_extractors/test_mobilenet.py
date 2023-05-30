import itertools

import pytest
import torch

from src.architectures.feature_extractors.mobilenet import MobileNet
from src.utils.types import Literal


@pytest.mark.parametrize(
    "batch_X, version, width_mul, load_from_torch, pretrained, freeze_extractor",
    itertools.product(
        ["mobilenet_data_batch"],
        ["v2", "v3_small", "v3_large"],  # version
        [0, 0.4, 0.8, 1.0],  # width_mul
        [True, False],  # load_from_torch
        [True, False],  # pretrained
        [True, False],  # freeze_extractor
    ),
)
def test_out_dim(
    batch_X: torch.Tensor,
    width_mul: float,
    version: Literal["v2", "v3_small", "v3_large"],
    load_from_torch: bool,
    pretrained: bool,
    freeze_extractor: bool,
    request,
):
    batch_X = request.getfixturevalue(batch_X)
    in_channels = batch_X[0].shape[0]
    model = MobileNet(
        in_channels=in_channels,
        width_mul=width_mul,
        version=version,
        load_from_torch=load_from_torch,
        pretrained=pretrained,
        freeze_extractor=freeze_extractor,
    )
    out = model(batch_X)
    assert out[0].shape[0] == model.out_dim
