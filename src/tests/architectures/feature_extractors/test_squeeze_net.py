import itertools

import pytest
import torch

from src.architectures.feature_extractors.squeeze_net import SqueezeNet


@pytest.mark.parametrize(
    "batch_X, base_e, incr_e, pct_3x3, freq, SR, simple_bypass",
    itertools.product(
        ["squeeze_net_data_batch"],
        [16, 128],  # base_e
        [16, 128],  # incr_e
        [0.2, 0.5, 0.8],  # pct_3x3
        [1, 2, 3],  # freq
        [0.125, 0.5],  # SR
        [True, False],  # simple_bypass
    ),
)
def test_out_dim(
    batch_X: torch.Tensor,
    base_e: int,
    incr_e: int,
    pct_3x3: float,
    freq: int,
    SR: float,
    simple_bypass: bool,
    request,
):
    batch_X = request.getfixturevalue(batch_X)
    in_channels = batch_X[0].shape[0]
    model = SqueezeNet(
        in_channels=in_channels,
        base_e=base_e,
        incr_e=incr_e,
        pct_3x3=pct_3x3,
        freq=freq,
        SR=SR,
        simple_bypass=simple_bypass,
    )
    out = model(batch_X)
    assert out[0].shape[0] == model.out_dim
