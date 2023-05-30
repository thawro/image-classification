import itertools

import pytest
import torch

from src.architectures.feature_extractors.vgg import VGG
from src.utils.types import Literal


@pytest.mark.parametrize(
    "batch_X, version, use_batch_norm, use_global_pool, load_from_torch, freeze_extractor",
    itertools.product(
        ["vgg_data_batch"],
        ["vgg11", "vgg13", "vgg16", "vgg19"],  # version
        [True, False],  # use_batch_norm
        [True, False],  # use_global_pool
        [True, False],  # load_from_torch
        [True, False],  # freeze_extractor
    ),
)
def test_out_dim(
    batch_X: torch.Tensor,
    version: Literal["vgg11", "vgg13", "vgg16", "vgg19"],
    use_batch_norm: bool,
    use_global_pool: bool,
    load_from_torch: bool,
    freeze_extractor: bool,
    request,
):
    batch_X = request.getfixturevalue(batch_X)
    in_channels = batch_X[0].shape[0]
    model = VGG(
        in_channels=in_channels,
        version=version,
        use_batch_norm=use_batch_norm,
        use_global_pool=use_global_pool,
        load_from_torch=load_from_torch,
        freeze_extractor=freeze_extractor,
    )
    out = model(batch_X)
    assert out[0].shape[0] == model.out_dim
