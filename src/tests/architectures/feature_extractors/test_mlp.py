import pytest
from src.architectures.feature_extractors.mlp import MLP
import itertools
import torch
from src.tests.conftest import RAW_BATCH_DATA, TRANSFORMED_BATCH_DATA

DATA = RAW_BATCH_DATA + TRANSFORMED_BATCH_DATA


@pytest.mark.parametrize(
    "batch_X, hidden_dims, use_batch_norm, dropout, activation",
    itertools.product(
        DATA,
        [[128], [256, 128], [256, 512, 256]],
        [True, False],
        [0, 0.5, 1.0],
        ["ReLU", "LeakyReLU", "Tanh"],
    ),
)
def test_out_dim(
    batch_X: str,
    hidden_dims: list[int],
    use_batch_norm: bool,
    dropout: float,
    activation: str,
    request: pytest.FixtureRequest,
):
    batch_X: torch.Tensor = request.getfixturevalue(batch_X)
    in_dim = batch_X[0].numel()
    model = MLP(
        in_dim=in_dim,
        hidden_dims=hidden_dims,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
        activation=activation,
    )
    out = model(batch_X)
    assert out[0].shape[0] == model.out_dim
