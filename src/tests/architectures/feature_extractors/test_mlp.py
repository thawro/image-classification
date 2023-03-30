import pytest
from src.architectures.feature_extractors.mlp import MLP
import itertools
import torch


@pytest.mark.parametrize(
    "batch_X, hidden_dims, use_batch_norm, dropout, activation",
    itertools.product(
        ["mnist_data", "cifar10_data", "cifar100_data"],
        [[128], [256, 128], [256, 512, 256]],
        [True, False],
        [0, 0.5, 1.0],
        ["ReLU", "LeakyReLU", "Tanh"],
    ),
)
def test_forward(
    batch_X: torch.Tensor,
    hidden_dims: list[int],
    use_batch_norm: bool,
    dropout: float,
    activation: str,
    request,
):
    batch_X = request.getfixturevalue(batch_X)
    in_dim = batch_X[0].numel()
    mlp = MLP(
        in_dim=in_dim,
        hidden_dims=hidden_dims,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
        activation=activation,
    )
    out = mlp(batch_X)
    assert out[0].shape[0] == mlp.out_shape
