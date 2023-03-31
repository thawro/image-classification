import pytest
import torch

RAW_BATCH_DATA = ["raw_mnist_batch", "raw_cifar10_batch", "raw_cifar100_batch"]
TRANSFORMED_BATCH_DATA = [
    "transformed_mnist_batch",
    "transformed_cifar10_batch",
    "transformed_cifar100_batch",
]


@pytest.fixture
def raw_mnist_batch() -> torch.Tensor:
    return torch.rand(64, 32, 32, 1)


@pytest.fixture
def raw_cifar10_batch() -> torch.Tensor:
    return torch.rand(64, 32, 32, 3)


@pytest.fixture
def raw_cifar100_batch() -> torch.Tensor:
    return torch.rand(64, 32, 32, 3)


@pytest.fixture
def transformed_mnist_batch() -> torch.Tensor:
    return torch.rand(64, 1, 32, 32)


@pytest.fixture
def transformed_cifar10_batch() -> torch.Tensor:
    return torch.rand(64, 3, 32, 32)


@pytest.fixture
def transformed_cifar100_batch() -> torch.Tensor:
    return torch.rand(64, 3, 32, 32)
