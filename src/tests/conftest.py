import pytest
import torch


@pytest.fixture
def mnist_data() -> torch.Tensor:
    return torch.rand(64, 32, 32, 1)


@pytest.fixture
def cifar10_data() -> torch.Tensor:
    return torch.rand(64, 32, 32, 3)


@pytest.fixture
def cifar100_data() -> torch.Tensor:
    return torch.rand(64, 32, 32, 3)
