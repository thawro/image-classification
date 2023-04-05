import pytest
import torch
from src.data.datamodule import ImageDataset
from torchvision.transforms import ToTensor, Normalize, RandomRotation, Compose
from src.data.transforms import MEAN_IMAGENET, STD_IMAGENET, MEAN_MNIST, STD_MNIST
import numpy as np


def get_transforms(mean, std):
    return [
        Compose([ToTensor()]),
        Compose([ToTensor(), RandomRotation(15)]),
        Compose([ToTensor(), Normalize(mean, std)]),
        Compose([ToTensor(), RandomRotation(15), Normalize(mean, std)]),
    ]


IMAGENET_TRANSFORMS = get_transforms(MEAN_IMAGENET, STD_IMAGENET)
MNIST_TRANSFORMS = get_transforms(MEAN_MNIST, STD_MNIST)

DATASETS = ["mnist_dataset", "cifar_dataset"]
RAW_BATCH_DATA = ["raw_mnist_batch", "raw_cifar10_batch", "raw_cifar100_batch"]
TRANSFORMED_BATCH_DATA = [
    "transformed_mnist_batch",
    "transformed_cifar10_batch",
    "transformed_cifar100_batch",
]

BATCH_SIZE = 64


@pytest.fixture
def raw_mnist_batch() -> torch.Tensor:
    return torch.rand(BATCH_SIZE, 32, 32, 1)


@pytest.fixture
def raw_cifar10_batch() -> torch.Tensor:
    return torch.rand(BATCH_SIZE, 32, 32, 3)


@pytest.fixture
def raw_cifar100_batch() -> torch.Tensor:
    return torch.rand(BATCH_SIZE, 32, 32, 3)


@pytest.fixture
def transformed_mnist_batch() -> torch.Tensor:
    return torch.rand(BATCH_SIZE, 1, 32, 32)


@pytest.fixture
def transformed_cifar10_batch() -> torch.Tensor:
    return torch.rand(BATCH_SIZE, 3, 32, 32)


@pytest.fixture
def transformed_cifar100_batch() -> torch.Tensor:
    return torch.rand(BATCH_SIZE, 3, 32, 32)


@pytest.fixture(params=IMAGENET_TRANSFORMS)
def cifar_dataset(request: pytest.FixtureRequest):
    _size = (BATCH_SIZE, 32, 32, 3)
    classes = [str(i) for i in range(10)]
    data = np.random.randint(0, 256, _size)
    targets = np.random.randint(0, len(classes), (BATCH_SIZE,))
    transform = request.param
    return ImageDataset(data, targets, classes, transform)


@pytest.fixture(params=MNIST_TRANSFORMS)
def mnist_dataset(request: pytest.FixtureRequest):
    _size = (BATCH_SIZE, 28, 28, 1)
    classes = [str(i) for i in range(10)]
    data = torch.randint(0, 256, _size)
    targets = torch.randint(0, len(classes), (BATCH_SIZE,))
    transform = request.param
    return ImageDataset(data, targets, classes, transform)


@pytest.fixture
def example_experiment_name():
    return "test"
