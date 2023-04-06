import numpy as np
import pytest
import torch
from torchvision.transforms import Compose, Normalize, RandomRotation, ToTensor

from src.data.datamodule import (
    CelebADataModule,
    CIFAR10DataModule,
    EMNISTDataModule,
    FashionMNISTDataModule,
    MNISTDataModule,
    StaticImageDataset,
    SVHNDataModule,
)
from src.data.transforms import MEAN_IMAGENET, MEAN_MNIST, STD_IMAGENET, STD_MNIST


def get_transforms(mean, std):
    return [
        {"train": Compose([ToTensor()]), "inference": Compose([ToTensor()])},
        {"train": Compose([ToTensor(), RandomRotation(15)]), "inference": Compose([ToTensor()])},
        {
            "train": Compose([ToTensor(), Normalize(mean, std)]),
            "inference": Compose([ToTensor(), Normalize(mean, std)]),
        },
        {
            "train": Compose([ToTensor(), RandomRotation(15), Normalize(mean, std)]),
            "inference": Compose([ToTensor(), Normalize(mean, std)]),
        },
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
def cifar_dataset(request: pytest.FixtureRequest) -> StaticImageDataset:
    _size = (BATCH_SIZE, 32, 32, 3)
    classes = [str(i) for i in range(10)]
    data = np.random.randint(0, 256, _size)
    targets = np.random.randint(0, len(classes), (BATCH_SIZE,))
    transforms = request.param
    return StaticImageDataset(data, targets, classes, transforms["train"])


@pytest.fixture(params=IMAGENET_TRANSFORMS)
def cifar_datamodule(request: pytest.FixtureRequest) -> CIFAR10DataModule:
    transforms = request.param
    datamodule = CIFAR10DataModule(
        train_transform=transforms["train"],
        inference_transform=transforms["inference"],
        batch_size=BATCH_SIZE,
    )
    return datamodule


@pytest.fixture(params=MNIST_TRANSFORMS)
def mnist_dataset(request: pytest.FixtureRequest) -> StaticImageDataset:
    _size = (BATCH_SIZE, 28, 28, 1)
    classes = [str(i) for i in range(10)]
    data = torch.randint(0, 256, _size)
    targets = torch.randint(0, len(classes), (BATCH_SIZE,))
    transforms = request.param
    return StaticImageDataset(data, targets, classes, transforms["train"])


@pytest.fixture(params=MNIST_TRANSFORMS)
def mnist_datamodule(request: pytest.FixtureRequest) -> MNISTDataModule:
    transforms = request.param
    datamodule = MNISTDataModule(
        train_transform=transforms["train"],
        inference_transform=transforms["inference"],
        batch_size=BATCH_SIZE,
    )
    return datamodule


@pytest.fixture(params=MNIST_TRANSFORMS)
def fashion_mnist_datamodule(request: pytest.FixtureRequest) -> FashionMNISTDataModule:
    transforms = request.param
    datamodule = FashionMNISTDataModule(
        train_transform=transforms["train"],
        inference_transform=transforms["inference"],
        batch_size=BATCH_SIZE,
    )
    return datamodule


@pytest.fixture(params=MNIST_TRANSFORMS)
def emnist_datamodule(request: pytest.FixtureRequest) -> EMNISTDataModule:
    transforms = request.param
    datamodule = EMNISTDataModule(
        train_transform=transforms["train"],
        inference_transform=transforms["inference"],
        batch_size=BATCH_SIZE,
    )
    return datamodule


@pytest.fixture(params=IMAGENET_TRANSFORMS)
def celeb_a_datamodule(request: pytest.FixtureRequest) -> CelebADataModule:
    transforms = request.param
    datamodule = CelebADataModule(
        train_transform=transforms["train"],
        inference_transform=transforms["inference"],
        batch_size=BATCH_SIZE,
    )
    return datamodule


@pytest.fixture(params=IMAGENET_TRANSFORMS)
def svhn_datamodule(request: pytest.FixtureRequest) -> SVHNDataModule:
    transforms = request.param
    datamodule = SVHNDataModule(
        train_transform=transforms["train"],
        inference_transform=transforms["inference"],
        batch_size=BATCH_SIZE,
    )
    return datamodule


@pytest.fixture
def example_experiment_name():
    return "test"
