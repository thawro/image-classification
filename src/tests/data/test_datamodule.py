import torch

from src.data.datamodule import (
    CIFAR10DataModule,
    FashionMNISTDataModule,
    ImageDataModule,
    MNISTDataModule,
)
from src.tests.conftest import BATCH_SIZE


def _test_datamodule(datamodule: ImageDataModule, num_all_samples: int):
    assert not datamodule.train
    assert not datamodule.val
    assert not datamodule.test
    datamodule.download_data()
    datamodule.setup()
    assert datamodule.train
    assert datamodule.val
    assert datamodule.test

    assert datamodule.train_dataloader()
    assert datamodule.val_dataloader()
    assert datamodule.test_dataloader()

    train_samples = len(datamodule.train)
    valid_samples = len(datamodule.val)
    test_samples = len(datamodule.test)
    assert train_samples + valid_samples + test_samples == num_all_samples

    batch = next(iter(datamodule.train_dataloader()))
    x, y = batch
    assert len(x) == BATCH_SIZE
    assert len(y) == BATCH_SIZE
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


def test_cifar_datamodule(cifar_datamodule: CIFAR10DataModule):
    _test_datamodule(cifar_datamodule, 60_000)


def test_mnist_datamodule(mnist_datamodule: MNISTDataModule):
    _test_datamodule(mnist_datamodule, 70_000)


def test_fashion_mnist_datamodule(fashion_mnist_datamodule: FashionMNISTDataModule):
    _test_datamodule(fashion_mnist_datamodule, 70_000)
