import torch

from src.data.datamodule import (
    CelebADataModule,
    CIFAR10DataModule,
    DynamicImageDataModule,
    EMNISTDataModule,
    FashionMNISTDataModule,
    ImageDataModule,
    MNISTDataModule,
    StaticImageDataModule,
    SVHNDataModule,
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


def _test_static_datamodule(datamodule: StaticImageDataModule, num_all_samples: int):
    _test_datamodule(datamodule, num_all_samples)


def _test_dynamic_datamodule(datamodule: DynamicImageDataModule, num_all_samples: int):
    _test_datamodule(datamodule, num_all_samples)


def test_cifar(cifar_datamodule: CIFAR10DataModule):
    _test_static_datamodule(cifar_datamodule, 60_000)


def test_mnist(mnist_datamodule: MNISTDataModule):
    _test_static_datamodule(mnist_datamodule, 70_000)


def test_fashion_mnist(fashion_mnist_datamodule: FashionMNISTDataModule):
    _test_static_datamodule(fashion_mnist_datamodule, 70_000)


def test_emnist(emnist_datamodule: EMNISTDataModule):
    _test_static_datamodule(emnist_datamodule, 814_255)


def test_celeb_a(celeb_a_datamodule: CelebADataModule):
    _test_dynamic_datamodule(celeb_a_datamodule, 202_599)


def test_svhn(svhn_datamodule: SVHNDataModule):
    _test_static_datamodule(svhn_datamodule, 99_289)
