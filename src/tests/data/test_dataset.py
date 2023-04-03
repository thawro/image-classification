import torch
from src.data.datamodule import ImageDataset
from src.tests.conftest import BATCH_SIZE


def test_mnist_len(mnist_dataset: ImageDataset):
    assert len(mnist_dataset) == BATCH_SIZE


def test_cifar_len(cifar_dataset: ImageDataset):
    assert len(cifar_dataset) == BATCH_SIZE


def _test_dataset_getitem(dataset: ImageDataset):
    data, target = dataset[0]
    assert torch.is_tensor(data)
    assert torch.is_floating_point(data)
    assert len(data.shape) == 3
    assert torch.is_tensor(target)
    assert target.dtype == torch.int16
    assert target <= len(dataset.classes)
    return data, target


def test_mnist_getitem(mnist_dataset: ImageDataset):
    data, target = _test_dataset_getitem(mnist_dataset)
    assert data.shape[0] == 1


def test_cifar_getitem(cifar_dataset: ImageDataset):
    data, target = _test_dataset_getitem(cifar_dataset)
    assert data.shape[0] == 3
