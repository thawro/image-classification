# from lightning.pytorch import LightningDataModule
import random
from abc import abstractmethod

import numpy as np
import torch
from PIL.Image import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import (
    CIFAR10,
    CIFAR100,
    EMNIST,
    MNIST,
    SVHN,
    CelebA,
    FashionMNIST,
)

from src.data.dataset import DynamicImageDataset, StaticImageDataset
from src.utils.types import Callable, Optional, TensorType, _img_transform, _stage
from src.utils.utils import ROOT

DATA_DIR = ROOT / "data"


class ImageDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = str(DATA_DIR),
        train_transform: _img_transform = None,
        inference_transform: _img_transform = None,
        batch_size: int = 64,
        seed: int = 42,
    ):
        super().__init__()
        self.train_transform = train_transform
        self.inference_transform = inference_transform
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seed = seed
        self.train, self.val, self.test = None, None, None

    def state_dict(self):
        return {
            "random_state": random.getstate(),
            "torch_random_state": torch.random.get_rng_state(),
            "numpy_random_state": np.random.get_state(),
        }

    def load_state_dict(self, state_dict):
        random.setstate(state_dict["random_state"])
        torch.random.set_rng_state(state_dict["torch_random_state"])
        np.random.set_state(state_dict["numpy_random_state"])

    @abstractmethod
    def download_data(self):
        pass

    @abstractmethod
    def load_dataset(self, split: str):
        pass

    @abstractmethod
    def set_train_val(self):
        pass

    def set_test(self):
        self.test = self.load_dataset(split="test")

    def setup(self, stage: Optional[_stage] = None):
        if stage == "fit" or stage is None:
            self.set_train_val()
        if stage == "test" or stage is None:
            self.set_test()

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def classes(self) -> list[str]:
        if self.train is not None:
            return self.train.classes
        return []

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("DataModule", "")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=16)


class StaticImageDataModule(ImageDataModule):
    mode: str = "train_split"

    @abstractmethod
    def load_dataset(self, split: str) -> StaticImageDataset:
        pass

    def set_train_val(self):
        if self.mode == "train_split":
            dataset = self.load_dataset(split="train")
            self.train, self.val = dataset.split_into_subsets(
                val_size=0.1,
                seed=self.seed,
                transforms=(self.train_transform, self.inference_transform),
            )
        else:
            self.train = self.load_dataset(split="train")
            self.val = self.load_dataset(split="val")


class DynamicImageDataModule(ImageDataModule):
    @abstractmethod
    def load_dataset(self, split: str) -> DynamicImageDataset:
        pass

    def set_train_val(self):
        self.train = self.load_dataset(split="train")
        self.val = self.load_dataset(split="val")


class MNISTDataModule(StaticImageDataModule):
    mode: str = "train_split"

    def load_dataset(self, split: str) -> StaticImageDataset:
        train = split == "train"
        transform = self.train_transform if train else self.inference_transform
        dataset = MNIST(root=self.data_dir, train=train, download=False, transform=transform)
        return StaticImageDataset.from_external(dataset)

    def download_data(self):
        MNIST(root=self.data_dir, train=True, download=True)
        MNIST(root=self.data_dir, train=False, download=True)


class CIFAR10DataModule(StaticImageDataModule):
    mode: str = "train_split"

    def load_dataset(self, split: str) -> StaticImageDataset:
        train = split == "train"
        transform = self.train_transform if train else self.inference_transform
        dataset = CIFAR10(root=self.data_dir, train=train, download=False, transform=transform)
        return StaticImageDataset.from_external(dataset)

    def download_data(self):
        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir, train=False, download=True)


class CIFAR100DataModule(StaticImageDataModule):
    mode: str = "train_split"

    def load_dataset(self, split: str) -> StaticImageDataset:
        train = split == "train"
        transform = self.train_transform if train else self.inference_transform
        dataset = CIFAR100(root=self.data_dir, train=train, download=False, transform=transform)
        return StaticImageDataset.from_external(dataset)

    def download_data(self):
        CIFAR100(root=self.data_dir, train=True, download=True)
        CIFAR100(root=self.data_dir, train=False, download=True)


class FashionMNISTDataModule(StaticImageDataModule):
    mode: str = "train_split"

    def load_dataset(self, split: str) -> StaticImageDataset:
        train = split == "train"
        transform = self.train_transform if train else self.inference_transform
        dataset = FashionMNIST(root=self.data_dir, train=train, download=False, transform=transform)
        return StaticImageDataset.from_external(dataset)

    def download_data(self):
        FashionMNIST(root=self.data_dir, train=True, download=True)
        FashionMNIST(root=self.data_dir, train=False, download=True)


class EMNISTDataModule(StaticImageDataModule):
    mode: str = "train_split"
    target_type = "byclass"

    def load_dataset(self, split: str) -> StaticImageDataset:
        train = split == "train"
        transform = self.train_transform if train else self.inference_transform
        dataset = EMNIST(
            root=self.data_dir,
            train=train,
            split=self.target_type,
            download=False,
            transform=transform,
        )
        return StaticImageDataset.from_external(dataset)

    def download_data(self):
        EMNIST(root=self.data_dir, train=True, split=self.target_type, download=True)
        EMNIST(root=self.data_dir, train=False, split=self.target_type, download=True)


class CelebADataModule(DynamicImageDataModule):
    target_type = "attr"

    def load_dataset(self, split: str) -> DynamicImageDataset:
        transform = self.train_transform if split == "train" else self.inference_transform
        if split == "val":
            split = "valid"
        dataset = CelebA(
            root=self.data_dir,
            split=split,
            target_type=self.target_type,
            download=False,
            transform=transform,
        )
        return DynamicImageDataset(dataset)

    def download_data(self):
        CelebA(root=self.data_dir, split="all", target_type=self.target_type, download=True)
