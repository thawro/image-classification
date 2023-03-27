# from lightning.pytorch import LightningDataModule
from pytorch_lightning import LightningDataModule

from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision import transforms as T
from torch.utils.data import DataLoader, random_split, Dataset
import torch
import matplotlib.pyplot as plt
from typing import Optional, Callable
from abc import abstractmethod


class ImageDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        transform: Optional[Callable] = None,
        batch_size: int = 64,
        seed: int = 42,
    ):
        super().__init__()
        self.transform = transform
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seed = seed

    @abstractmethod
    def data_loading_fn(self, *args, **kwargs) -> Dataset:
        pass

    def download_data(self):
        self.data_loading_fn(self.data_dir, train=True, download=True)
        self.data_loading_fn(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            data = self.data_loading_fn(self.data_dir, train=True, transform=self.transform)
            val_pcnt = 0.1
            n_val = int(len(data) * val_pcnt)
            n_train = len(data) - n_val
            generator = torch.Generator().manual_seed(self.seed)
            self.train, self.val = random_split(data, [n_train, n_val], generator=generator)
        if stage == "test" or stage is None:
            self.test = self.data_loading_fn(self.data_dir, train=False, transform=self.transform)

    @property
    def n_classes(self):
        return len(self.train.dataset.classes)

    def plot_images(self, split: str, n=10, transform: Optional[Callable] = None):
        dataset = getattr(self, split).dataset
        fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 4))
        for ax, (img, label) in zip(axes, dataset):
            if transform is not None:
                img = transform(img)
            img = img.permute(1, 2, 0)

            ax.imshow(img, cmap="gray")
            ax.set_title(f"{dataset.classes[label]}", fontsize=16)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=10 * self.batch_size, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=10 * self.batch_size, num_workers=16)


class MNISTDatamodule(ImageDataModule):
    def data_loading_fn(self, *args, **kwargs) -> Dataset:
        return MNIST(*args, **kwargs)


class CIFAR10Datamodule(ImageDataModule):
    def data_loading_fn(self, *args, **kwargs) -> Dataset:
        return CIFAR10(*args, **kwargs)


class CIFAR100Datamodule(ImageDataModule):
    def data_loading_fn(self, *args, **kwargs) -> Dataset:
        return CIFAR100(*args, **kwargs)
