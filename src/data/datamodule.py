# from lightning.pytorch import LightningDataModule
from pytorch_lightning import LightningDataModule

from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision import transforms as T
from torch.utils.data import DataLoader, random_split, Dataset
import torch
import matplotlib.pyplot as plt
from typing import Optional, Callable
from abc import abstractmethod
from sklearn.model_selection import train_test_split
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, data, targets, classes, transform):
        data = data if isinstance(data, torch.Tensor) else torch.from_numpy(data)
        if len(data.shape) == 3:  # MNSIT: no channels dimensionality
            data = data.unsqueeze(-1)
        self.data = data.float() / 255
        self.targets = targets
        self.classes = classes
        self.transform = transform

    def __getitem__(self, idx):
        data = self.data[idx]
        targets = self.targets[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data, targets

    def __len__(self):
        return len(self.data)


class ImageDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        train_transform: Optional[Callable] = None,
        inference_transform: Optional[Callable] = None,
        batch_size: int = 64,
        seed: int = 42,
    ):
        super().__init__()
        self.train_transform = train_transform
        self.inference_transform = inference_transform
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
            dataset = self.data_loading_fn(self.data_dir, train=True, transform=None)
            data, targets = dataset.data, dataset.targets
            train_idxs, val_idxs = train_test_split(range(len(data)), test_size=0.1, random_state=self.seed)
            train_data, train_targets = data[train_idxs], np.array(targets)[train_idxs]
            val_data, val_targets = data[val_idxs], np.array(targets)[val_idxs]
            self.train = ImageDataset(train_data, train_targets, dataset.classes, self.train_transform)
            self.val = ImageDataset(val_data, val_targets, dataset.classes, self.inference_transform)
        if stage == "test" or stage is None:
            dataset = self.data_loading_fn(self.data_dir, train=False, transform=self.inference_transform)
            self.test = ImageDataset(dataset.data, dataset.targets, dataset.classes, self.inference_transform)

    @property
    def n_classes(self):
        return len(self.train.classes)

    def plot_images(self, split: str, n=10, transform: Optional[Callable] = None):
        dataset = getattr(self, split)
        fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 4))
        for (
            idx,
            ax,
        ) in enumerate(axes):
            img, label = dataset[idx]
            if transform is not None:
                img = transform(img)
            img = img.permute(1, 2, 0)
            ax.imshow(img, cmap="gray")
            ax.set_title(f"{dataset.classes[label]}", fontsize=16)
        return fig

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
