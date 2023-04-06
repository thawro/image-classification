# from lightning.pytorch import LightningDataModule
import random
from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchtyping import TensorType
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST

from src.utils.types import Callable, Optional, _Image_Dataset, _int_array, _stage


class ImageDataset(Dataset):
    def __init__(
        self,
        data: TensorType["batch", "height", "width", "channels"] | TensorType["batch", "height", "width"] | _int_array,
        targets: TensorType["batch"] | _int_array,
        classes: list[str],
        transform: Optional[
            Callable[
                [Image.Image],
                TensorType["channels", "height", "width"],
            ]
        ] = None,
    ):
        if isinstance(data, torch.Tensor):
            data = data.numpy().squeeze()
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)
        elif isinstance(targets, list):
            targets = torch.Tensor(targets)

        data = data.astype(np.uint8)
        targets = targets.to(torch.int64)
        self.data = data
        self.targets = targets
        self.classes = classes
        self.transform = transform if transform is not None else T.PILToTensor()

    def __getitem__(self, idx: int) -> tuple[TensorType["channels", "height", "width"], torch.Tensor]:
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def dummy_input(self):
        return self[0][0]

    @property
    def dummy_input_shape(self):
        return self.dummy_input.shape


class ImageDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        train_transform: Optional[
            Callable[
                [Image.Image],
                TensorType["channels", "height", "width"],
            ]
        ] = None,
        inference_transform: Optional[
            Callable[
                [Image.Image],
                TensorType["channels", "height", "width"],
            ]
        ] = None,
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
    def data_loading_fn(self, *args, **kwargs) -> _Image_Dataset:
        pass

    def download_data(self):
        self.data_loading_fn(self.data_dir, train=True, download=True)
        self.data_loading_fn(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[_stage] = None):
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
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def classes(self) -> list[str]:
        return self.train.classes

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("DataModule", "")

    def plot_images(
        self,
        split: _stage,
        n: int = 10,
        transform: Optional[
            Callable[
                [TensorType["channels", "height", "width"]],
                TensorType["channels", "height", "width"],
            ]
        ] = None,
    ):
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
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=16)


class MNISTDataModule(ImageDataModule):
    def data_loading_fn(self, *args, **kwargs) -> MNIST:
        return MNIST(*args, **kwargs)


class CIFAR10DataModule(ImageDataModule):
    def data_loading_fn(self, *args, **kwargs) -> CIFAR10:
        return CIFAR10(*args, **kwargs)


class CIFAR100DataModule(ImageDataModule):
    def data_loading_fn(self, *args, **kwargs) -> CIFAR100:
        return CIFAR100(*args, **kwargs)


class FashionMNISTDataModule(ImageDataModule):
    def data_loading_fn(self, *args, **kwargs) -> FashionMNIST:
        return FashionMNIST(*args, **kwargs)
