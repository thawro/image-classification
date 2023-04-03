# from lightning.pytorch import LightningDataModule
from pytorch_lightning import LightningDataModule
from PIL import Image
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torch.utils.data import DataLoader, Dataset
import torch
import matplotlib.pyplot as plt
from abc import abstractmethod
from sklearn.model_selection import train_test_split
import numpy as np
from torchtyping import TensorType
from src.utils.types import _int_array, _Image_Dataset, _stage, Optional, Callable
from src.data.transforms import BasicImageTransform
import torchvision.transforms as T


class ImageDataset(Dataset):
    def __init__(
        self,
        data: TensorType["batch", "height", "width", "channels"]
        | TensorType["batch", "height", "width"]
        | _int_array,
        targets: TensorType["batch"] | _int_array,
        classes: list[str],
        transform: Optional[
            Callable[
                [Image.Image],
                TensorType["channels", "height", "width"],
            ]
        ] = BasicImageTransform(),
    ):
        if isinstance(data, torch.Tensor):
            data = data.numpy().squeeze()
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)

        data = data.astype(np.uint8)
        targets = targets.to(torch.int16)
        self.data = data
        self.targets = targets
        self.classes = classes
        self.transform = transform if transform is not None else T.PILToTensor()

    def __getitem__(
        self, idx: int
    ) -> tuple[TensorType["channels", "height", "width"], torch.Tensor]:
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)


class ImageDataModule(LightningDataModule):
    name: str = ""

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
            train_idxs, val_idxs = train_test_split(
                range(len(data)), test_size=0.1, random_state=self.seed
            )
            train_data, train_targets = data[train_idxs], np.array(targets)[train_idxs]
            val_data, val_targets = data[val_idxs], np.array(targets)[val_idxs]
            self.train = ImageDataset(
                train_data, train_targets, dataset.classes, self.train_transform
            )
            self.val = ImageDataset(
                val_data, val_targets, dataset.classes, self.inference_transform
            )
        if stage == "test" or stage is None:
            dataset = self.data_loading_fn(
                self.data_dir, train=False, transform=self.inference_transform
            )
            self.test = ImageDataset(
                dataset.data, dataset.targets, dataset.classes, self.inference_transform
            )

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def classes(self) -> list[str]:
        return self.train.classes

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


class MNISTDatamodule(ImageDataModule):
    name: str = "MNIST"

    def data_loading_fn(self, *args, **kwargs) -> MNIST:
        return MNIST(*args, **kwargs)


class CIFAR10Datamodule(ImageDataModule):
    name: str = "CIFAR10"

    def data_loading_fn(self, *args, **kwargs) -> CIFAR10:
        return CIFAR10(*args, **kwargs)


class CIFAR100Datamodule(ImageDataModule):
    name: str = "CIFAR100"

    def data_loading_fn(self, *args, **kwargs) -> CIFAR100:
        return CIFAR100(*args, **kwargs)
