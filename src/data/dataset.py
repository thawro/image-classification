import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.utils.types import (
    Callable,
    Optional,
    Self,
    Tensor,
    TensorType,
    _img_transform,
    _int_array,
    _StaticImageDataset,
)


class StaticImageDataset(Dataset):
    def __init__(
        self,
        data: TensorType["batch", "height", "width", "channels"] | TensorType["batch", "height", "width"] | _int_array,
        targets: TensorType["batch"] | _int_array,
        classes: list[str],
        transform: _img_transform = None,
    ):
        if isinstance(data, Tensor):
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

    def split_into_subsets(self, val_size: float, transforms: tuple[Callable, Callable], seed=42) -> tuple[Self, Self]:
        idxs_0, idxs_1 = train_test_split(range(len(self)), test_size=val_size, random_state=seed)
        data_0, targets_0 = self.data[idxs_0], np.array(self.targets)[idxs_0]
        data_1, targets_1 = self.data[idxs_1], np.array(self.targets)[idxs_1]
        dataset_0 = StaticImageDataset(data_0, targets_0, self.classes, transforms[0])
        dataset_1 = StaticImageDataset(data_1, targets_1, self.classes, transforms[1])
        return dataset_0, dataset_1

    def plot_images(
        self,
        n: int = 10,
        transform: Optional[
            Callable[
                [TensorType["channels", "height", "width"]],
                TensorType["channels", "height", "width"],
            ]
        ] = None,
    ):
        fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 4))
        for (
            idx,
            ax,
        ) in enumerate(axes):
            img, label = self[idx]
            if transform is not None:
                img = transform(img)
            img = img.permute(1, 2, 0)
            ax.imshow(img, cmap="gray")
            ax.set_title(f"{self.classes[label]}", fontsize=16)
        return fig

    @property
    def dummy_input(self):
        return self[0][0]

    @property
    def dummy_input_shape(self):
        return self.dummy_input.shape

    @classmethod
    def from_external(cls, dataset: _StaticImageDataset) -> Self:
        return StaticImageDataset(dataset.data, dataset.targets, dataset.classes, dataset.transform)


class DynamicImageDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __getitem__(self, idx: int) -> tuple[TensorType["channels", "height", "width"], torch.Tensor]:
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    @property
    def dummy_input(self):
        return self.dataset[0][0]

    @property
    def dummy_input_shape(self):
        return self.dummy_input.shape
