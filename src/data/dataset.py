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
    _float_array,
    _img_transform,
    _int_array,
    _StaticImageDataset,
)
from src.utils.utils import log


class StaticImageDataset(Dataset):
    def __init__(
        self,
        data: TensorType["batch", "height", "width", "channels"]
        | TensorType["batch", "height", "width"]
        | _float_array,
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

    def __getitem__(self, idx: int) -> tuple[TensorType["channels", "height", "width"], Tensor]:
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
        img_transform: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 4))
        for (
            idx,
            ax,
        ) in enumerate(axes):
            img, label = self[idx]
            if img_transform is not None:
                img = img_transform(img)
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
        data = dataset.data
        ds_name = dataset.__class__.__name__
        shape = np.array(data.shape)
        if len(shape) > 3:  # omit MNIST case
            n_channels = 3 if 3 in shape else 1  # RGB or GREY
            shape_idx = np.where(shape == n_channels)[0][0]

            if shape_idx == 1:
                log.info("Transposing data from [N, C, H, W] to [N, H, W, C] shape")
                data = data.transpose(0, 2, 3, 1)  # N, C, H, W -> N, H, W, C
        try:
            targets = dataset.targets
        except AttributeError as e:
            log.error(e)
            log.info(f"Dataset {ds_name} has no 'targets' attr. Using 'labels' attr instead.")
            targets = dataset.labels

        try:
            classes = dataset.classes
        except AttributeError as e:
            log.error(e)
            log.info(f"Dataset {ds_name} has no 'classes' attr. Using unique targets instead.")
            classes = [str(class_) for class_ in np.unique(targets)]

        return StaticImageDataset(data, targets, classes, dataset.transform)


class DynamicImageDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        targets_attr: Optional[str] = None,
        classes_attr: Optional[str] = None,
    ):
        self.dataset = dataset
        if targets_attr is not None:
            self.targets = getattr(dataset, targets_attr)
        if classes_attr is not None:
            classes = getattr(dataset, classes_attr)
            self.classes = list(filter(None, classes))  # filter empty strings

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
