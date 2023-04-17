from typing import Any, Callable, Literal, Optional, Protocol, Self, Sequence, TypedDict

import matplotlib.figure
import numpy as np
import plotly.graph_objects
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor
from torch.nn.common_types import _size_2_t
from torchtyping import TensorType
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, MNIST, CelebA, FashionMNIST

_int_array = NDArray[np.int_]
_float_array = NDArray[np.float_]
_int_list = list[int]
_float_list = list[float]
_size_2_t_list = _size_2_t | list[_size_2_t]
_stage = Literal["train", "val", "test"]
_img_transform = Optional[Callable[[Image.Image], Tensor]]
_metrics_average = Literal["micro", "macro", "weighted", "none"]
_task = Literal["binary", "multiclass", "multilabel"]


class _StaticImageDataset(Protocol):
    data: Tensor | _float_array
    targets: Tensor | _int_array
    classes: list[str]
    transform: Optional[Callable[[Image.Image], Tensor]]


Outputs = dict[Literal["loss", "probs", "preds", "targets"], Tensor]
_Figure = plotly.graph_objects.Figure | matplotlib.figure.Figure
HeadType = Literal["linear", "convolutional"]
