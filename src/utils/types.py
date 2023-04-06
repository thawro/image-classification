from typing import Callable, Literal, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from torch import Tensor
from torch.nn.common_types import _size_2_t
from torchvision.datasets import CIFAR10, CIFAR100, MNIST

_int_array = NDArray[np.int_]
_float_array = NDArray[np.float_]
_int_list = list[int]
_float_list = list[float]
_Image_Dataset = MNIST | CIFAR10 | CIFAR100
_size_2_t_list = _size_2_t | list[_size_2_t]
_stage = Literal["train", "val", "test"]
