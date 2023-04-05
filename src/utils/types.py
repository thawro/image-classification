from torch.nn.common_types import _size_2_t
from typing import Literal, Sequence, Optional, Callable
from torch import Tensor
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from numpy.typing import NDArray
import numpy as np


_int_array = NDArray[np.int_]
_float_array = NDArray[np.float_]
_Image_Dataset = MNIST | CIFAR10 | CIFAR100
_size_2_t_list = _size_2_t | list[_size_2_t]
_stage = Literal["train", "val", "test"]