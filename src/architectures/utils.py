from collections import OrderedDict

from torch import nn

from src.utils.types import _named_modules, _size_2_t


def make_named_sequential(named_layers: _named_modules):
    return nn.Sequential(OrderedDict(named_layers))


def get_padding(kernel_size: _size_2_t, stride: _size_2_t = 1, dilation: int = 1) -> tuple[int, int]:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    _get_pad = lambda stride, dilation, kernel: ((stride - 1) + dilation * (kernel - 1)) // 2
    return tuple(_get_pad(s, dilation, k) for s, k in zip(stride, kernel_size))
