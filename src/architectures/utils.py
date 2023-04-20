from collections import OrderedDict

from torch import nn

from src.utils.types import _named_modules


def make_named_sequential(named_layers: _named_modules):
    return nn.Sequential(OrderedDict(named_layers))
