from abc import abstractmethod

from torch import nn
from torchtyping import TensorType

from src.utils.types import _any_dict


class FeatureExtractor(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: TensorType["batch", "channels", "height", "width"]) -> TensorType["batch", "out_dim"]:
        out = x
        for name, module in self.net.named_children():
            out = module(out)
            # print(name, out.shape)
        return out

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    @property
    def name(self):
        return self.__class__.__name__

    @property
    @abstractmethod
    def params(self) -> _any_dict:
        raise NotImplementedError()

    @property
    @abstractmethod
    def out_dim(self) -> int:
        raise NotImplementedError()


class ExternalFeatureExtractor(FeatureExtractor):
    def __init__(self, net: nn.Module, out_channels: int, params: _any_dict = {}):
        super().__init__(net)
        self._params = params
        self.out_channels = out_channels

    @property
    def params(self) -> _any_dict:
        return self._params

    @property
    def out_dim(self) -> int:
        return self.out_channels
