from abc import abstractmethod

from torch import nn
from torchtyping import TensorType

from src.utils.types import Any


class FeatureExtractor(nn.Module):
    def __init__(self, net: nn.Sequential):
        super().__init__()
        self.net = net

    def forward(self, x: TensorType["batch", "channels", "height", "width"]) -> TensorType["batch", "out_dim"]:
        out = x
        for name, module in self.net.named_children():
            out = module(out)
            # print(name, out.shape)
        return out

    @property
    def name(self):
        return self.__class__.__name__

    @property
    @abstractmethod
    def params(self) -> dict[str, Any]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def out_dim(self) -> int:
        raise NotImplementedError()
