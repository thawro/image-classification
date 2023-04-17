from abc import abstractmethod

from torch import nn
from torchtyping import TensorType

from src.utils.types import Any


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Identity()

    def forward(self, x: TensorType["batch", "channels", "height", "width"]) -> TensorType["batch", "out_dim"]:
        return self.net(x)

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
