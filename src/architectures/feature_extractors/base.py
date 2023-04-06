from abc import abstractmethod

from torch import nn
from torchtyping import TensorType


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
    def out_dim(self) -> int:
        pass
