from torch import nn
from abc import abstractmethod
from torchtyping import TensorType


class FeatureExtractor(nn.Module):
    name: str = ""

    def __init__(self):
        super().__init__()
        self.net = nn.Identity()

    def forward(
        self, x: TensorType["batch", "channels", "height", "width"]
    ) -> TensorType["batch", "out_dim"]:
        return self.net(x)

    @property
    @abstractmethod
    def out_dim(self) -> int:
        pass
