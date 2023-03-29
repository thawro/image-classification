from torch import nn
from abc import abstractmethod


class FeatureExtractor(nn.Module):
    name: str = ""

    def __init__(self):
        super().__init__()
        self.net = nn.Identity()

    def forward(self, x):
        return self.net(x)

    @property
    @abstractmethod
    def out_shape(self):
        pass
