from torch import nn
from torchtyping import TensorType


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Linear(in_dim, n_classes)

    def forward(self, x: TensorType["batch", "in_dim"]) -> TensorType["batch", "n_classes"]:
        out = self.net(x)
        return out
