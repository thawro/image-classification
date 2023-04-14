from torch import nn
from torchtyping import TensorType


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, out_layer: nn.Module):
        super().__init__()
        self.net = nn.Linear(in_dim, num_classes)
        self.out_layer = out_layer

    def forward(self, x: TensorType["batch", "in_dim"]) -> TensorType["batch", "num_classes"]:
        out = self.net(x)
        return self.out_layer(out)


class MulticlassClassificationHead(ClassificationHead):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__(in_dim, num_classes, out_layer=nn.LogSoftmax(dim=1))


class MultilabelClassificationHead(ClassificationHead):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__(in_dim, num_classes, out_layer=nn.Sigmoid())
