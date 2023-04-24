from torch import nn
from torchtyping import TensorType


class ClassificationHead(nn.Module):
    def __init__(self, net: nn.Module, out_layer: nn.Module):
        super().__init__()
        self.net = net
        self.out_layer = out_layer

    def forward(self, x: TensorType["batch", "in_dim"]) -> TensorType["batch", "num_classes"]:
        out = self.net(x)
        return self.out_layer(out)


class LinearClassificationHead(ClassificationHead):
    def __init__(self, in_dim: int, num_classes: int, out_layer: nn.Module, dropout: float = 0):
        net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )
        super().__init__(net=net, out_layer=out_layer)


class ConvolutionalClassificationHead(ClassificationHead):
    def __init__(self, in_dim: int, num_classes: int, out_layer: nn.Module):
        net = nn.Sequential(
            nn.Conv2d(in_dim, num_classes, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        super().__init__(net=net, out_layer=out_layer)


class MulticlassLinearClassificationHead(LinearClassificationHead):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0):
        super().__init__(in_dim, num_classes, out_layer=nn.LogSoftmax(dim=1), dropout=dropout)


class MultilabelLinearClassificationHead(LinearClassificationHead):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0):
        super().__init__(in_dim, num_classes, out_layer=nn.Sigmoid(), dropout=dropout)


class MulticlassConvolutionalClassificationHead(ConvolutionalClassificationHead):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__(in_dim, num_classes, out_layer=nn.LogSoftmax(dim=1))


class MultilabelConvolutionalClassificationHead(ConvolutionalClassificationHead):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__(in_dim, num_classes, out_layer=nn.Sigmoid())
