import torch
import torchvision.transforms as T
from src.utils.types import Tensor, Sequence

MEAN_MNIST = [0.1307]
STD_MNIST = [0.3081]

MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]


class Permute:
    def __init__(self, dims: tuple[int, ...] | list[int]):
        self.dims = list(dims)

    def __call__(self, sample: Tensor) -> Tensor:
        return torch.permute(sample, self.dims)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dims={self.dims})"


class UnNormalize(T.Normalize):
    def __init__(self, mean: Sequence[float], std: Sequence[float], *args, **kwargs):
        new_mean = [-m / s for m, s in zip(mean, std)]
        new_std = [1 / s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)


class ImgNormalize(T.Normalize):
    def __init__(self, n_channels: int):
        assert n_channels in [1, 3]
        if n_channels == 1:
            super().__init__(mean=MEAN_MNIST, std=STD_MNIST)
        super().__init__(mean=MEAN_IMAGENET, std=STD_IMAGENET)


class ImgUnNormalize(UnNormalize):
    def __init__(self, n_channels: int):
        assert n_channels in [1, 3]
        if n_channels == 1:
            super().__init__(mean=MEAN_MNIST, std=STD_MNIST)
        super().__init__(mean=MEAN_IMAGENET, std=STD_IMAGENET)
