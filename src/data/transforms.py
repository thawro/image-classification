import torch
import torchvision.transforms as T


class Permute:
    def __init__(self, dims: list[int]):
        self.dims = list(dims)

    def __call__(self, sample: torch.Tensor):
        return torch.permute(sample, self.dims)


class ImgNormalize(T.Normalize):
    def __init__(self, n_channels: int):
        assert n_channels in [1, 3]
        if n_channels == 1:
            mean = [0.1307]
            std = [0.3081]
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        super().__init__(mean=mean, std=std)


class UnNormalize(T.Normalize):
    def __init__(self, mean, std, *args, **kwargs):
        new_mean = [-m / s for m, s in zip(mean, std)]
        new_std = [1 / s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)
