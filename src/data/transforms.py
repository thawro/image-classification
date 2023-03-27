import torch
from torchvision.transforms import Normalize


class Permute:
    def __init__(self, dims: list[int]):
        self.dims = list(dims)

    def __call__(self, sample: torch.Tensor):
        return torch.permute(sample, self.dims)


class ImgNormalize(Normalize):
    def __init__(self, n_channels: int):
        assert n_channels in [1, 3]
        if n_channels == 1:
            mean = [0.1307]
            std = [0.3081]
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        super().__init__(mean=mean, std=std)
