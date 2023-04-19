from collections import OrderedDict

from torch import nn

from src.data.datamodule import ImageDataModule
from src.module.base import BaseImageClassifier


def make_named_sequential(named_layers: list[tuple[str, nn.Module]]):
    return nn.Sequential(OrderedDict(named_layers))


def get_params(datamodule: ImageDataModule, model: BaseImageClassifier):
    params = {
        "learnable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "dataset": datamodule.name,
        "model": model.name,
    }
    params.update(model.feature_extractor.params)
    return params
