from torchvision.transforms import Compose
from omegaconf import DictConfig
from src.data.datamodule import ImageDataModule
from hydra.utils import instantiate
from src.architectures.model import ImageClassifier
from src.architectures.feature_extractors.base import FeatureExtractor
import torch


def instantiate_transforms(cfg: DictConfig) -> tuple[Compose, Compose]:
    transforms = cfg.transforms.items()
    train_transform = Compose([instantiate(transform["train"]) for _, transform in transforms])
    inference_transform = Compose(
        [instantiate(transform["inference"]) for _, transform in transforms]
    )
    return train_transform, inference_transform


def instantiate_feature_extractor(
    cfg: DictConfig, dummy_input_shape: torch.Size
) -> FeatureExtractor:
    class_name = cfg.feature_extractor._target_.split(".")[-1]
    if any([name in class_name.lower() for name in ["resnet", "cnn"]]):  # ResNet, DeepCNN
        in_channels = dummy_input_shape[0]
        feature_extractor = instantiate(cfg.feature_extractor)(in_channels=in_channels)
    else:  # MLP
        in_dim = dummy_input_shape.numel()
        feature_extractor = instantiate(cfg.feature_extractor)(in_dim=in_dim)
    return feature_extractor


def instantiate_datamodule(cfg: DictConfig) -> ImageDataModule:
    train_transform, inference_transform = instantiate_transforms(cfg)
    datamodule = instantiate(
        cfg.datamodule, train_transform=train_transform, inference_transform=inference_transform
    )
    datamodule.download_data()
    datamodule.setup(stage="fit")
    return datamodule


def instantiate_model(cfg: DictConfig, datamodule: ImageDataModule) -> ImageClassifier:
    feature_extractor = instantiate_feature_extractor(cfg, datamodule.train.dummy_input_shape)
    head = instantiate(cfg.head)(in_dim=feature_extractor.out_dim, n_classes=datamodule.n_classes)
    model = instantiate(cfg.model)(
        feature_extractor=feature_extractor, head=head, classes=datamodule.classes
    )
    return model
