import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, Trainer
from torchvision.transforms import Compose

from src.architectures.feature_extractors.base import FeatureExtractor
from src.architectures.head import ClassificationHead
from src.data.datamodule import ImageDataModule
from src.loggers.wandb import WandbLoggerWrapper
from src.module.base import BaseImageClassifier
from src.utils.utils import log


def instantiate_transforms(cfg: DictConfig) -> tuple[Compose, Compose]:
    log.info("Instantiating Transforms..")
    transforms = cfg.transforms.items()
    train_transform = Compose([instantiate(transform["train"]) for _, transform in transforms])
    inference_transform = Compose([instantiate(transform["inference"]) for _, transform in transforms])
    return train_transform, inference_transform


def instantiate_datamodule(cfg: DictConfig) -> ImageDataModule:
    log.info("Instantiating DataModule..")
    train_transform, inference_transform = instantiate_transforms(cfg)
    datamodule = instantiate(cfg.datamodule)(train_transform=train_transform, inference_transform=inference_transform)
    datamodule.download_data()
    datamodule.setup(stage="fit")
    return datamodule


def instantiate_feature_extractor(cfg: DictConfig, dummy_input_shape: torch.Size) -> FeatureExtractor:
    log.info("Instantiating Feature Extractor..")
    class_name = cfg.feature_extractor._target_.split(".")[-1]
    if any([name in class_name.lower() for name in ["resnet", "cnn"]]):  # ResNet, DeepCNN
        in_channels = dummy_input_shape[0]
        feature_extractor = instantiate(cfg.feature_extractor)(in_channels=in_channels)
    else:  # MLP
        in_dim = dummy_input_shape.numel()
        feature_extractor = instantiate(cfg.feature_extractor)(in_dim=in_dim)
    return feature_extractor


def instantiate_head(cfg: DictConfig, in_dim: int, n_classes: int) -> ClassificationHead:
    log.info("Instantiating ClassificationHead..")
    return instantiate(cfg.head, in_dim=in_dim, n_classes=n_classes)


def instantiate_model(cfg: DictConfig, datamodule: ImageDataModule) -> BaseImageClassifier:
    log.info("Instantiating Model..")
    if "ckpt_path" in cfg:
        log.info(f"Instantiating Model from {cfg.ckpt_path} checkpoint..")
        return BaseImageClassifier.load_from_checkpoint(cfg.ckpt_path)
    feature_extractor = instantiate_feature_extractor(cfg, datamodule.train.dummy_input_shape)
    return instantiate(cfg.model)(feature_extractor=feature_extractor, classes=datamodule.classes)


def instantiate_logger(cfg: DictConfig) -> WandbLoggerWrapper:
    log.info("Instantiating Logger..")
    logger: WandbLoggerWrapper = instantiate(cfg.logger)
    logger.log_config(OmegaConf.to_object(cfg))
    return logger


def instantiate_callbacks(cfg: DictConfig) -> dict[str, Callback]:
    log.info("Instantiating Callbacks..")
    return {name: instantiate(cbk_cfg) for name, cbk_cfg in cfg.callbacks.items()}


def instantiate_trainer(cfg: DictConfig, logger: WandbLoggerWrapper, callbacks: list[Callback], **kwargs) -> Trainer:
    log.info("Instantiating Trainer..")
    return instantiate(cfg.trainer, logger=logger, callbacks=callbacks, **kwargs)
