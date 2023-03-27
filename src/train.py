from torchvision import transforms as T
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from data.datamodule import ImageDataModule
from utils import print_config_tree


def create_datamodule(cfg: DictConfig) -> ImageDataModule:
    transforms = cfg.transforms.items()
    train_transform = T.Compose([instantiate(transform["train"]) for _, transform in transforms])
    inference_transform = T.Compose([instantiate(transform["inference"]) for _, transform in transforms])
    datamodule = instantiate(cfg.datamodule)(train_transform=train_transform, inference_transform=inference_transform)
    datamodule.download_data()
    datamodule.setup(stage="fit")
    return datamodule


def create_model(cfg: DictConfig, datamodule: ImageDataModule):
    feature_extractor_name = cfg.feature_extractor._target_.split(".")[-1]
    sample_shape = datamodule.train[0][0].shape
    if any([name in feature_extractor_name.lower() for name in ["resnet", "cnn"]]):  # ResNet, DeepCNN
        in_channels = sample_shape[0]
        feature_extractor = instantiate(cfg.feature_extractor)(in_channels=in_channels)
    else:  # MLP
        in_dim = sample_shape.numel()
        feature_extractor = instantiate(cfg.feature_extractor)(in_dim=in_dim)
    head = instantiate(cfg.head)(feature_extractor.out_shape, datamodule.n_classes)
    model = instantiate(cfg.model)(feature_extractor=feature_extractor, head=head)
    return model


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    print_config_tree(cfg, keys="all")
    datamodule = create_datamodule(cfg)
    train_fig = datamodule.plot_images(split="train")
    val_fig = datamodule.plot_images(split="val")
    train_fig.savefig("train.png")
    val_fig.savefig("val.png")
    # datamodule.plot_images(split="test")
    model = create_model(cfg, datamodule)
    logger = instantiate(cfg.logger)
    trainer = instantiate(cfg.trainer, logger=logger)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
