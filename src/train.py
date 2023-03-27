from torch import nn
from torchvision import transforms as T
import hydra
from omegaconf import DictConfig, OmegaConf
import torch


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    mean = [0.1307]
    std = [0.3081]
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            # nn.Flatten(start_dim=0, end_dim=-1),
        ]
    )
    datamodule = hydra.utils.instantiate(cfg.datamodule)(transform=transform)
    datamodule.download_data()
    datamodule.setup(stage="fit")
    # feature_extractor = hydra.utils.instantiate(cfg.feature_extractor)(in_dim=28 * 28)
    feature_extractor = hydra.utils.instantiate(cfg.feature_extractor)(in_channels=1)
    head = hydra.utils.instantiate(cfg.head)(feature_extractor.out_shape, datamodule.n_classes)
    model = hydra.utils.instantiate(cfg.model)(feature_extractor=feature_extractor, head=head)
    logger = hydra.utils.instantiate(cfg.logger)
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
