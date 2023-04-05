import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from src.data.datamodule import ImageDataModule
from src.architectures.model import ImageClassifier
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import Callback
from src.utils.utils import print_config_tree
import torch
from src.utils.hydra import instantiate_datamodule, instantiate_model


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig):
    print_config_tree(cfg, keys="all")
    torch.set_float32_matmul_precision("medium")
    datamodule: ImageDataModule = instantiate_datamodule(cfg)
    model: ImageClassifier = instantiate_model(cfg, datamodule)
    logger: WandbLogger = instantiate(cfg.logger)
    callbacks: list[Callback] = [
        instantiate(callback_cfg) for _, callback_cfg in cfg.callbacks.items()
    ]
    params = {"dataset": datamodule.name, "model": model.feature_extractor.name}
    logger.log_hyperparams(params)
    trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
