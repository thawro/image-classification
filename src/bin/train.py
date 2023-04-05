import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from src.utils.hydra import (
    instantiate_callbacks,
    instantiate_datamodule,
    instantiate_logger,
    instantiate_model,
    instantiate_trainer,
)
from src.utils.utils import close_loggers, print_config_tree


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    print_config_tree(cfg, keys="all")
    torch.set_float32_matmul_precision("medium")
    datamodule = instantiate_datamodule(cfg)
    model = instantiate_model(cfg, datamodule=datamodule)
    logger = instantiate_logger(cfg)
    callbacks = instantiate_callbacks(cfg)
    trainer = instantiate_trainer(cfg, logger=logger, callbacks=callbacks)
    params = {"dataset": datamodule.name, "model": model.feature_extractor.name}
    logger.log_hyperparams(params)
    trainer.fit(model, datamodule=datamodule)
    close_loggers()


if __name__ == "__main__":
    main()
