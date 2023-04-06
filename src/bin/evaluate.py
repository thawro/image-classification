import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything

from src.architectures.model import ImageClassifier
from src.data.datamodule import ImageDataModule
from src.utils.hydra import (
    instantiate_datamodule,
    instantiate_logger,
    instantiate_model,
    instantiate_trainer,
)
from src.utils.utils import close_loggers, log, print_config_tree


def evaluate(
    trainer: Trainer,
    model: ImageClassifier,
    datamodule: ImageDataModule,
    ckpt_path: str,
):
    log.info("Evaluating on validation set..")
    trainer.validate(model, datamodule, ckpt_path=ckpt_path)

    log.info("Evaluating on test set..")
    trainer.test(model, datamodule, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path="../../configs", config_name="evaluate")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    print_config_tree(cfg, keys="all")
    torch.set_float32_matmul_precision("medium")
    datamodule = instantiate_datamodule(cfg)
    model = instantiate_model(cfg, datamodule=datamodule)
    if cfg.run_name == "auto":
        cfg.run_name = f"{model.name}_evaluate"
    logger = instantiate_logger(cfg)
    params = {"dataset": datamodule.name, "model": model.name}
    logger.log_hyperparams(params)
    trainer = instantiate_trainer(cfg, callbacks=[], logger=logger)
    evaluate(trainer, model, datamodule, cfg.ckpt_path)
    close_loggers()


if __name__ == "__main__":
    main()
