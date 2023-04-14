import hydra
import torch
import torch.backends.cudnn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from src.bin.evaluate import evaluate
from src.utils.hydra import (
    instantiate_callbacks,
    instantiate_datamodule,
    instantiate_logger,
    instantiate_model,
    instantiate_trainer,
)
from src.utils.utils import close_loggers, print_config_tree

torch.backends.cudnn.benchmark = True


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    print_config_tree(cfg, keys="all")
    torch.set_float32_matmul_precision("medium")
    datamodule = instantiate_datamodule(cfg)
    model = instantiate_model(cfg, datamodule=datamodule)
    if cfg.run_name == "auto":
        cfg.run_name = f"{model.name}"
    logger = instantiate_logger(cfg)
    logger.log_config(OmegaConf.to_object(cfg))
    callbacks = instantiate_callbacks(cfg)
    trainer = instantiate_trainer(cfg, logger=logger, callbacks=list(callbacks.values()))
    params = {"dataset": datamodule.name, "model": model.feature_extractor.name}
    logger.log_hyperparams(params)
    trainer.fit(model, datamodule=datamodule)

    ckpt_callback: ModelCheckpoint = callbacks["model_checkpoint"]
    evaluate(trainer, model, datamodule, ckpt_callback.best_model_path)
    close_loggers()


if __name__ == "__main__":
    main()
