import hydra
import torch
import torch.backends.cudnn
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from src.architectures.utils import get_params
from src.bin.evaluate import evaluate
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
    if cfg.run_name == "auto":
        cfg.run_name = model.name
    callbacks = instantiate_callbacks(cfg)
    logger = instantiate_logger(cfg)
    trainer = instantiate_trainer(cfg, logger=logger, callbacks=list(callbacks.values()))
    params = get_params(datamodule, model)
    logger.log_hyperparams(params)
    trainer.fit(model, datamodule=datamodule)

    ckpt_callback: ModelCheckpoint = callbacks["model_checkpoint"]
    evaluate(trainer, model, datamodule, ckpt_callback.best_model_path)
    close_loggers()


if __name__ == "__main__":
    main()
