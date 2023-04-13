import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger

from src.bin.train import main
from src.data.datamodule import ImageDataModule
from src.module.multiclass import MulticlassImageClassifier
from src.tests.utils import CONFIG_NAME, CONFIGS_PATH, create_hydra_train_config
from src.utils.hydra import (
    instantiate_callbacks,
    instantiate_datamodule,
    instantiate_logger,
    instantiate_model,
    instantiate_trainer,
)
from src.utils.utils import close_loggers


def test_train_initialize(
    example_experiment_name,
    tmp_path,
) -> None:
    with hydra.initialize(version_base=None, config_path=str(CONFIGS_PATH)):
        cfg = create_hydra_train_config(
            experiment_name=example_experiment_name,
            overrided_cfgs={},
            config_name=CONFIG_NAME,
            output_path=tmp_path,
        )
        cfg.run_name = "_test_train_initialize"
        assert cfg.transforms
        assert cfg.datamodule
        assert cfg.feature_extractor
        assert cfg.model
        assert cfg.logger
        assert cfg.callbacks
        assert cfg.trainer

        datamodule = instantiate_datamodule(cfg)
        assert isinstance(datamodule, ImageDataModule)

        model = instantiate_model(cfg, datamodule=datamodule)
        assert isinstance(model, MulticlassImageClassifier)

        logger = instantiate_logger(cfg)
        assert isinstance(logger, Logger)

        callbacks = instantiate_callbacks(cfg)
        print(callbacks)
        assert all(isinstance(cbk, Callback) for name, cbk in callbacks.items())

        trainer = instantiate_trainer(cfg, logger=logger, callbacks=list(callbacks.values()))
        assert isinstance(trainer, Trainer)
        close_loggers()


def test_train_main(
    example_experiment_name,
    tmp_path,
) -> None:
    with hydra.initialize(version_base=None, config_path=str(CONFIGS_PATH)):
        cfg = create_hydra_train_config(
            experiment_name=example_experiment_name,
            overrided_cfgs={},
            config_name=CONFIG_NAME,
            output_path=tmp_path,
        )
        cfg.run_name = "_test_train_main"
        main(cfg)
