import pytest
from src.tests.utils import (
    create_hydra_config,
    CONFIGS_PATH,
    CONFIG_NAME,
)
import hydra
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers import Logger


@pytest.mark.parametrize(
    "cfg_path, expected",
    [
        ["wandb.yaml", WandbLogger],
    ],
)
def test_logger_instantiation(
    cfg_path: str,
    expected: Logger,
    tmp_path,
) -> None:
    with hydra.initialize(version_base=None, config_path=str(CONFIGS_PATH)):
        cfg = create_hydra_config(
            experiment_name=None,
            overrided_default="logger",
            overrided_config=[cfg_path],
            config_name=CONFIG_NAME,
            output_path=tmp_path,
        )
        cfg.logger.name = "_test"
        logger = hydra.utils.instantiate(cfg.logger)
        assert isinstance(logger, expected)
