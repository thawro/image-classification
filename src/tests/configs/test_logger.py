import hydra
import pytest
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers.wandb import WandbLogger

from src.tests.utils import CONFIG_NAME, CONFIGS_PATH, create_hydra_config


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
            overrided_cfgs={"logger": [cfg_path]},
            config_name=CONFIG_NAME,
            output_path=tmp_path,
        )
        cfg.logger.name = "_test_logger"
        logger = hydra.utils.instantiate(cfg.logger)
        assert isinstance(logger, expected)
