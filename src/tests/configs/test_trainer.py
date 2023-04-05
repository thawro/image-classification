import pytest
from src.tests.utils import (
    create_hydra_config,
    CONFIGS_PATH,
    CONFIG_NAME,
)
import hydra
from pytorch_lightning import Trainer


@pytest.mark.parametrize(
    "cfg_path, expected",
    [
        ["default.yaml", Trainer],
    ],
)
def test_callbacks_instantiation(
    cfg_path: str,
    expected: Trainer,
    tmp_path,
) -> None:
    with hydra.initialize(version_base=None, config_path=str(CONFIGS_PATH)):
        cfg = create_hydra_config(
            experiment_name=None,
            overrided_default="trainer",
            overrided_config=[cfg_path],
            config_name=CONFIG_NAME,
            output_path=tmp_path,
        )
        trainer = hydra.utils.instantiate(cfg.trainer)
        assert isinstance(trainer, expected)
