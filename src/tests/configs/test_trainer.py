import hydra
import pytest
from pytorch_lightning import Trainer

from src.tests.utils import CONFIG_NAME, CONFIGS_PATH, create_hydra_config


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
            overrided_cfgs={"trainer": [cfg_path]},
            config_name=CONFIG_NAME,
            output_path=tmp_path,
        )
        trainer = hydra.utils.instantiate(cfg.trainer)
        assert isinstance(trainer, expected)
