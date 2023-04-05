import pytest
from src.tests.utils import (
    create_hydra_config,
    CONFIGS_PATH,
    CONFIG_NAME,
)
import hydra
from src.architectures.head import ClassificationHead


@pytest.mark.parametrize(
    "cfg_path, expected",
    [
        ["default.yaml", ClassificationHead],
    ],
)
def test_head_instantiation(
    cfg_path: str,
    expected: ClassificationHead,
    tmp_path,
) -> None:
    with hydra.initialize(version_base=None, config_path=str(CONFIGS_PATH)):
        cfg = create_hydra_config(
            experiment_name=None,
            overrided_default="head",
            overrided_config=[cfg_path],
            config_name=CONFIG_NAME,
            output_path=tmp_path,
        )
        head = hydra.utils.instantiate(cfg.head)
        assert isinstance(head, expected)
