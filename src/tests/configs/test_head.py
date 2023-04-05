import hydra
import pytest

from src.architectures.head import ClassificationHead
from src.tests.utils import CONFIG_NAME, CONFIGS_PATH, create_hydra_config


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
            overrided_cfgs={"head": [cfg_path]},
            config_name=CONFIG_NAME,
            output_path=tmp_path,
        )
        head = hydra.utils.instantiate(cfg.head)
        assert isinstance(head, expected)
