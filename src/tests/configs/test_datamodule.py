import hydra
import pytest

from src.data.datamodule import (
    CIFAR10DataModule,
    CIFAR100DataModule,
    ImageDataModule,
    MNISTDataModule,
)
from src.tests.utils import CONFIG_NAME, CONFIGS_PATH, create_hydra_config


@pytest.mark.parametrize(
    "cfg_path, expected",
    [
        ["cifar10.yaml", CIFAR10DataModule],
        ["cifar100.yaml", CIFAR100DataModule],
        ["mnist.yaml", MNISTDataModule],
    ],
)
def test_datamodule_initialize(
    cfg_path: str,
    expected: ImageDataModule,
    tmp_path,
) -> None:
    with hydra.initialize(version_base=None, config_path=str(CONFIGS_PATH)):
        cfg = create_hydra_config(
            experiment_name=None,
            overrided_cfgs={"datamodule": [cfg_path]},
            config_name=CONFIG_NAME,
            output_path=tmp_path,
        )
        datamodule = hydra.utils.instantiate(cfg.datamodule)
        assert isinstance(datamodule, expected)
