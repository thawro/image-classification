import hydra
import pytest
from pytorch_lightning import LightningModule
from torch import nn

from src.architectures.model import ImageClassifier
from src.tests.utils import CONFIG_NAME, CONFIGS_PATH, create_hydra_config


@pytest.mark.parametrize(
    "cfg_path, expected",
    [
        ["default.yaml", ImageClassifier],
    ],
)
def test_model_instantiation(
    cfg_path: str,
    expected: LightningModule,
    tmp_path,
) -> None:
    with hydra.initialize(version_base=None, config_path=str(CONFIGS_PATH)):
        cfg = create_hydra_config(
            experiment_name=None,
            overrided_cfgs={"model": [cfg_path]},
            config_name=CONFIG_NAME,
            output_path=tmp_path,
        )
        model = hydra.utils.instantiate(cfg.model)(feature_extractor=nn.Identity(), head=nn.Identity(), classes=[])
        assert isinstance(model, expected)
