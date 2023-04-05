import hydra
import pytest
from pytorch_lightning.callbacks import Callback, EarlyStopping, RichProgressBar

from src.evaluation.callbacks import ExamplePredictionsLogger
from src.tests.utils import CONFIG_NAME, CONFIGS_PATH, create_hydra_config


@pytest.mark.parametrize(
    "cfg_path, expected",
    [
        ["best_examples.yaml", [ExamplePredictionsLogger]],
        ["random_examples.yaml", [ExamplePredictionsLogger]],
        ["worst_examples.yaml", [ExamplePredictionsLogger]],
        ["early_stopping.yaml", [EarlyStopping]],
        ["rich_progress_bar.yaml", [RichProgressBar]],
        [
            "default.yaml",
            [
                RichProgressBar,
                EarlyStopping,
                ExamplePredictionsLogger,
                ExamplePredictionsLogger,
                ExamplePredictionsLogger,
            ],
        ],
    ],
)
def test_callbacks_initialize(
    cfg_path: str,
    expected: list[Callback],
    tmp_path,
) -> None:
    with hydra.initialize(version_base=None, config_path=str(CONFIGS_PATH)):
        cfg = create_hydra_config(
            experiment_name=None,
            overrided_cfgs={"callbacks": [cfg_path]},
            config_name=CONFIG_NAME,
            output_path=tmp_path,
        )
        for i, (_, params) in enumerate(cfg.callbacks.items()):
            callback = hydra.utils.instantiate(params)
            assert isinstance(callback, expected[i])
