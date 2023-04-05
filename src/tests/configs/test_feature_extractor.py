import hydra
import pytest

from src.architectures.feature_extractors import base, cnn, mlp, resnet
from src.data.datamodule import ImageDataset
from src.tests.utils import CONFIG_NAME, CONFIGS_PATH, create_hydra_config
from src.utils.hydra import instantiate_feature_extractor


@pytest.mark.parametrize(
    "cfg_path, expected",
    [
        ["deep_cnn.yaml", cnn.DeepCNN],
        ["mlp.yaml", mlp.MLP],
        ["resnet.yaml", resnet.ResNet],
    ],
)
def test_feature_extraction_instantiation(
    cfg_path: str,
    expected: base.FeatureExtractor,
    mnist_dataset: ImageDataset,
    tmp_path,
) -> None:
    with hydra.initialize(version_base=None, config_path=str(CONFIGS_PATH)):
        cfg = create_hydra_config(
            experiment_name=None,
            overrided_cfgs={"feature_extractor": [cfg_path]},
            config_name=CONFIG_NAME,
            output_path=tmp_path,
        )
        feature_extractor = instantiate_feature_extractor(cfg, mnist_dataset.dummy_input_shape)
        assert isinstance(feature_extractor, expected)
