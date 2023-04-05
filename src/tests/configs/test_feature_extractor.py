import pytest
from src.tests.utils import (
    create_hydra_config,
    CONFIGS_PATH,
    CONFIG_NAME,
)
import hydra
from src.architectures.feature_extractors import mlp, cnn, resnet, base
from src.utils.hydra import instantiate_feature_extractor
from src.data.datamodule import ImageDataset


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
            overrided_default="feature_extractor",
            overrided_config=[cfg_path],
            config_name=CONFIG_NAME,
            output_path=tmp_path,
        )
        feature_extractor = instantiate_feature_extractor(cfg, mnist_dataset.dummy_input_shape)
        assert isinstance(feature_extractor, expected)
