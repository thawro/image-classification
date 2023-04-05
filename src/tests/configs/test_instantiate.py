import hydra
import pytest
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, RichProgressBar
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import nn
from torch.nn import Flatten, Identity
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ToTensor

from src.architectures.feature_extractors import base, cnn, mlp, resnet
from src.architectures.head import ClassificationHead
from src.architectures.model import ImageClassifier
from src.data.datamodule import (
    CIFAR10DataModule,
    CIFAR100DataModule,
    ImageDataModule,
    ImageDataset,
    MNISTDataModule,
)
from src.data.transforms import ImgNormalize, Permute
from src.evaluation.callbacks import ExamplePredictionsLogger
from src.tests.utils import CONFIG_NAME, CONFIGS_PATH, create_hydra_config
from src.utils.hydra import instantiate_feature_extractor
from src.utils.types import Callable


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
def test_callbacks(
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


@pytest.mark.parametrize(
    "cfg_path, expected",
    [
        ["cifar10.yaml", CIFAR10DataModule],
        ["cifar100.yaml", CIFAR100DataModule],
        ["mnist.yaml", MNISTDataModule],
    ],
)
def test_datamodule(
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


@pytest.mark.parametrize(
    "cfg_path, expected",
    [
        ["deep_cnn.yaml", cnn.DeepCNN],
        ["mlp.yaml", mlp.MLP],
        ["resnet.yaml", resnet.ResNet],
    ],
)
def test_feature_extractor(
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


@pytest.mark.parametrize(
    "cfg_path, expected",
    [
        ["default.yaml", ClassificationHead],
    ],
)
def test_head(
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


@pytest.mark.parametrize(
    "cfg_path, expected",
    [
        ["wandb.yaml", WandbLogger],
    ],
)
def test_logger(
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
        cfg.run_name = "_test_logger_initialize"
        logger = hydra.utils.instantiate(cfg.logger)
        assert isinstance(logger, expected)


@pytest.mark.parametrize(
    "cfg_path, expected",
    [
        ["default.yaml", ImageClassifier],
    ],
)
def test_model(
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


@pytest.mark.parametrize(
    "cfg_path, expected",
    [
        ["default.yaml", Trainer],
    ],
)
def test_trainer(
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


@pytest.mark.parametrize(
    "cfg_path, expected",
    [
        ["flatten.yaml", [(Flatten, Flatten)]],
        ["grey_normalize.yaml", [(ImgNormalize, ImgNormalize)]],
        ["rgb_normalize.yaml", [(ImgNormalize, ImgNormalize)]],
        ["horizontal_flip.yaml", [(RandomHorizontalFlip, Identity)]],
        ["permute.yaml", [(Permute, Permute)]],
        ["rotation.yaml", [(RandomRotation, Identity)]],
        ["to_tensor.yaml", [(ToTensor, ToTensor)]],
        ["default.yaml", [(ToTensor, ToTensor), (ImgNormalize, ImgNormalize)]],
    ],
)
def test_transforms(
    cfg_path: str,
    expected: list[tuple[Callable, Callable]],
    tmp_path,
) -> None:
    with hydra.initialize(version_base=None, config_path=str(CONFIGS_PATH)):
        cfg = create_hydra_config(
            experiment_name=None,
            overrided_cfgs={"transforms": [cfg_path]},
            config_name=CONFIG_NAME,
            output_path=tmp_path,
        )
        for i, (_, params) in enumerate(cfg.transforms.items()):
            train_transform = hydra.utils.instantiate(params["train"])
            inference_transform = hydra.utils.instantiate(params["inference"])
            train_expected, inference_expected = expected[i]
            assert isinstance(train_transform, train_expected)
            assert isinstance(inference_transform, inference_expected)
