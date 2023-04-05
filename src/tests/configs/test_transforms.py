import hydra
import pytest
from torch.nn import Flatten, Identity
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ToTensor

from src.data.transforms import ImgNormalize, Permute
from src.tests.utils import CONFIG_NAME, CONFIGS_PATH, create_hydra_config
from src.utils.types import Callable


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
def test_transforms_instantiation(
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
