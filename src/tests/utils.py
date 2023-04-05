from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import hydra
from src.utils.types import Optional

CONFIGS_PATH = Path("../../../configs")
CONFIGS_PATH_REL = Path("configs")

CONFIG_NAME = "train"


def get_configs(config_path: Path, experiment_path: str) -> list[str]:
    path = config_path / experiment_path
    config_names = [x.name for x in path.iterdir() if x.is_file()]
    return config_names


def create_hydra_config(
    experiment_name: Optional[str],
    overrided_default: str,
    overrided_config: str | list[str],
    config_name: str,
    output_path: Path,
) -> DictConfig:
    overrides = [f"{overrided_default}={overrided_config}", f"hydra.run.dir={output_path}"]
    if experiment_name is not None:
        overrides.insert(0, f"experiment={experiment_name}")
    cfg = hydra.compose(
        config_name=config_name,
        overrides=overrides,
        return_hydra_config=True,
    )
    HydraConfig().set_config(cfg)
    return cfg
