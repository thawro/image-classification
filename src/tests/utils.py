from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

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
    overrided_cfgs: dict[str, str | list[str]],
    config_name: str,
    output_path: Path,
) -> DictConfig:
    overrides = [f"hydra.run.dir={output_path}"]
    overrides += [f"{name}={cfg_path}" for name, cfg_path in overrided_cfgs.items()]
    if experiment_name is not None:
        overrides.insert(0, f"experiment={experiment_name}")
    cfg = hydra.compose(
        config_name=config_name,
        overrides=overrides,
        return_hydra_config=True,
    )
    HydraConfig().set_config(cfg)
    return cfg


def create_hydra_train_config(
    experiment_name: Optional[str],
    overrided_cfgs: dict[str, str | list[str]],
    config_name: str,
    output_path: Path,
) -> DictConfig | ListConfig:
    overrided_cfgs["debug"] = "default.yaml"
    cfg = create_hydra_config(
        experiment_name=experiment_name,
        overrided_cfgs=overrided_cfgs,
        config_name=config_name,
        output_path=output_path,
    )
    cfg = OmegaConf.to_yaml(cfg)
    cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.hydra.job.id = 0
    cfg.hydra.job.num = 0
    cfg.hydra.hydra_help.hydra_help = True
    cfg.hydra.runtime.output_dir = output_path
    return cfg
