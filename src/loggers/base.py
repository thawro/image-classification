from abc import abstractmethod

import yaml
from pytorch_lightning import LightningModule

from src.utils.types import Any, Tensor


class BaseLogger:
    def __init__(self, project_name: str, run_name: str, output_dir: str, **kwargs: Any) -> None:
        self.project_name = project_name
        self.run_name = run_name
        self.output_dir = output_dir

    @abstractmethod
    def log_config(self, cfg: dict):
        path = f"{self.output_dir}/hydra_config.yaml"
        with open(str(path), "w") as yaml_file:
            yaml.dump(cfg, yaml_file, default_flow_style=False)
        self.log_artifact(path)

    @abstractmethod
    def log_artifact(self, path: str):
        raise NotImplementedError()

    @abstractmethod
    def log_model(self, model: LightningModule, dummy_input: Tensor):
        raise NotImplementedError()
