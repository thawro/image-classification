from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger

import wandb
from src.loggers.base import BaseLogger
from src.utils.types import Any, Optional, Tensor
from src.utils.utils import log


class WandbLoggerWrapper(BaseLogger, WandbLogger):
    def __init__(
        self,
        project_name: str,
        run_name: str,
        save_dir: str,
        id: Optional[str] = None,
        **kwargs: Any,
    ):
        BaseLogger.__init__(self, project_name=project_name, run_name=run_name, output_dir=save_dir)
        WandbLogger.__init__(
            self,
            name=run_name,
            save_dir=save_dir,
            id=id,
            project=project_name,
            **kwargs,
        )

    def log_config(self, cfg: dict):
        super().log_config(cfg)
        wandb.config.update(cfg)

    def log_artifact(self, path: str):
        if "." not in path.split("/")[-1]:  # logging whole directory
            path += "/*"
        log.info(f"Saving artifact {path} to wandb run {self.run_name} from project {self.project_name}")
        wandb.save(path, base_path=self.output_dir)

    def log(self, items: dict):
        self.experiment.log(items)

    def log_model(self, model: LightningModule, dummy_input: Tensor):
        raise NotImplementedError()
