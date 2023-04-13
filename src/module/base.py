from abc import abstractmethod

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import MetricCollection
from torchtyping import TensorType

import wandb
from src.architectures.feature_extractors.base import FeatureExtractor
from src.utils.namespace import SPLITS
from src.utils.types import Tensor, _metrics_average, _stage


class BaseImageClassifier(LightningModule):
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        classes: list[str],
        lr: float = 1e-3,
    ):
        super().__init__()
        if len(classes) <= 1 or not all(isinstance(el, str) for el in classes):
            raise ValueError("classes must be list of strings and its length must be greater than 1")
        self.feature_extractor = feature_extractor
        self.classes = classes
        self.num_classes = len(classes)
        self.lr = lr
        self.save_hyperparameters(ignore=["feature_extractor", "head"])
        self.outputs = {split: [] for split in SPLITS}
        self.examples = {split: {} for split in SPLITS}
        self.logged_metrics = {}

        metrics = self.get_classification_metrics(num_classes=self.num_classes, average="weighted")

        self.train_metrics = metrics.clone(prefix=f"train/")
        self.val_metrics = metrics.clone(prefix=f"val/")
        self.test_metrics = metrics.clone(prefix=f"test/")
        self.metrics = {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "test": self.test_metrics,
        }

        self.head = nn.Identity()
        self.loss_fn = nn.Identity()

    @classmethod
    @abstractmethod
    def get_classification_metrics(cls, num_classes: int, average: _metrics_average) -> MetricCollection:
        raise NotImplementedError()

    @property
    def name(self):
        return self.feature_extractor.name

    def forward(self, x: TensorType["batch", "channels", "height", "width"]) -> TensorType["batch", "n_classes"]:
        features = self.feature_extractor(x)
        return self.head(features)

    @abstractmethod
    def _common_step(
        self,
        batch: TensorType["batch", "channels", "height", "width"],
        batch_idx: int,
        stage: _stage,
    ) -> Tensor:
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage="test")

    def _common_epoch_end(self, stage: _stage):
        outputs = self.outputs[stage]
        loss = torch.concat([output["loss"] for output in outputs]).mean().item()
        metrics = self.metrics[stage].compute()
        if self.trainer.sanity_checking:
            return loss
        loss_name = f"{stage}/loss"
        self.log(loss_name, loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.logged_metrics.update({k: v.item() for k, v in metrics.items()})
        self.logged_metrics[loss_name] = loss
        # self.logger.experiment.log(self.logged_metrics, step=self.current_epoch) # TODO: bug with media logging
        wandb.log(self.logged_metrics, step=self.current_epoch)

        print(self.logged_metrics.keys())
        outputs.clear()
        self.logged_metrics.clear()
        self.metrics[stage].reset()

    def on_train_epoch_end(self):
        self._common_epoch_end("train")

    def on_validation_epoch_end(self):
        self._common_epoch_end("val")

    def on_test_epoch_end(self):
        self._common_epoch_end("test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
