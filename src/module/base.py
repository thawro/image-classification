from abc import abstractmethod

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import MetricCollection
from torchtyping import TensorType
from torchvision.models import mobilenet_v3_small

import wandb
from src.architectures.feature_extractors.base import FeatureExtractor
from src.architectures.head import ClassificationHead
from src.architectures.utils import make_named_sequential
from src.evaluation.visualizers import ClassificationVisualizer
from src.utils.namespace import SPLITS
from src.utils.types import Outputs, Tensor, _stage, _task


class BaseImageClassifier(LightningModule):
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        head: ClassificationHead,
        task: _task,
        loss_fn: nn.Module,
        metrics: MetricCollection,
        classes: list[str],
        lr: float = 0.01,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        if len(classes) <= 1 or not all(isinstance(el, str) for el in classes):
            raise ValueError("classes must be list of strings and its length must be greater than 1")
        # TODO
        layers = [("feature_extractor", feature_extractor), ("head", head)]
        self.net = make_named_sequential(layers)
        self.task = task

        self.loss_fn = loss_fn
        self.classes = classes
        self.visualizer = ClassificationVisualizer(task=task, backend="plotly")
        self.num_classes = len(classes)
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=["feature_extractor", "head"])
        self.outputs = {split: [] for split in SPLITS}
        self.examples = {split: {} for split in SPLITS}
        self.logged_metrics = {}

        self.train_metrics = metrics.clone(prefix=f"train/")
        self.val_metrics = metrics.clone(prefix=f"val/")
        self.test_metrics = metrics.clone(prefix=f"test/")
        self.metrics = {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "test": self.test_metrics,
        }

    @property
    def name(self):
        return self.net[0].name

    def forward(self, x: TensorType["batch", "channels", "height", "width"]) -> TensorType["batch", "n_classes"]:
        return self.net(x)

    @abstractmethod
    def _produce_outputs(self, images: Tensor, targets: Tensor) -> Outputs:
        pass

    def _common_step(
        self,
        batch: TensorType["batch", "channels", "height", "width"],
        batch_idx: int,
        stage: _stage,
    ) -> Tensor:
        images, targets = batch
        outputs = self._produce_outputs(images, targets)
        outputs["targets"] = targets
        if stage != "train" and batch_idx == 0:
            examples = {"images": images, "targets": targets} | outputs
            self.examples[stage] = {k: v.cpu() for k, v in examples.items()}
            del examples
        self.metrics[stage].update(outputs["probs"], targets)
        self.outputs[stage].append(outputs)
        return outputs["loss"].mean()

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
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.7,
            patience=7,
            threshold=0.0001,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
