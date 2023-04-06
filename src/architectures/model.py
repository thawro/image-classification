import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchtyping import TensorType

from src.architectures.feature_extractors.base import FeatureExtractor
from src.architectures.head import ClassificationHead
from src.evaluation.metrics import get_classification_metrics
from src.utils.types import _stage


class ImageClassifier(LightningModule):
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        head: ClassificationHead,
        classes: list[str],
        lr: float = 1e-3,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.head = head
        self.classes = classes
        self.lr = lr
        self.save_hyperparameters(ignore=["feature_extractor", "head"])
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.metrics = {}

    def forward(self, x: TensorType["batch", "channels", "height", "width"]) -> TensorType["batch", "n_classes"]:
        features = self.feature_extractor(x)
        return self.head(features)

    def _common_step(
        self,
        batch: TensorType["batch", "channels", "height", "width"],
        batch_idx: int,
        stage: _stage,
    ) -> torch.Tensor:
        x, targets = batch
        out = self(x)
        log_probs = F.log_softmax(out, dim=1)
        preds = log_probs.argmax(dim=1)
        loss = F.nll_loss(log_probs, targets)
        outputs = {"loss": loss, "preds": preds.cpu(), "targets": targets.cpu()}
        if stage != "train":
            outputs["data"] = x.cpu()
            outputs["probs"] = F.softmax(out, dim=1).cpu()
        getattr(self, f"{stage}_step_outputs").append(outputs)
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage="test")

    def _common_epoch_end(self, stage: _stage):
        outputs = getattr(self, f"{stage}_step_outputs")
        preds = torch.concat([output["preds"] for output in outputs])
        targets = torch.concat([output["targets"] for output in outputs])
        loss = torch.tensor([output["loss"] for output in outputs]).mean().item()
        metrics = get_classification_metrics(targets, preds) | {"loss": loss}

        if self.trainer.sanity_checking:
            return loss
        elif self.trainer.state.fn in ["validate", "test", "predict"]:
            metrics = {f"{stage}_{name}": value for name, value in metrics.items()}
        else:
            metrics = {f"{stage}/{name}": value for name, value in metrics.items()}
            self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)
        self.metrics.update(metrics)
        self.logger.experiment.log(self.metrics, step=self.current_epoch)
        outputs.clear()

    def on_train_epoch_end(self):
        self._common_epoch_end("train")

    def on_validation_epoch_end(self):
        self._common_epoch_end("val")

    def on_test_epoch_end(self):
        self._common_epoch_end("test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    @property
    def name(self):
        return self.feature_extractor.name
