import torch.nn.functional as F
from evaluation.metrics import get_classification_metrics
from architectures.feature_extractors.base import FeatureExtractor
from architectures.head import ClassificationHead
from pytorch_lightning import LightningModule
import torch


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

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.head(features)

    def _common_step(self, batch, batch_idx, stage):
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

    def _common_epoch_end(self, stage: str):
        outputs = getattr(self, f"{stage}_step_outputs")
        preds = torch.concat([output["preds"] for output in outputs])
        targets = torch.concat([output["targets"] for output in outputs])
        loss = torch.tensor([output["loss"] for output in outputs]).mean().item()
        if self.trainer.state.fn != "fit" or self.trainer.sanity_checking:
            return loss
        metrics = get_classification_metrics(targets, preds) | {"loss": loss}
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)
        metrics = {f"{stage}/{name}": value for name, value in metrics.items()}
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
