import torch.nn.functional as F
from evaluation import get_classification_metrics

# from lightning.pytorch import LightningModule
from pytorch_lightning import LightningModule
import torch
from torch import nn


class ImageClassifier(LightningModule):
    def __init__(self, feature_extractor: nn.Module, head: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.head = head
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.head(features)

    def _common_step(self, batch, batch_idx, stage):
        x, labels = batch
        log_probs = self(x)
        preds = log_probs.argmax(axis=1)
        loss = F.nll_loss(log_probs, labels)
        metrics = get_classification_metrics(labels.cpu(), preds.cpu())
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, logger=True)
        self.log(
            f"{stage}/acc",
            metrics["accuracy"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        x, labels = batch
        log_probs = self.net(x)
        preds = log_probs.argmax(axis=1)
        return preds

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
