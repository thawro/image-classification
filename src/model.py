import torch.nn.functional as F
from evaluation import get_classification_metrics
import pytorch_lightning as pl
import torch
from torch import nn


class ImageClassifier(pl.LightningModule):
    def __init__(self, net: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.net = net
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x):
        return self.net(x)

    def _common_step(self, batch, batch_idx, stage):
        x, labels = batch
        log_probs = self(x)
        preds = log_probs.argmax(axis=1)
        loss = F.nll_loss(log_probs, labels)
        metrics = get_classification_metrics(labels.cpu(), preds.cpu())
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        x, labels = batch
        preds = self.net(x)
        preds = preds.argmax(axis=1)
        return preds

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
