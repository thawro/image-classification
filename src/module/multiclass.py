import torch
from torch import nn
from torchmetrics import MetricCollection

from src.architectures.feature_extractors.base import FeatureExtractor
from src.architectures.head import MulticlassClassificationHead
from src.evaluation.metrics import multiclass_classification_metrics
from src.module.base import BaseImageClassifier
from src.utils.types import Tensor, TensorType, _metrics_average, _stage


class MulticlassImageClassifier(BaseImageClassifier):
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        classes: list[str],
        lr: float = 1e-3,
    ):
        super().__init__(feature_extractor=feature_extractor, classes=classes, lr=lr)
        self.head = MulticlassClassificationHead(in_dim=feature_extractor.out_dim, n_classes=len(classes))
        self.loss_fn = nn.NLLLoss(reduction="none")

    @classmethod
    def get_classification_metrics(cls, num_classes: int, average: _metrics_average) -> MetricCollection:
        return multiclass_classification_metrics(num_classes=num_classes, average=average)

    def _common_step(
        self,
        batch: TensorType["batch", "channels", "height", "width"],
        batch_idx: int,
        stage: _stage,
    ) -> Tensor:
        x, targets = batch
        log_probs = self(x)
        loss = self.loss_fn(log_probs, targets)
        outputs = {"loss": loss, "probs": torch.exp(log_probs).cpu()}
        if stage != "train":
            self.examples[stage] = {"data": x.cpu(), "targets": targets.cpu()} | outputs
        preds = log_probs.argmax(dim=1)
        self.metrics[stage].update(preds, targets)
        self.outputs[stage].append(outputs)
        return loss.mean()
