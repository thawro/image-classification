import torch
from torch import nn
from torchmetrics import MetricCollection

from src.architectures.feature_extractors.base import FeatureExtractor
from src.architectures.head import MulticlassClassificationHead
from src.evaluation.metrics import multiclass_classification_metrics
from src.module.base import BaseImageClassifier
from src.utils.types import Outputs, Tensor, TensorType, _metrics_average, _stage


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

    def _produce_outputs(self, imgs: Tensor, targets: Tensor) -> Outputs:
        log_probs = self(imgs)
        probs = torch.exp(log_probs)
        loss = self.loss_fn(log_probs, targets)
        preds = log_probs.argmax(dim=1)
        return {"loss": loss, "probs": probs, "preds": preds}
