import torch
from torch import nn

from src.architectures.feature_extractors.base import FeatureExtractor
from src.architectures.head import ClassificationHead
from src.evaluation.metrics import multiclass_classification_metrics
from src.module.base import BaseImageClassifier
from src.utils.types import Outputs, Tensor


class MulticlassImageClassifier(BaseImageClassifier):
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        head: ClassificationHead,
        classes: list[str],
        lr: float = 0.01,
        weight_decay: float = 0.01,
    ):
        num_classes = len(classes)
        super().__init__(
            feature_extractor=feature_extractor,
            classes=classes,
            lr=lr,
            weight_decay=weight_decay,
            task="multiclass",
            head=head,
            loss_fn=nn.NLLLoss(reduction="none"),
            metrics=multiclass_classification_metrics(num_classes=num_classes, average="weighted"),
        )

    def _produce_outputs(self, images: Tensor, targets: Tensor) -> Outputs:
        log_probs = self(images)
        probs = torch.exp(log_probs)
        loss = self.loss_fn(log_probs, targets)
        preds = log_probs.argmax(dim=1)
        return {"loss": loss, "probs": probs, "preds": preds}
