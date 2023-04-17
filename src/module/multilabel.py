from torch import nn

from src.architectures.feature_extractors.base import FeatureExtractor
from src.architectures.head import (
    MultilabelConvolutionalClassificationHead,
    MultilabelLinearClassificationHead,
)
from src.evaluation.metrics import multilabel_classification_metrics
from src.module.base import BaseImageClassifier
from src.utils.types import HeadType, Outputs, Tensor


class MultilabelImageClassifier(BaseImageClassifier):
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        head_type: HeadType,
        classes: list[str],
        lr: float = 1e-3,
    ):
        num_classes = len(classes)
        if head_type == "linear":
            HeadClass = MultilabelLinearClassificationHead
        else:
            HeadClass = MultilabelConvolutionalClassificationHead
        super().__init__(
            feature_extractor=feature_extractor,
            classes=classes,
            lr=lr,
            task="multilabel",
            head=HeadClass(feature_extractor.out_dim, num_classes),
            loss_fn=nn.BCELoss(reduction="none"),
            metrics=multilabel_classification_metrics(num_classes=num_classes, average="weighted"),
        )

    def _produce_outputs(
        self,
        images: Tensor,
        targets: Tensor,
    ) -> Outputs:
        probs = self(images)
        loss = self.loss_fn(probs, targets.float())
        preds = probs.argmax(dim=1)
        return {"loss": loss, "probs": probs, "preds": preds}
