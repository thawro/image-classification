from torch import nn
from torchmetrics import MetricCollection

from src.architectures.feature_extractors.base import FeatureExtractor
from src.architectures.head import MultilabelClassificationHead
from src.evaluation.metrics import multilabel_classification_metrics
from src.module.base import BaseImageClassifier
from src.utils.types import Tensor, TensorType, _metrics_average, _stage


class MultilabelImageClassifier(BaseImageClassifier):
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        classes: list[str],
        lr: float = 1e-3,
    ):
        super().__init__(feature_extractor=feature_extractor, classes=classes, lr=lr)
        self.loss_fn = nn.BCELoss(reduction="none")
        self.head = MultilabelClassificationHead(in_dim=feature_extractor.out_dim, n_classes=len(classes))

    @classmethod
    def get_classification_metrics(cls, num_classes: int, average: _metrics_average) -> MetricCollection:
        return multilabel_classification_metrics(num_classes=num_classes, average=average)

    def _common_step(
        self,
        batch: TensorType["batch", "channels", "height", "width"],
        batch_idx: int,
        stage: _stage,
    ) -> Tensor:
        x, targets = batch
        probs = self(x)
        loss = self.loss_fn(probs, targets.float())
        outputs = {"loss": loss, "probs": probs}
        if stage != "train":
            self.examples[stage] = {"data": x.cpu(), "targets": targets.cpu()}
        self.metrics[stage].update(probs, targets)
        self.outputs[stage].append(outputs)
        return loss.mean()
