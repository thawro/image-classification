from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelF1Score,
)

from src.utils.types import _metrics_average


def multiclass_classification_metrics(num_classes: int, average: _metrics_average = "macro") -> MetricCollection:
    return MetricCollection(
        {
            "fscore": MulticlassF1Score(num_classes=num_classes, average=average),
            "accuracy": MulticlassAccuracy(num_classes=num_classes, average=average),
            # "auroc": MulticlassAUROC(num_classes=num_classes, average=average),
        }
    )


def multilabel_classification_metrics(num_classes: int, average: _metrics_average = "macro") -> MetricCollection:
    return MetricCollection(
        {
            "accuracy": MultilabelAccuracy(num_labels=num_classes, average=average),
            "fscore": MultilabelF1Score(num_labels=num_classes, average=average),
            # "auroc": MultilabelAUROC(num_classes=num_classes, average=average),
        }
    )
