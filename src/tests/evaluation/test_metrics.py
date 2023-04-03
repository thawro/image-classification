import pytest
from src.utils.types import Sequence
from src.evaluation.metrics import get_classification_metrics


@pytest.mark.parametrize(
    "y_true, y_pred",
    [
        [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]],
        [[0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1]],
        [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]],
        [[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]],
        [[0, 1, 2, 3, 4, 5], [0, 1, 2, 0, 0, 0]],
    ],
)
def test_classification_metrics(
    y_true: Sequence,
    y_pred: Sequence,
):
    metrics = get_classification_metrics(y_true, y_pred)
    assert isinstance(metrics, dict)
    assert all(isinstance(value, float) for metric, value in metrics.items())
    assert "accuracy" in metrics
    assert "fscore" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["fscore"] <= 1.0
