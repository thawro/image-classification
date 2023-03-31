from sklearn.metrics import accuracy_score, f1_score
from src.utils.types import Tensor


def get_classification_metrics(y_true: Tensor, y_pred: Tensor) -> dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    fscore = float(f1_score(y_true, y_pred, average="macro"))
    return {"accuracy": acc, "fscore": fscore}
