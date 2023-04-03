from sklearn.metrics import accuracy_score, f1_score
from src.utils.types import Sequence


def get_classification_metrics(y_true: Sequence, y_pred: Sequence) -> dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    fscore = float(f1_score(y_true, y_pred, average="macro"))
    return {"accuracy": acc, "fscore": fscore}
