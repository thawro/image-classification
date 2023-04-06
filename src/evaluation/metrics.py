from sklearn.metrics import accuracy_score, f1_score

from src.utils.types import Tensor, _int_array, _int_list

_y_type = Tensor | _int_array | _int_list


def get_classification_metrics(y_true: _y_type, y_pred: _y_type) -> dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    fscore = float(f1_score(y_true, y_pred, average="macro"))
    return {"accuracy": acc, "fscore": fscore}
