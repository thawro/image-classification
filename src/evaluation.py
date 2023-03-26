from sklearn.metrics import accuracy_score, f1_score


def get_classification_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "fscore": f1_score(y_true, y_pred),
    }
