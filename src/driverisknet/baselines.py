import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def _score(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)),
            "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0))}


def majority_baseline(y_train, y_test):
    majority = int(np.bincount(y_train).argmax())
    return {**_score(y_test, np.full_like(y_test, majority)), "class_id": majority}


def fit_ttc_heuristic(x_train, y_train):
    """Fit two thresholds using only minimum TTC; intentionally weak sanity baseline."""
    values = x_train[:, :, 6].min(1)
    candidates = np.quantile(values, np.linspace(.05, .95, 31))
    best = (-1., None)
    for critical in candidates[:-1]:
        for warning in candidates[candidates > critical]:
            pred = np.where(values <= critical, 2, np.where(values <= warning, 1, 0))
            score = f1_score(y_train, pred, average="macro", zero_division=0)
            if score > best[0]: best = (score, (float(critical), float(warning)))
    return best[1]


def ttc_heuristic(x, y, thresholds):
    critical, warning = thresholds; values = x[:, :, 6].min(1)
    pred = np.where(values <= critical, 2, np.where(values <= warning, 1, 0))
    return {**_score(y, pred), "critical_below": critical, "warning_below": warning}

