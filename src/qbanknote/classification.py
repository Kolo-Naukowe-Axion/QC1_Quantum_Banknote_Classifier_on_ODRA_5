"""Classification helpers for hardware evaluation workflows."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def predictions_to_labels(predictions: np.ndarray) -> np.ndarray:
    """Map continuous QNN outputs to labels in ``{-1, 1}``."""
    predictions = np.asarray(predictions).reshape(-1)
    return np.where(predictions > 0, 1, -1).astype(np.float32)


def evaluate_predictions(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Return accuracy and F1 for binary labels in ``{-1, 1}``."""
    labels = predictions_to_labels(predictions)
    y_true = np.asarray(y_true).reshape(-1)
    return {
        "accuracy": float(accuracy_score(y_true, labels)),
        "f1": float(f1_score(y_true, labels, pos_label=1)),
    }
