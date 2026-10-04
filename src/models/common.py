from __future__ import annotations

import numpy as np


def topk_accuracy(proba: np.ndarray, classes: np.ndarray, y_true, k: int) -> float:
    idx = np.argsort(-proba, axis=1)[:, :k]
    top = classes[idx]
    y = np.asarray(y_true)
    return float((top == y[:, None]).any(axis=1).mean())
