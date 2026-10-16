from __future__ import annotations
import numpy as np


def minmax_normalize(points, train_min, train_max, eps=1e-8):
    train_min = np.asarray(train_min, dtype=np.float32).reshape(1, 3)
    train_max = np.asarray(train_max, dtype=np.float32).reshape(1, 3)
    scale = np.maximum(train_max - train_min, eps)
    return (points.astype(np.float32) - train_min) / scale


def out_of_training_range_ratio(normalized):
    if normalized.size == 0:
        return 1.0
    bad = np.any((normalized < 0.0) | (normalized > 1.0), axis=1)
    return float(np.mean(bad))
