from __future__ import annotations

import numpy as np


def limits_normalize(points, train_min, train_max, eps=1e-4):
    """Active DAPIER-ACT PointCloud normalization.

    Matches DP3PointGlobalEncoder:
        2 * (xyz - min) / (max - min) - 1

    No clipping is applied. Validation/real points outside the training limits
    are intentionally allowed to produce values outside [-1, 1].
    """
    points = np.asarray(points, dtype=np.float32)
    train_min = np.asarray(train_min, dtype=np.float32).reshape(1, 3)
    train_max = np.asarray(train_max, dtype=np.float32).reshape(1, 3)

    scale = train_max - train_min
    scale = np.where(scale < eps, 1.0, scale).astype(np.float32)

    return (2.0 * (points - train_min) / scale - 1.0).astype(np.float32)


def out_of_training_range_ratio(normalized):
    """Fraction of points with any coordinate outside the trained [-1, 1] range."""
    normalized = np.asarray(normalized)
    if normalized.size == 0:
        return 1.0

    bad = np.any(
        (normalized < -1.0) | (normalized > 1.0),
        axis=1,
    )
    return float(np.mean(bad))


def legacy_zero_one_minmax(points, train_min, train_max, eps=1e-8):
    """Old scaffold helper kept only for explicit compatibility experiments.

    This is NOT the normalization used by the active DP3-head checkpoint.
    """
    points = np.asarray(points, dtype=np.float32)
    train_min = np.asarray(train_min, dtype=np.float32).reshape(1, 3)
    train_max = np.asarray(train_max, dtype=np.float32).reshape(1, 3)
    scale = np.maximum(train_max - train_min, eps)
    return (points - train_min) / scale
