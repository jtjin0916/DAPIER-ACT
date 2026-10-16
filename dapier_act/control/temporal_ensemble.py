from __future__ import annotations
import numpy as np


def exponential_temporal_ensemble(overlapping_predictions, k=0.01):
    """Combine current-timestep predictions ordered oldest -> newest."""
    if overlapping_predictions.ndim != 2:
        raise ValueError("Expected [M, action_dim]")
    m = overlapping_predictions.shape[0]
    age = np.arange(m - 1, -1, -1, dtype=np.float64)
    weights = np.exp(-k * age)
    weights /= weights.sum()
    return (overlapping_predictions * weights[:, None]).sum(axis=0)
