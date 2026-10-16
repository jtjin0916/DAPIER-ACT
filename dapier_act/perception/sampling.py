from __future__ import annotations
import numpy as np


def deterministic_candidate_subset(points, max_points):
    """Reference deterministic reduction for QA.

    Replace with verified historical candidate selection when checkpoint
    compatibility is required.
    """
    if len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[idx]


def farthest_point_sample_numpy(points, n_samples):
    """Simple O(N*K) reference FPS, intended for validation rather than runtime."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got {points.shape}")
    if len(points) < n_samples:
        raise ValueError(f"Need >= {n_samples} points, got {len(points)}")
    selected = np.empty(n_samples, dtype=np.int64)
    selected[0] = 0
    min_dist2 = np.full(len(points), np.inf, dtype=np.float64)
    for i in range(1, n_samples):
        p = points[selected[i - 1]]
        dist2 = np.sum((points - p) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
        selected[i] = int(np.argmax(min_dist2))
    return points[selected]
