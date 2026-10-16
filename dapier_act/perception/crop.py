from __future__ import annotations
import numpy as np


def crop_xyz(points, x, y, z):
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got {points.shape}")
    mask = (
        (points[:, 0] >= x[0]) & (points[:, 0] <= x[1]) &
        (points[:, 1] >= y[0]) & (points[:, 1] <= y[1]) &
        (points[:, 2] >= z[0]) & (points[:, 2] <= z[1])
    )
    return points[mask]
