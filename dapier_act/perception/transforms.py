from __future__ import annotations
import numpy as np


def validate_transform(T, atol=1e-5):
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform, got {T.shape}")
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=atol):
        raise ValueError("Invalid homogeneous-transform last row")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=atol):
        raise ValueError("Rotation is not orthonormal")
    if np.linalg.det(R) <= 0:
        raise ValueError("Rotation determinant must be positive")


def invert_transform(T_A_B):
    """Return T_B_A from T_A_B."""
    validate_transform(T_A_B)
    R = T_A_B[:3, :3]
    t = T_A_B[:3, 3]
    out = np.eye(4, dtype=T_A_B.dtype)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def transform_points(T_A_B, points_B):
    """Map Nx3 points from frame B to frame A."""
    validate_transform(T_A_B)
    if points_B.ndim != 2 or points_B.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got {points_B.shape}")
    return points_B @ T_A_B[:3, :3].T + T_A_B[:3, 3]
