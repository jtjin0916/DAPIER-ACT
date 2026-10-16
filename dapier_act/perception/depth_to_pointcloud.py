from __future__ import annotations
import numpy as np


def depth_to_xyz(depth, fx, fy, cx, cy, depth_scale_to_m=1.0):
    """Convert HxW depth image to camera-frame Nx3 points in meters."""
    if depth.ndim != 2:
        raise ValueError(f"Expected HxW depth, got {depth.shape}")
    z = depth.astype(np.float32) * np.float32(depth_scale_to_m)
    h, w = z.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x = (u - np.float32(cx)) * z / np.float32(fx)
    y = (v - np.float32(cy)) * z / np.float32(fy)
    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0)
    return pts[valid]
