from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .depth_to_pointcloud import depth_to_xyz
from .transforms import transform_points
from .crop import crop_xyz
from .sampling import deterministic_candidate_subset, farthest_point_sample_numpy
from .normalization import limits_normalize, out_of_training_range_ratio


@dataclass
class PointCloudQA:
    raw_count: int
    cropped_count: int
    sampled_count: int
    out_of_range_ratio: float
    sampled_min: tuple[float, float, float]
    sampled_max: tuple[float, float, float]
    normalized_min: tuple[float, float, float]
    normalized_max: tuple[float, float, float]


def preprocess_head_pointcloud(
    depth,
    *,
    fx,
    fy,
    cx,
    cy,
    depth_scale_to_m,
    T_task_camera,
    train_min,
    train_max,
    crop_x=(0.00, 0.40),
    crop_y=(-0.25, 0.25),
    crop_z=(0.005, 0.30),
    candidate_points=8192,
    num_points=512,
):
    """Real-camera frontend matching the active DP3-head preprocessing order.

    depth -> camera XYZ -> Task frame -> crop -> candidate reduction
    -> metric-space FPS -> train-limit normalization to [-1,1]

    Note: the active RoboTwin model receives Task-cropped metric XYZ and performs
    candidate reduction/FPS/normalization inside DP3PointGlobalEncoder. This
    helper is for QA / real-camera preprocessing comparison and intentionally
    exposes the exact sampled/normalized result.
    """
    camera = depth_to_xyz(
        depth,
        fx,
        fy,
        cx,
        cy,
        depth_scale_to_m,
    )
    task = transform_points(T_task_camera, camera)
    cropped = crop_xyz(task, crop_x, crop_y, crop_z)

    if len(cropped) == 0:
        raise RuntimeError("Task-frame crop produced zero points.")

    candidates = deterministic_candidate_subset(
        cropped,
        candidate_points,
    )
    sampled = farthest_point_sample_numpy(
        candidates,
        num_points,
    )

    if len(sampled) == 0:
        raise RuntimeError("Point sampling produced zero points.")

    normalized = limits_normalize(
        sampled,
        train_min,
        train_max,
    )

    qa = PointCloudQA(
        raw_count=len(camera),
        cropped_count=len(cropped),
        sampled_count=len(sampled),
        out_of_range_ratio=out_of_training_range_ratio(normalized),
        sampled_min=tuple(np.min(sampled, axis=0).astype(float)),
        sampled_max=tuple(np.max(sampled, axis=0).astype(float)),
        normalized_min=tuple(np.min(normalized, axis=0).astype(float)),
        normalized_max=tuple(np.max(normalized, axis=0).astype(float)),
    )

    return normalized, qa
