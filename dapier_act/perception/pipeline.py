from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .depth_to_pointcloud import depth_to_xyz
from .transforms import transform_points
from .crop import crop_xyz
from .sampling import deterministic_candidate_subset, farthest_point_sample_numpy
from .normalization import minmax_normalize, out_of_training_range_ratio


@dataclass
class PointCloudQA:
    raw_count: int
    cropped_count: int
    sampled_count: int
    out_of_range_ratio: float


def preprocess_head_pointcloud(
    depth,
    *,
    fx, fy, cx, cy,
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
    camera = depth_to_xyz(depth, fx, fy, cx, cy, depth_scale_to_m)
    task = transform_points(T_task_camera, camera)
    cropped = crop_xyz(task, crop_x, crop_y, crop_z)
    if len(cropped) < num_points:
        raise RuntimeError(f"Only {len(cropped)} Task-frame points remain; need {num_points}.")
    candidates = deterministic_candidate_subset(cropped, candidate_points)
    sampled = farthest_point_sample_numpy(candidates, num_points)
    normalized = minmax_normalize(sampled, train_min, train_max)
    qa = PointCloudQA(
        raw_count=len(camera),
        cropped_count=len(cropped),
        sampled_count=len(sampled),
        out_of_range_ratio=out_of_training_range_ratio(normalized),
    )
    return normalized, qa
