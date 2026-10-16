from __future__ import annotations
from dataclasses import dataclass
import numpy as np


CANONICAL_JOINT_ORDER = (
    "left_shoulder_pan",
    "left_shoulder_lift",
    "left_elbow_flex",
    "left_wrist_flex",
    "left_wrist_roll",
    "left_gripper",
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow_flex",
    "right_wrist_flex",
    "right_wrist_roll",
    "right_gripper",
)


@dataclass
class CanonicalObservation:
    head_points: np.ndarray
    left_wrist_rgb: np.ndarray
    right_wrist_rgb: np.ndarray
    qpos: np.ndarray
    timestamps_s: dict[str, float]

    def validate(self):
        if self.head_points.shape != (512, 3):
            raise ValueError(f"head_points: {self.head_points.shape}")
        if self.qpos.shape != (12,):
            raise ValueError(f"qpos: {self.qpos.shape}")
        for name, img in (("left_wrist_rgb", self.left_wrist_rgb), ("right_wrist_rgb", self.right_wrist_rgb)):
            if img.ndim != 3 or img.shape[-1] != 3:
                raise ValueError(f"{name}: {img.shape}")
        if not np.isfinite(self.head_points).all():
            raise ValueError("Non-finite head points")
        if not np.isfinite(self.qpos).all():
            raise ValueError("Non-finite qpos")
