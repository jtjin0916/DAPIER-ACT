from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from dapier_act.observation.schema import CanonicalObservation


@dataclass
class RawRealObservation:
    depth: np.ndarray
    left_rgb: np.ndarray
    right_rgb: np.ndarray
    raw_joint_state: object
    timestamps_s: dict[str, float]


class RealObservationAdapter:
    """Hardware integration boundary.

    Intentionally incomplete: implement only after verifying actual SO-101
    driver/camera APIs. Do not guess motor ordering or gripper units.
    """

    def __init__(self, joint_mapper, pointcloud_preprocessor):
        self.joint_mapper = joint_mapper
        self.pointcloud_preprocessor = pointcloud_preprocessor

    def __call__(self, raw):
        points = self.pointcloud_preprocessor(raw.depth)
        qpos = self.joint_mapper(raw.raw_joint_state)
        obs = CanonicalObservation(
            head_points=points,
            left_wrist_rgb=raw.left_rgb,
            right_wrist_rgb=raw.right_rgb,
            qpos=qpos,
            timestamps_s=raw.timestamps_s,
        )
        obs.validate()
        return obs
