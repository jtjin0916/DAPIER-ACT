from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ActionLimits:
    lower: np.ndarray
    upper: np.ndarray
    max_step: np.ndarray


def validate_action(action, dim=12):
    if action.shape != (dim,):
        raise ValueError(f"Expected ({dim},), got {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("Action contains NaN/Inf")


def clamp_absolute_target(target, current, limits):
    validate_action(target)
    validate_action(current)
    bounded = np.clip(target, limits.lower, limits.upper)
    delta = np.clip(bounded - current, -limits.max_step, limits.max_step)
    return current + delta
