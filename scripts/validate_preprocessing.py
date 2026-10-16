#!/usr/bin/env python3

import numpy as np

from dapier_act.perception.transforms import invert_transform, transform_points
from dapier_act.perception.crop import crop_xyz
from dapier_act.perception.normalization import (
    limits_normalize,
    out_of_training_range_ratio,
)


def main():
    T_task_camera = np.eye(4, dtype=np.float64)
    T_camera_task = invert_transform(T_task_camera)
    assert np.allclose(
        T_camera_task @ T_task_camera,
        np.eye(4),
        atol=1e-6,
    )

    pts = np.array(
        [
            [0.10, 0.00, 0.10],
            [0.20, 0.20, 0.10],
            [0.50, 0.00, 0.10],
        ],
        dtype=np.float32,
    )

    task = transform_points(T_task_camera, pts)
    cropped = crop_xyz(
        task,
        (0.0, 0.4),
        (-0.25, 0.25),
        (0.005, 0.30),
    )
    assert len(cropped) == 2

    train_min = np.array([0.0, -0.25, 0.0], np.float32)
    train_max = np.array([0.4, 0.25, 0.3], np.float32)

    norm = limits_normalize(
        cropped,
        train_min,
        train_max,
    )

    print("Transform round trip: OK")
    print("Crop: OK")
    print("Active DP3 limits normalization [-1,1]:")
    print(norm)
    print(
        "Out-of-training-range ratio:",
        out_of_training_range_ratio(norm),
    )


if __name__ == "__main__":
    main()
