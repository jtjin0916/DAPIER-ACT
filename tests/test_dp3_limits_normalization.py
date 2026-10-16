import numpy as np
import torch

from dapier_act.model_spatial import DP3PointGlobalEncoder
from dapier_act.perception.normalization import (
    limits_normalize,
    out_of_training_range_ratio,
)


def test_numpy_and_model_normalization_match():
    point_min = [0.0, -1.0, 10.0]
    point_max = [2.0, 1.0, 20.0]

    xyz_np = np.array(
        [
            [0.0, -1.0, 10.0],
            [2.0, 1.0, 20.0],
            [3.0, 2.0, 25.0],
        ],
        dtype=np.float32,
    )

    enc = DP3PointGlobalEncoder(
        point_min=point_min,
        point_max=point_max,
    )

    torch_norm = enc.normalize_points(
        torch.from_numpy(xyz_np)
    ).numpy()

    numpy_norm = limits_normalize(
        xyz_np,
        point_min,
        point_max,
    )

    assert np.allclose(torch_norm, numpy_norm)
    assert np.allclose(numpy_norm[0], [-1.0, -1.0, -1.0])
    assert np.allclose(numpy_norm[1], [1.0, 1.0, 1.0])

    # No clipping.
    assert np.all(numpy_norm[2] > 1.0)
    assert out_of_training_range_ratio(numpy_norm) == 1.0 / 3.0
