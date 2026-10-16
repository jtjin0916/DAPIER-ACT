import numpy as np
from dapier_act.perception.transforms import invert_transform, transform_points


def test_inverse_roundtrip():
    T = np.eye(4)
    T[:3, 3] = [0.1, -0.2, 0.3]
    inv = invert_transform(T)
    assert np.allclose(inv @ T, np.eye(4), atol=1e-8)
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    out = transform_points(T, pts)
    back = transform_points(inv, out)
    assert np.allclose(back, pts)
