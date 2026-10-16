import numpy as np
from dapier_act.perception.crop import crop_xyz


def test_crop():
    p = np.array([[0.10, 0.00, 0.10], [0.20, -0.20, 0.20], [0.50, 0.00, 0.10]])
    c = crop_xyz(p, (0.0, 0.4), (-0.25, 0.25), (0.005, 0.30))
    assert c.shape == (2, 3)
