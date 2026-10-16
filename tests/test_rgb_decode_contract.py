import cv2
import h5py
import numpy as np

from dapier_act.dataset_multimodal import (
    RGB_DECODE_HISTORICAL,
    RGB_DECODE_XPOLICYLAB,
    decode_rgb_frame,
)


def _make_legacy_like_jpeg():
    # XPolicyLab legacy trajectories were produced by handing an RGB array
    # directly to cv2.imencode. We intentionally mimic that storage contract.
    image_rgb_semantics = np.zeros((32, 48, 3), dtype=np.uint8)
    image_rgb_semantics[:, :16] = [230, 20, 10]
    image_rgb_semantics[:, 16:32] = [20, 220, 30]
    image_rgb_semantics[:, 32:] = [15, 40, 225]

    ok, encoded = cv2.imencode(
        ".jpg",
        image_rgb_semantics,
        [int(cv2.IMWRITE_JPEG_QUALITY), 100],
    )
    assert ok
    return encoded.reshape(-1)


def _write_episode(path, encoded):
    vlen = h5py.vlen_dtype(np.dtype("uint8"))
    with h5py.File(path, "w") as root:
        group = root.require_group("vision/cam_left_wrist")
        ds = group.create_dataset("colors", shape=(1,), dtype=vlen)
        ds[0] = encoded


def test_historical_mode_is_exact_rb_swap_of_xpolicylab_mode(tmp_path):
    episode = tmp_path / "episode.hdf5"
    _write_episode(episode, _make_legacy_like_jpeg())

    with h5py.File(episode, "r") as root:
        official = decode_rgb_frame(
            root,
            "cam_left_wrist",
            0,
            decode_mode=RGB_DECODE_XPOLICYLAB,
        )
        historical = decode_rgb_frame(
            root,
            "cam_left_wrist",
            0,
            decode_mode=RGB_DECODE_HISTORICAL,
        )

    assert official.shape == historical.shape
    assert np.array_equal(historical[..., ::-1], official)
    assert not np.array_equal(historical, official)


def test_unknown_rgb_decode_mode_is_rejected(tmp_path):
    episode = tmp_path / "episode.hdf5"
    _write_episode(episode, _make_legacy_like_jpeg())

    with h5py.File(episode, "r") as root:
        try:
            decode_rgb_frame(
                root,
                "cam_left_wrist",
                0,
                decode_mode="ambiguous",
            )
        except ValueError as exc:
            assert "Unsupported RGB decode mode" in str(exc)
        else:
            raise AssertionError("Unknown decode mode was accepted")
