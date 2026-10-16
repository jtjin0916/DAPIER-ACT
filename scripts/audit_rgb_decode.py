#!/usr/bin/env python3

import argparse

import h5py
import numpy as np

from dapier_act.dataset_multimodal import decode_rgb_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode")
    parser.add_argument("--camera", default="cam_left_wrist")
    parser.add_argument("--timestep", type=int, default=0)
    args = parser.parse_args()

    try:
        from XPolicyLab.utils.process_data import decode_image_bit
    except Exception as exc:
        raise RuntimeError(
            "XPolicyLab import failed. Run this script from the RoboTwin "
            "environment with XPolicyLab on PYTHONPATH."
        ) from exc

    key = f"vision/{args.camera}/colors"

    with h5py.File(args.episode, "r") as root:
        historical = decode_rgb_frame(
            root,
            args.camera,
            args.timestep,
        )
        raw = root[key][args.timestep]
        official = np.asarray(
            decode_image_bit(raw)
        )

    print("historical:", historical.shape, historical.dtype)
    print("official:  ", official.shape, official.dtype)

    if official.shape != historical.shape:
        print("RESULT: SHAPE MISMATCH")
        raise SystemExit(2)

    same = np.array_equal(historical, official)
    rb_swap = np.array_equal(
        historical[..., ::-1],
        official,
    )

    diff = np.abs(
        historical.astype(np.int16)
        - official.astype(np.int16)
    )

    print("exactly equal:", same)
    print("R/B swap equal:", rb_swap)
    print("mean abs diff:", float(diff.mean()))
    print("max abs diff:", int(diff.max()))

    if same:
        print("RESULT: MATCH")
    elif rb_swap:
        print("RESULT: CHANNEL-ORDER MISMATCH")
    else:
        print("RESULT: DECODER MISMATCH")


if __name__ == "__main__":
    main()
