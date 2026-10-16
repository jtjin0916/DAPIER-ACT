#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch

from dapier_act.model_multimodal import DapierACTMultimodal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    path = Path(args.checkpoint).expanduser()
    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    expected_type = "multimodal_dp3head256_resnet18"
    actual_type = checkpoint.get("model_type")
    if actual_type != expected_type:
        raise ValueError(
            f"Expected model_type={expected_type!r}, got {actual_type!r}"
        )

    cfg = dict(checkpoint.get("model_config", {}))

    if not cfg:
        cfg = {
            "qpos_dim": int(checkpoint.get("qpos_dim", 12)),
            "action_dim": int(checkpoint.get("action_dim", 12)),
            "chunk_size": int(checkpoint["chunk_size"]),
            "hidden_dim": int(checkpoint.get("hidden_dim", 256)),
            "latent_dim": int(checkpoint.get("latent_dim", 32)),
            "num_point_samples": int(
                checkpoint.get("num_point_samples", 512)
            ),
            "candidate_points": int(
                checkpoint.get("candidate_points", 8192)
            ),
            "point_min": checkpoint["point_min"],
            "point_max": checkpoint["point_max"],
            "nheads": 8,
            "encoder_layers": 4,
            "decoder_layers": 4,
            "ff_dim": 1024,
            "dropout": 0.0,
            "pretrained_wrist_backbone": False,
        }

    # Never download ImageNet weights for a strict checkpoint-load test.
    cfg["pretrained_wrist_backbone"] = False

    model = DapierACTMultimodal(**cfg)
    result = model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    params = sum(p.numel() for p in model.parameters())

    print("checkpoint:", path)
    print("model_type:", actual_type)
    print("chunk_size:", model.chunk_size)
    print("parameters:", f"{params:,}")
    print("missing_keys:", result.missing_keys)
    print("unexpected_keys:", result.unexpected_keys)
    print("STRICT LOAD: PASS")


if __name__ == "__main__":
    main()
