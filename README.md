# DAPIER-ACT

Multimodal Action Chunking Transformer research scaffold for dual SO-101 manipulation.

**Inputs**
- Head depth -> Task-frame PointCloud
- Left wrist RGB
- Right wrist RGB
- Dual SO-101 joint state

**Policy output**
- Joint-space action chunk
- Intended contract: `20 x 12`

This repository is a **public/refactor scaffold and handoff package**. It separates model code, sensor preprocessing, RoboTwin simulation adapters, real-robot adapters, and Sim-to-Real validation.

> Important: the included generalized frontend is **not a trained generalized checkpoint** and is not guaranteed to be checkpoint-compatible with the current private/legacy DAPIER-ACT implementation. Before publishing weights, compare tensor names/shapes and run regression tests against the original training/evaluation code.

## Current research architecture

```text
Head depth
  -> depth-to-XYZ
  -> Camera -> Task transform
  -> workspace crop
  -> candidate <= 8192
  -> FPS 512
  -> checkpoint-specific normalization
  -> DP3-inspired PointCloud encoder -> 1 x 256 token

Left wrist RGB  -> ResNet18 -> 1x1 projection -> spatial tokens
Right wrist RGB -> ResNet18 -> 1x1 projection -> spatial tokens

qpos -> canonical 12D state
visual tokens + qpos/latent tokens
  -> existing ACT/CVAE backend
  -> 20 x 12 joint-space action chunk
```

The PointCloud encoder should be described as **DP3-inspired**, not as the full DP3 policy.

## Confirmed in simulation

- Dual SO-101 Pick -> Lift -> Return-to-Stow behavior.
- Joint-space Return-to-Stow path.
- Fixed-grid wooden-block data collection.
- Horizontal cylindrical marker top-down grasp.
- Marker lift and hold through return-to-stow.
- Deterministic XY x yaw collection schedule.

## Still unverified on real hardware

- camera extrinsic,
- Task-frame origin,
- Sim/Real PointCloud equivalence,
- RGB preprocessing equivalence,
- joint sign/order equivalence,
- gripper scaling,
- synchronization and latency,
- real closed-loop success rate,
- temporal ensembling benefit,
- unseen-geometry generalization.

Do not call a real-robot failure "model failure" until preprocessing, frames, normalization, timing and action mapping have been ruled out.

## Recommended execution order

```text
1. Unit tests
2. Camera only
3. PointCloud visualization
4. qpos read-only
5. Recorded observation -> offline inference
6. Live observation -> inference, motor disabled
7. Single bounded motor command
8. Low-speed short horizon
9. Closed-loop newest-action policy
10. Temporal-ensemble A/B test
11. Full Pick -> Lift -> Stow
```

## Key files

- `HANDOFF_TO_NEXT_GPT.md`
- `docs/coordinate_frames.md`
- `docs/pointcloud_pipeline.md`
- `docs/sim2real_validation.md`
- `docs/generalization_plan.md`
- `configs/model/dapier_act_generalized.yaml`
- `dapier_act/model/generalized_frontend.py`
- `scripts/validate_preprocessing.py`

## RoboTwin note

RoboTwin 2.0 currently documents XPolicyLab-based evaluation and warns that HDF5 RGB should be decoded via `decode_image_bit`; it already returns RGB. A custom dataloader must not blindly add an extra BGR<->RGB conversion.

Official project: https://github.com/RoboTwin-Platform/RoboTwin

## Publication warning

Before publishing checkpoints/assets/datasets, audit licenses and redistribution terms of RoboTwin-derived code/assets, XPolicyLab, ACT implementation, DP3-derived/adapted code, pretrained ResNet weights, SO-101 assets/URDFs, camera SDKs, and datasets.

See `THIRD_PARTY_NOTICES.md`.
