# DAPIER-ACT DP3-inspired Head 256D

## Final Head pipeline

```text
Raw RoboTwin World XYZ
→ World→Task transform
→ Task crop: X [0.00, 0.40], Y [-0.25, 0.25], Z [-0.01, 0.30] m
→ candidate reduction (8192)
→ metric-space FPS512
→ train-only per-axis min/max limits normalization to [-1, 1]
→ PointNet: 3→64→128→256, LayerNorm + ReLU
→ symmetric max pool
→ 1 × 256D Head token
→ ACT multimodal context
```

The original DP3 256→64 projection is intentionally omitted because ACT uses
`hidden_dim=256`. The two wrist branches remain independent ResNet18 encoders,
80 × 256D tokens each.

## Token shapes

```text
Head        : [B,  1, 256]
Left wrist  : [B, 80, 256]
Right wrist : [B, 80, 256]
Visual      : 161 tokens
+ qpos      : 1 token
+ latent z  : 1 token
Context     : [B, 163, 256]
```

## Training

From the RoboTwin repository root:

```bash
PYTHONPATH=$PWD python dapier_act/train_multimodal.py \
  --dataset-dir data/demo_so101_pgripper_os30a_30/move_woodenblock_randomized/so101_pgripper_so101_pgripper/data \
  --output-dir checkpoints/dapier_act_multimodal_dp3head256_os30a_30 \
  --epochs 50 \
  --batch-size 4 \
  --chunk-size 20 \
  --lr 1e-4 \
  --kl-weight 0.01 \
  --val-episodes 6 \
  --split-seed 42 \
  --num-point-samples 512 \
  --candidate-points 8192 \
  --device cuda
```

Before epoch 1, the trainer:

1. creates the 24/6 episode split,
2. computes qpos/action mean/std from train episodes only,
3. precomputes Task-cropped FPS512 point clouds once for train/validation,
4. fits PointCloud min/max from train FPS512 observations only,
5. saves `dataset_stats.pkl` and the episode split,
6. reuses the fixed-size PointCloud cache through all epochs.

The original HDF5 files are not modified.

## Rerun QA

`view_multimodal_sync_features_rerun.py` now defaults to the final Task frame and
FPS512 configuration. It can visualize raw World PCD, Task PCD, crop, sampled
DP3 points, wrist RGB, joint state/action, and compatible feature activations.

## Baseline preservation

The original `SpatialPointTokenEncoder` and `DapierACTSpatial` are preserved, so
`train_spatial.py` remains the Spatial256 baseline. Copies of all files modified
for this migration are in `dapier_act/legacy/backups/*_before_dp3head256.py`.
