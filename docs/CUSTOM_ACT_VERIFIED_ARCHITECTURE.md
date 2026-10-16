# Verified Custom DAPIER-ACT Architecture

This document is derived from the active DP3-head implementation preserved in
`reference/robotwin_dapier_act/`.

## Observation

```text
qpos:             12D
Head PointCloud:  Task-frame cropped XYZ
Left wrist:       RGB 240x320
Right wrist:      RGB 240x320
```

## Base ACT / CVAE

```text
hidden_dim       256
latent_dim        32
attention heads    8
context encoder    4 layers
action decoder     4 layers
CVAE encoder       2 layers
FFN              1024
dropout           0.0
chunk_size         20
action_dim          12
```

Training latent:

```text
[CLS] + qpos + GT action chunk
-> sinusoidal position
-> 2-layer CVAE encoder
-> mu / logvar
-> sampled z
```

Inference latent:

```text
z = 0
```

## Head PointCloud

```text
Task-frame cropped XYZ in meters
-> deterministic linspace candidate reduction <=8192
-> FPS 512 in metric space
-> train-set point_min/point_max normalization:
   2 * (xyz - min)/(max-min) - 1
-> Linear 3->64 + LayerNorm + ReLU
-> Linear 64->128 + LayerNorm + ReLU
-> Linear 128->256 + LayerNorm + ReLU
-> max pool over points
-> one 256D token
```

No clipping is applied after limits normalization.

## Wrist RGB

Each wrist uses an independent encoder.

```text
uint8 RGB
-> float32 [0,1]
-> no resize
-> no ImageNet mean/std normalization
-> ResNet18 with FrozenBatchNorm2d through layer4
-> [B,512,8,10]
-> 1x1 Conv 512->256
-> flatten
-> 80 x 256 tokens
-> add 2D sine position
-> add camera/source embedding
```

Left and right ResNet parameters are not shared.

## Context

```text
latent       1
qpos         1
head         1
left wrist  80
right wrist 80
----------------
total       163 tokens
```

The 163 tokens enter the four-layer context Transformer encoder. Twenty learned
action queries enter the four-layer Transformer decoder and produce `[B,20,12]`.

## State/action normalization

```text
qpos_normalized   = (qpos - qpos_mean) / qpos_std
action_normalized = (action - action_mean) / action_std

inference:
action = prediction * action_std + action_mean
```

## Simulation Task frame

The historical simulation checkpoint uses:

```text
origin_world = (0.0, -0.35, 0.74)
rpy_deg      = (0.0, 0.0, 90.0)
crop_min     = (0.00, -0.25, 0.005)
crop_max     = (0.40,  0.25, 0.30)
```

These numeric simulator transforms are not real-camera calibration values.

## Open audits

1. Strict-load the existing v4 checkpoint into the public runtime.
2. Compare historical OpenCV HDF5 RGB decoding with XPolicyLab
   `decode_image_bit` on the actual dataset.
3. Freeze one episode/timestep and compare model-ready tensors.
4. Compare newest-only vs current temporal aggregation vs corrected
   newest-heavy aggregation.
5. Calibrate real camera -> Task transform before motor actuation.
