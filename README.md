# DAPIER-ACT

Multimodal Action Chunking Transformer project for dual SO-101 manipulation.

> **Current status:** the working Custom ACT training/evaluation code still lives in the RoboTwin workspace. This public repository is being converted into a standalone project that can (1) train the same Custom ACT, (2) evaluate it in RoboTwin, and (3) later deploy the same observation/action contract on the real dual SO-101.

## Target system

- Head depth -> Task-frame PointCloud
- Left wrist RGB
- Right wrist RGB
- Dual SO-101 qpos
- Output: `20 x 12` joint-space action chunk

Canonical intent:

```text
left  = 5 arm joints + gripper
right = 5 arm joints + gripper
```

Exact joint order, signs, gripper convention, normalization and real motor mapping must be re-verified against the active dataset/driver before real actuation.

## Current Custom ACT architecture

```text
Head Depth
  -> Depth-to-XYZ
  -> Camera -> Task transform
  -> Workspace crop
  -> Candidate <= 8192
  -> FPS 512
  -> Checkpoint-specific normalization
  -> DP3-inspired PointCloud encoder
  -> 1 x 256 global token

Left Wrist RGB
  -> independent ResNet18
  -> layer4 feature map
  -> 1x1 projection 512 -> 256
  -> spatial tokens

Right Wrist RGB
  -> independent ResNet18
  -> layer4 feature map
  -> 1x1 projection 512 -> 256
  -> spatial tokens

qpos + visual tokens + CVAE latent context
  -> ACT/CVAE Transformer
  -> 20 x 12 action chunk
```

The PointCloud branch is **DP3-inspired**; it is not the complete DP3 policy.

## Where the working code currently lives

```text
~/RoboTwin/dapier_act/
    -> training, dataset and Custom ACT-side implementation

~/RoboTwin/XPolicyLab/policy/DAPIER_ACT/
    -> RoboTwin/XPolicyLab evaluation-side integration

~/DAPIER-ACT/
    -> public refactor / Sim-to-Real / validation repository
```

### Migration rule

The first public integration should be behavior-preserving.

1. Copy the working implementation into `reference/`.
2. Make the public repository call that implementation.
3. Freeze one reference episode.
4. Compare preprocessing tensors.
5. Compare raw action output.
6. Only then refactor modules one by one.

Do not silently reimplement the ACT backend and assume old checkpoint compatibility.

## Confirmed in simulation

- [x] Dual SO-101 environment
- [x] Custom multimodal ACT training path
- [x] Head PointCloud branch
- [x] Independent left/right wrist RGB encoders
- [x] ACT/CVAE action-chunk backend in the working RoboTwin tree
- [x] RoboTwin/XPolicyLab closed-loop evaluation path
- [x] Pick -> Grasp -> Lift
- [x] Planned joint-space Return-to-Stow
- [x] Object hold verification
- [x] Wooden-block 50-grid data collection and diagnostic analysis
- [x] Horizontal cylindrical marker
- [x] Marker top-down grasp
- [x] Marker lift + Return-to-Stow smoke test
- [x] Deterministic XY x yaw collection schedule

## Public-repository code already implemented

- [x] depth -> XYZ
- [x] homogeneous coordinate transforms
- [x] Task-frame crop
- [x] candidate reduction
- [x] reference FPS
- [x] PointCloud min/max normalization
- [x] DP3-inspired point frontend
- [x] dual wrist ResNet18 frontend
- [x] canonical observation schema
- [x] temporal ensemble utility
- [x] basic action clamp
- [x] stale-data watchdog scaffold
- [x] coordinate/crop tests
- [x] Sim-to-Real validation docs
- [x] verified working ACT/CVAE runtime migrated from the active RoboTwin reference
- [ ] standalone public training command
- [ ] standalone RoboTwin evaluation command
- [ ] checkpoint-equivalence regression test

## Data collection currently in progress

### Wooden block

Deterministic 50-cell XY grid.

Diagnostic search:

```text
70 attempts
50 successful episodes
20 failed attempts
```

A successful HDF5 count is not necessarily equal to unique-grid coverage because failed seeds may be skipped and schedules may wrap.

### Marker dataset

Current object:

```text
horizontal cylinder
length   = 100 mm
diameter = 18 mm
grasp    = top-down short-axis pinch
```

Smoke test confirmed grasp, ~4.89 cm lift, joint-space Return-to-Stow and hold after return.

Current collection:

```text
Dataset A
50 XY x yaw [0,10,...,90]
= 500 intended cases
```

Planned second collection:

```text
Dataset B
50 XY x yaw [100,110,...,170]
= 400 intended cases
```

Final target:

```text
50 XY x 18 yaw
= 900 deterministic XY-yaw cases
```

The current 0-90 degree collection is intentionally being allowed to finish before adding 100-170 degrees.

## Generalization plan

A larger training set alone is **not** proof of generalization.

- Spatial: train on the 50-cell grid, test between grid positions.
- Orientation: train at 0,10,...,170 deg and test at 5,15,...,175 deg.
- Geometry: evaluate held-out marker-like shapes such as different length/diameter, hexagonal pens, rounded rectangles and cap/body markers.

Generalization should be reported using closed-loop task success on held-out conditions.

## Evaluation metrics

Offline:
- training/validation L1
- KL
- per-action-dimension MAE
- horizon-wise MAE over 20 steps
- raw newest vs temporal-ensemble difference
- PointCloud distribution statistics

Closed loop:
- grasp success
- lift success
- lift height
- object retained during return
- Stow reached
- full task success

Report separately for seen XY/yaw, unseen XY, unseen yaw and unseen geometry.

## Temporal ensembling

A previous simulation comparison found a case where temporal ensembling reduced lift performance relative to the newest raw prediction. Therefore temporal ensembling is treated as an **ablation**, not an assumed improvement.

The runtime should log:
- newest raw action
- ensemble action
- actual executed action

## Sim-to-Real PointCloud contract

The real robot must reproduce the same semantic observation used in simulation.

```text
Depth
 -> metric depth
 -> Camera XYZ
 -> Camera -> Task transform
 -> Task-frame crop
 -> candidate reduction
 -> FPS 512
 -> checkpoint-specific normalization
 -> PointCloud encoder
```

Task-frame meaning:

```text
+X = robot forward
+Y = robot left
+Z = up
```

Transform convention:

```text
T_A_B maps coordinates in B into A

p_base = T_base_camera @ p_camera
p_task = T_task_base @ p_base

T_task_camera = T_task_base @ T_base_camera
```

Simulator numeric transforms must not be copied directly to real hardware.

## Legacy PointCloud reference

```text
crop X: [0.00, 0.40]
crop Y: [-0.25, 0.25]
crop Z: [0.005, 0.30]

candidate <= 8192
FPS = 512
```

Legacy v4 normalization values are examples only and must not be reused for a new generalized checkpoint unless its metadata explicitly references them.

## Checkpoint contract

Do not release only `best.pt`.

Recommended:

```text
checkpoint/
├── best.pt
├── model_config.yaml
├── preprocessing.yaml
├── normalization.npz
└── metadata.json
```

## Real-hardware status

Still unverified:

- [ ] real head-camera extrinsic
- [ ] real Task-frame origin
- [ ] Sim/Real PointCloud equivalence
- [ ] Sim/Real RGB equivalence
- [ ] exact real joint order/sign
- [ ] gripper scaling
- [ ] qpos/action normalization parity
- [ ] sensor synchronization
- [ ] inference/control latency
- [ ] real control frequency
- [ ] real closed-loop success
- [ ] temporal-ensemble benefit on hardware
- [ ] unseen-geometry generalization on hardware

Do not call a failed real grasp "model failure" until frame, preprocessing, normalization, timing and action mapping have been verified.

## Required real-hardware validation order

```text
1. Unit tests
2. Head camera only
3. Raw PointCloud visualization
4. Camera -> Task transform visualization
5. Crop / FPS / normalization QA
6. qpos read-only
7. Recorded observation -> offline inference
8. Live observation -> inference, motors disabled
9. One bounded motor command
10. Low-speed short horizon
11. Closed-loop newest-only policy
12. Temporal-ensemble A/B
13. Full Pick -> Lift -> Stow
```

The real runner should default to dry-run and require explicit actuation enablement.

## RoboTwin RGB compatibility

RoboTwin 2.0 currently documents that HDF5 image bytes should be decoded through its/XPolicyLab `decode_image_bit` path. Do not blindly add a second BGR->RGB conversion in a custom dataloader.

Official RoboTwin:
https://github.com/RoboTwin-Platform/RoboTwin

## Publication / licensing

Before copying working runtime files into this public repository, preserve applicable notices and audit the exact source licenses.

Relevant upstream components:
- RoboTwin
- XPolicyLab
- ACT implementation source
- DP3-derived source if code was copied/adapted
- torchvision/pretrained ResNet weights
- SO-101 assets/URDFs
- camera SDKs
- datasets/checkpoints

RoboTwin upstream currently uses MIT.

XPolicyLab upstream currently uses Apache-2.0. If files are copied or modified from XPolicyLab, preserve its Apache-2.0 attribution/license requirements and mark modified files appropriately.

See `THIRD_PARTY_NOTICES.md`.

## Immediate milestones

- [ ] Import working `~/RoboTwin/dapier_act/` into `reference/robotwin_dapier_act/`.
- [ ] Import required `~/RoboTwin/XPolicyLab/policy/DAPIER_ACT/` into `reference/xpolicylab_dapier_act/`.
- [ ] Compare checkpoint keys and tensor shapes.
- [ ] Freeze one reference episode.
- [ ] Compare old vs public preprocessing tensors.
- [ ] Compare old vs public raw action output.
- [ ] Make this repository trainable without reaching into `~/RoboTwin`.
- [ ] Make RoboTwin evaluation use an adapter into this repository.
- [ ] Finish marker 0-90 degree collection.
- [ ] Collect marker 100-170 degree data.
- [ ] QA all intended 900 XY-yaw combinations.
- [ ] Retrain a generalized candidate checkpoint.
- [ ] Evaluate unseen XY / unseen yaw / unseen geometry.
- [ ] Begin staged real SO-101 validation.

## Critical rule

The hard part of real deployment is not merely loading the neural network. The real system must reproduce the same frame meaning, PointCloud crop, FPS behavior, normalization, RGB preprocessing, qpos semantics, action semantics and timing that the policy saw during training.

## Custom ACT migration verification

The working Custom ACT reference has now been imported from the active RoboTwin workspace.

Verified during migration:

```text
RoboTwin working files
        ==
public reference files
        by SHA256

Active Custom ACT dependency path
        does not depend on
legacy / PTv3 experiment directories

Python compileall
        PASS

Public regression tests
        3 passed
```

The imported reference is kept as a behavior-preserving baseline.

The next migration step is to compare the reference implementation and the clean public implementation using the same:

```text
HDF5 episode
checkpoint
timestep
```

and verify equivalence for:

```text
PointCloud preprocessing
RGB preprocessing
qpos
encoded features
raw 20 x 12 action chunk
```

Only after these regression checks pass should the clean public implementation replace the reference implementation.

## Verified public Custom ACT runtime

The active DP3-head Custom ACT source is now installed under `dapier_act/`
as a behavior-preserving copy of the verified RoboTwin implementation.

Verified architecture:

```text
Head PointCloud:
Task crop -> candidates <=8192 -> metric FPS512
-> train limits normalization [-1,1]
-> 3->64->128->256 + LayerNorm/ReLU
-> max pool -> 1 token

Left Wrist RGB:
240x320 RGB [0,1] -> independent ResNet18 -> 80 tokens

Right Wrist RGB:
240x320 RGB [0,1] -> independent ResNet18 -> 80 tokens

Context:
latent 1 + qpos 1 + head 1 + left 80 + right 80
= 163 x 256

ACT/CVAE:
latent 32
8 attention heads
4-layer context encoder
4-layer action decoder
20 learned action queries
-> 20 x 12 action chunk
```

The former provisional `dapier_act/model/` scaffold is retained under
`experimental/provisional_model_scaffold/` for history, but it is not the
active checkpoint-compatible implementation.

Remaining validation gates:

- strict-load the existing v4 checkpoint,
- compare actual HDF5 RGB decoding against XPolicyLab `decode_image_bit`,
- freeze an episode/timestep and compare model-ready tensors,
- evaluate temporal aggregation variants,
- verify RoboTwin standalone evaluation,
- calibrate and validate real SO-101 observation/action semantics.

## Wrist RGB contract

RGB decoding is now explicit because an audit found a reproducible channel
mismatch between the historical DAPIER HDF5 loader and the current local
XPolicyLab runtime decoder.

Both the v4 wooden-block corpus and the marker corpus satisfy:

```text
historical loader vs XPolicyLab decoder
exactly equal: false
R/B swap equal: true
```

New/generalized training therefore defaults to:

```text
--rgb-decode-mode xpolicylab_rgb
```

Historical v4 training tensors can be reproduced explicitly with:

```text
--rgb-decode-mode historical_bgr2rgb
```

New checkpoints store the selected RGB decode contract in metadata. The
frozen `reference/` implementation is intentionally unchanged.
