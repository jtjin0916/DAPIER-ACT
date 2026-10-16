# DAPIER-ACT Project Handoff for the Next GPT / Developer

## Operating rule

Do not patch from memory. Read the actual file, config, dataset field, checkpoint metadata and tensor shape first.

For every modification:
1. reproduce,
2. inspect,
3. isolate the smallest cause,
4. modify,
5. run regression tests.

For Sim-to-Real failures, verify sensors/frames/preprocessing/action mapping before blaming the model.

## Goal

Dual SO-101 manipulation using a Custom ACT policy with:
- Head PointCloud,
- Left wrist RGB,
- Right wrist RGB,
- qpos,
- 20 x 12 joint-space action chunks.

The common task objective is:

```text
Stow/Open -> Approach -> Grasp -> Close -> Lift -> Return to Stow while holding
```

Do not mix unrelated objectives into one dataset unless an explicit task-conditioning mechanism is added.

## Current multimodal design

```text
Head PCD -> DP3-inspired 3->64->128->256 point MLP -> max pool -> 256D global token
Left RGB -> independent ResNet18 -> layer4 -> 1x1 512->256 -> spatial tokens
Right RGB -> independent ResNet18 -> layer4 -> 1x1 512->256 -> spatial tokens
qpos -> 12D canonical state
ACT/CVAE transformer -> chunk 20 x 12
```

Treat exact backend details as **to-be-verified against current source** before claiming checkpoint compatibility.

## Simulation Task frame currently used

Task-frame semantic convention:
- +X = robot forward
- +Y = robot left
- +Z = up

Legacy simulation mapping:

```text
Task origin in World: (0, -0.35, 0.74)
X_T = Y_W + 0.35
Y_T = -X_W
Z_T = Z_W - 0.74
```

Do NOT copy these numeric extrinsics to real hardware.

## PointCloud pipeline

```text
Depth -> Camera XYZ -> Camera -> Task transform -> Task-frame crop
-> candidate <= 8192 -> FPS 512 -> training-stat normalization -> PointCloud encoder
```

Legacy Task crop:

```text
X_T: [0.00, 0.40]
Y_T: [-0.25, 0.25]
Z_T: [0.005, 0.30]
```

Legacy v4 dataset min/max example:

```text
min = [ 0.0301993787, -0.1533667743, 0.0050008297 ]
max = [ 0.2908546031,  0.1525760889, 0.2638573647 ]
```

These are checkpoint/dataset-specific. Recompute/save statistics for every new training dataset and package them with the checkpoint.

## Coordinate-transform convention

Use one convention everywhere:

```text
T_A_B transforms coordinates expressed in B into coordinates expressed in A.
```

Thus:

```text
p_base = T_base_camera @ p_camera
p_task = T_task_base @ p_base
T_task_camera = T_task_base @ T_base_camera
```

Implement and test inverse round trips.

## Real depth conversion

Pinhole:

```text
x = (u - cx) * d / fx
y = (v - cy) * d / fy
z = d
```

Verify the depth scale before use. Never assume mm or m.

## Wooden-block dataset

Grid:
- X: 10 positions between 0.18 and 0.26 m.
- Y: `[-0.08, +0.08, -0.04, +0.04, 0.0]`.
- 50 cells.

Diagnostic search:
- 70 attempts,
- 50 successful episodes,
- 20 failed attempts.
- Success-only cells: 35.
- Mixed cells: 6.
- Failure-only cells: 9.
- At least one success: 41/50.

Failure-only indices: `0, 1, 5, 6, 8, 20, 39, 44, 49`.
Mixed: `2, 3, 13, 15, 17, 18`.

Important: successful HDF5 count is not necessarily unique-grid coverage. Reconstruct coverage from seed/grid/yaw logs.

## Return-to-Stow

Legacy target:

```text
[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]
[0.0, -1.519, 1.507, 0.0, 0.0]
```

Success concept:

```text
LiftSuccess AND ReturnSuccess AND ObjectHeld
```

## Marker dataset

Current marker-like object:
- horizontal cylinder,
- length 100 mm,
- diameter 18 mm,
- approximately 20 g,
- top-down short-axis pinch.

Smoke behavior verified:
- lift delta about 4.89 cm,
- lift success,
- joint-space return,
- hold success.

Collection plan:

```text
Dataset A: 50 XY x yaw [0,10,...,90] = 500 cases
Dataset B: 50 XY x yaw [100,110,...,170] = 400 cases
Combined: 50 XY x 18 yaw = 900 deterministic cases
```

A uniform cylinder has 180-degree axis symmetry, so 0 and 180 degrees are geometrically equivalent for the current grasp objective.

## Generalization evaluation

Spatial:
- evaluate between training-grid positions.

Orientation:
- train: 0,10,...,170 degrees,
- test: 5,15,...,175 degrees.

Geometry:
- train on controlled elongated primitives,
- test on unseen marker-like shapes such as different diameter/length, hexagonal pen, rounded rectangular marker, cap/body marker.

Do not claim "generalization" from training loss/validation loss alone. Use closed-loop task success on held-out positions/yaws/geometries.

## Temporal ensembling

One previous test showed:
- temporal ensemble ON: lift failed around 2.29 cm,
- newest raw prediction: crossed a 3 cm success threshold,
- newest-vs-executed difference reached about 0.117-0.123 rad in some joints.

Therefore temporal ensembling is an ablation, not a guaranteed improvement. Log newest raw action, ensemble action, and executed action.

## Real-robot critical checks

Before actuation, verify:
- joint names and index order,
- left/right order,
- sign,
- units,
- zero offsets,
- gripper representation,
- qpos normalization,
- action denormalization,
- RGB channel order,
- image resize/crop,
- PointCloud crop/FPS/stats,
- camera intrinsic,
- camera extrinsic,
- Task frame,
- timestamp skew,
- inference latency,
- command latency,
- control rate,
- joint position limits,
- velocity limits,
- stale-data watchdog,
- emergency stop.

## What is NOT yet verified

Do not state these as complete:
- real camera extrinsic,
- real Task origin,
- real control frequency,
- real sensor synchronization,
- real joint/action sign equivalence,
- real gripper scaling,
- Sim/Real RGB equivalence,
- Sim/Real PointCloud distribution equivalence,
- real temporal-ensemble benefit,
- real marker success rate,
- unseen-geometry success.

## Core principle

The difficult part of deploying DAPIER-ACT on hardware is not writing another neural network. It is reproducing the **same observation/action semantics** that the policy saw in training.
