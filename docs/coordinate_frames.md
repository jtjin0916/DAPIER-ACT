# Coordinate Frames

Use:

```text
T_A_B = transform that maps coordinates in frame B into frame A.
```

For homogeneous points: `p_A = T_A_B @ p_B`.

## Real pipeline

```text
camera -> base -> task
T_task_camera = T_task_base @ T_base_camera
```

Task semantics should remain stable across sim and real:
- +X forward,
- +Y left,
- +Z up.

Do not reuse simulator numeric transforms on hardware.

## Calibration requirements

Store camera intrinsics, depth scale, distortion policy, `T_base_camera`, and `T_task_base` or equivalent.

## Mandatory verification

1. Matrix inverse test.
2. Known-point test.
3. Axis-direction test.
4. Table-plane test.
5. PointCloud overlay in Task frame.

Do not accept "looks roughly right" as the only validation.
