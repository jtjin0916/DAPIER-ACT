# Sim-to-Real Critical Validation

A failed grasp is not sufficient evidence of a neural-policy failure.

## Validation order

1. Geometry and frames: depth units, intrinsics, distortion, axes, extrinsic direction, multiplication order, Task origin.
2. PointCloud: visualize raw, Task transformed, crop box, cropped, FPS 512, normalized points.
3. RGB: RGB/BGR, resolution, resize/crop, value range and normalization.
4. Robot state: 12D order, signs, radians/degrees, offsets, gripper scaling, left/right ordering.
5. Action: semantic meaning, normalization/denormalization, absolute-vs-delta, frequency, clipping.
6. Time: sensor timestamps, inference time, command time.
7. Model: only after 1-6 pass, attribute bad actions to generalization/model error.

For RoboTwin HDF5 data, verify the current `decode_image_bit` behavior in the repository version in use. Do not add a second channel swap without evidence.

## Regression test

Save one reference observation and compare old training code, old eval code, and public/refactored code. Model-ready tensors should agree within a documented tolerance.
