# PointCloud Preprocessing Contract

```text
depth -> metric depth -> camera XYZ -> Task-frame transform -> crop
-> candidate reduction -> FPS -> checkpoint-specific normalization -> model
```

Legacy crop:

```yaml
x: [0.00, 0.40]
y: [-0.25, 0.25]
z: [0.005, 0.30]
```

Legacy parameters:
- candidates <= 8192,
- model points = 512.

Crop must happen in the same coordinate frame used during training.

For train and real observations, log point count before/after crop, XYZ min/max, XYZ mean/std, normalization out-of-range ratio, and NaN/Inf count.
