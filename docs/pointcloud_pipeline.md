# PointCloud Preprocessing Contract

Active DP3-head order:

```text
depth
-> metric camera XYZ
-> Camera -> Task transform
-> Task-frame crop
-> deterministic candidate reduction <=8192
-> FPS 512 in metric meters
-> checkpoint train point_min / point_max
-> 2 * (xyz-min)/(max-min) - 1
-> model
```

Simulation Task-frame crop:

```yaml
x: [0.00, 0.40]
y: [-0.25, 0.25]
z: [0.005, 0.30]
```

Normalization is **not clipped**. Therefore validation or real points outside
the train limits can legitimately become values outside `[-1, 1]`.

For train/eval/real observations, log:

- points before and after crop,
- sampled XYZ min/max,
- normalized XYZ min/max,
- fraction outside the trained `[-1,1]` range,
- NaN/Inf count,
- transform/calibration version.

FPS must happen before normalization because the active policy samples using
metric-space Euclidean geometry.
