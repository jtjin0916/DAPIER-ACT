# Generalization Plan

"Generalized" must mean performance on held-out conditions, not simply a larger training set.

## Spatial
Train on deterministic 50-cell XY grid; evaluate on between-grid positions.

## Orientation
Training yaw: `0, 10, 20, ..., 170 deg`.
Evaluation yaw: `5, 15, 25, ..., 175 deg`.

For the uniform cylinder, 0 and 180 degrees are axis-symmetric.

## Geometry
Baseline: length 100 mm, diameter 18 mm cylinder.

Possible train diversification:
- length 90/100/110 mm,
- diameter 16/18/20 mm,
- circular/hexagonal/rounded-rectangular elongated primitives.

Held-out evaluation examples:
- marker with cap,
- different diameter,
- different length,
- hexagonal pen,
- rounded rectangular marker.

## Metrics
Offline: action MAE by dimension, horizon-wise MAE, raw-vs-ensemble difference, PointCloud distribution shift.

Closed loop: grasp success, lift success, lift height, hold through return, stow reached, full task success.

Report separately for seen XY/seen yaw, unseen XY, unseen yaw, and unseen geometry.
