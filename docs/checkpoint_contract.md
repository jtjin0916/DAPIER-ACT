# Checkpoint Contract

A deployable model is a package, not a `.pt` file.

```text
checkpoint/
  best.pt
  model_config.yaml
  preprocessing.yaml
  normalization.npz
  metadata.json
```

Metadata should record git commit, training dataset version, train/validation episode IDs, split seed, epoch/best metric, PyTorch/CUDA versions, temporal-ensemble settings and model tensor schema.

Preprocessing should record Task-frame semantics, crop, point count, candidate count, image resolution, RGB preprocessing, qpos ordering and action ordering.

Never recompute normalization from a live real-robot episode.
