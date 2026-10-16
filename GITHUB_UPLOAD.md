# GitHub Publication Plan

Recommended repository name: `DAPIER-ACT`

## Why a separate repository

Keep the public policy/refactor package independent from the larger DAPIER project so users can understand:
- model contract,
- preprocessing,
- simulation adapter,
- real-robot adapter,
- validation protocol
without unrelated project files.

## First public release

Tag suggestion: `v0.1.0-scaffold`

The first release should explicitly say:
- simulation/refactor scaffold,
- real hardware deployment not yet validated end-to-end,
- no generalized checkpoint is included yet,
- generalization claims require held-out closed-loop evaluation.

## Do not upload yet

Do not publish the following until audited:
- raw HDF5 datasets,
- current private checkpoints,
- third-party assets/meshes/URDFs,
- SDK binaries,
- credentials/tokens,
- absolute local paths.

## Suggested first commits

1. `Initial public DAPIER-ACT scaffold`
2. `Add verified legacy ACT backend adapter` after regression testing
3. `Add marker-900 dataset metadata and evaluation protocol`
4. `Add real SO-101 dry-run adapter` after hardware validation
5. `Release generalized checkpoint` only after held-out evaluation
