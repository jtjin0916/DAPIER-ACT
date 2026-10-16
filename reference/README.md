# Working Custom ACT Reference

This directory contains behavior-preserving copies of the Custom DAPIER-ACT
implementation currently used in the RoboTwin development workspace.

The purpose of this directory is regression validation during the public refactor.

## Training-side active dependency chain

train_multimodal.py
 -> dataset.py
 -> dataset_multimodal.py
 -> pointcloud_preprocess.py
 -> model.py
 -> model_spatial.py
 -> model_multimodal.py

## Evaluation-side integration

XPolicyLab/policy/DAPIER_ACT/model.py
 -> XPolicyLab ModelTemplate / process_data
 -> dapier_act.model_multimodal
 -> dapier_act.pointcloud_preprocess

## Verification

The core copied files were verified against the working RoboTwin files using
SHA256 and were byte-for-byte identical at import time.

Legacy experiments, backup models and PTv3 experiments were intentionally
excluded from the public active reference because they are not dependencies of
the current multimodal Custom ACT path.

The reference implementation should not be refactored in-place.

Clean public modules should first reproduce:
1. preprocessing tensors,
2. encoded features,
3. raw 20x12 action chunks,

before the reference implementation is retired.
