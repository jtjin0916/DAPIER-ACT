#!/usr/bin/env python3

from pathlib import Path


path = Path("README.md")
text = path.read_text(encoding="utf-8")

replacements = {
    "- [ ] verified ACT/CVAE backend migrated from the working RoboTwin code":
        "- [x] verified working ACT/CVAE runtime migrated from the active RoboTwin reference",
    "- [ ] public training entrypoint verified end-to-end":
        "- [ ] public training entrypoint verified end-to-end",
    "- [ ] public RoboTwin evaluation entrypoint verified end-to-end":
        "- [ ] public RoboTwin evaluation entrypoint verified end-to-end",
}

for old, new in replacements.items():
    text = text.replace(old, new)

section = """
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
"""

if "## Verified public Custom ACT runtime" not in text:
    text = text.rstrip() + "\n\n" + section.strip() + "\n"

path.write_text(text, encoding="utf-8")
print("README updated.")
