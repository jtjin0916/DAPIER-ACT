#!/usr/bin/env python3

from pathlib import Path
import textwrap


path = Path("README.md")
text = path.read_text(encoding="utf-8")

marker = "## Wrist RGB contract"

section = r"""
## Wrist RGB contract

RGB decoding is now explicit because an audit found a reproducible channel
mismatch between the historical DAPIER HDF5 loader and the current local
XPolicyLab runtime decoder.

Both the v4 wooden-block corpus and the marker corpus satisfy:

```text
historical loader vs XPolicyLab decoder
exactly equal: false
R/B swap equal: true
```

New/generalized training therefore defaults to:

```text
--rgb-decode-mode xpolicylab_rgb
```

Historical v4 training tensors can be reproduced explicitly with:

```text
--rgb-decode-mode historical_bgr2rgb
```

New checkpoints store the selected RGB decode contract in metadata. The
frozen `reference/` implementation is intentionally unchanged.
"""

if marker not in text:
    text = text.rstrip() + "\n\n" + textwrap.dedent(section).strip() + "\n"
    path.write_text(text, encoding="utf-8")
    print("README RGB contract section added.")
else:
    print("README already contains RGB contract section.")
