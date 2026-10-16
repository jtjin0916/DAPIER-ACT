# Wrist RGB contract

The public runtime now distinguishes two wrist-image decode contracts.

## Evidence

The v4 wooden-block corpus and the marker corpus were both audited against
the local XPolicyLab `decode_image_bit` implementation.

In both corpora:

```text
XPL-RGB1 marker: false

historical DAPIER loader
vs
XPolicyLab decoder

exactly equal: false
R/B swap equal: true
```

Multiple v4 timesteps reproduced the same relationship.

The local XPolicyLab policy server calls `decode_obs_images()` before the
policy adapter. Its `decode_image_bit()` contract treats the result of
`cv2.imdecode()` as the recovered RGB array for this legacy corpus and does
not apply `COLOR_BGR2RGB`.

The historical DAPIER training loader did apply `COLOR_BGR2RGB`, so its
wrist tensors were red/blue swapped relative to current XPolicyLab runtime
RGB semantics.

## Supported modes

### `historical_bgr2rgb`

```text
HDF5 bytes
-> cv2.imdecode
-> COLOR_BGR2RGB
-> float [0,1]
```

This mode exists only to reproduce the wrist tensors used by the historical
v4 training loader.

### `xpolicylab_rgb`

```text
HDF5 bytes
-> cv2.imdecode
-> no additional channel swap
-> float [0,1]
```

This matches the audited local XPolicyLab legacy-corpus decoder and is the
default for new/generalized training.

## Checkpoint metadata

New checkpoints record:

```text
rgb_decode_mode
rgb_channel_order = RGB
rgb_range = [0.0, 1.0]
rgb_resize = None
rgb_contract_version = 1
```

Do not infer the RGB contract from an old checkpoint that lacks this
metadata.

## Existing v4 checkpoint

The v4 model structure remains strict-load compatible. The RGB issue is an
observation-contract issue, not a state-dict issue.

For closed-loop diagnosis, keep the historical reference unchanged and
compare:

```text
A. current XPolicyLab runtime RGB
B. runtime RGB with R/B swapped to reproduce v4 training tensors
```

This A/B test is required before attributing any prior manipulation failure
to the RGB mismatch.
