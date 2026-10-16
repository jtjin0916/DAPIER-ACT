# Architecture

## Public contract

```text
Observation:
  head_points      float32 [B, 512, 3]
  left_wrist_rgb   float32 [B, 3, H, W]
  right_wrist_rgb  float32 [B, 3, H, W]
  qpos             float32 [B, 12]

Output:
  action_chunk     float32 [B, 20, 12]
```

The public scaffold deliberately separates the multimodal frontend from the exact historical ACT/CVAE backend.

## Frontend

Point branch: DP3-inspired shared point MLP `3 -> 64 -> 128 -> 256`, followed by global max pooling to produce one 256D token.

RGB branches: two independent ResNet18 backbones, layer4 features projected from 512 to 256 channels with a 1x1 convolution and flattened to spatial tokens.

For 240x320 images, a standard ResNet18 layer4 output is commonly 8x10, giving 80 tokens per wrist. Verify this against the actual preprocessing/backbone implementation.

## Backend

The historical project uses an ACT/CVAE transformer backend. Do not silently replace it with a generic transformer and call it checkpoint-compatible.

Before connecting old weights, verify module names, tensor shapes, positional encoding, qpos injection, latent-z handling, action normalization, and checkpoint keys.
