# Public Release Checklist

- [ ] Compare public frontend tensor shapes against actual Custom ACT source.
- [ ] Verify checkpoint key compatibility.
- [ ] Confirm exact 12D qpos/action order.
- [ ] Confirm gripper units/sign.
- [ ] Confirm training RGB preprocessing.
- [ ] Confirm RoboTwin HDF5 decode path.
- [ ] Export checkpoint-specific PointCloud/qpos/action stats.
- [ ] Run one frozen-reference observation through old and new code.
- [ ] Verify output equivalence within documented tolerance.
- [ ] Remove secrets, absolute local paths and tokens.
- [ ] Do not commit raw datasets/checkpoints accidentally.
- [ ] Audit third-party licenses.
- [ ] Add limitations/model card.
- [ ] Label unverified real-hardware assumptions.
- [ ] Run unit tests.
- [ ] Test installation from a fresh clone.
