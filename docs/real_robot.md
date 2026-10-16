# Real Dual SO-101 Deployment

## Deployment phases

1. camera read only,
2. point-cloud visualization,
3. qpos read only,
4. offline recorded observation inference,
5. live dry-run inference,
6. one bounded command,
7. low-speed short horizon,
8. closed loop,
9. temporal-ensemble comparison.

A real runner should require an explicit `--enable-actuation` flag before commands are sent.

Each policy step should record sensor timestamps, qpos, PointCloud statistics, newest raw action, ensemble action if used, clamped action, actual command, policy latency and control-loop interval.

The policy is not a safety controller. Apply independent position, velocity and per-step limits, stale-data timeout, command watchdog and emergency stop.
