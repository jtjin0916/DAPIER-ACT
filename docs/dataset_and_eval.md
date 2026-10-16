# Dataset and Evaluation

## Marker schedule

First collection: 50 XY x 10 yaw (0 to 90 deg) = 500 cases.
Second collection: 50 XY x 8 yaw (100 to 170 deg) = 400 cases.
Combined: 900 intended XY-yaw combinations.

Because the collector can skip failures and continue seeds, count intended unique combinations from logs/scene metadata instead of equating HDF5 count with coverage.

Recommended numeric episode metadata includes seed, grid_index, case_index, task_x, task_y, yaw_deg, active_arm, lift_success, return_success and hold_success.
