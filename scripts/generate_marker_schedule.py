#!/usr/bin/env python3
import csv
from pathlib import Path
import numpy as np


def main():
    x = np.linspace(0.18, 0.26, 10)
    y = [-0.08, +0.08, -0.04, +0.04, 0.0]
    yaw = list(range(0, 180, 10))
    rows = []
    case = 0
    grid = 0
    for xi in x:
        for yi in y:
            for deg in yaw:
                rows.append({"case_index": case, "grid_index": grid, "task_x": float(xi), "task_y": float(yi), "yaw_deg": deg})
                case += 1
            grid += 1
    out = Path("marker_grid50_yaw18_schedule.csv")
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    assert len(rows) == 900
    print(f"Wrote {len(rows)} cases -> {out}")


if __name__ == "__main__":
    main()
