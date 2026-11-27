"""Success on held-out T2 by starting height: the table behind Josiah's 2026-11-27 observation (log.md).

Rebuilds every eval start (reset seed 50_000 + seed, then the env's own chain; starts do not depend on
actions) and records the height of the forearm point (link5_c2 centre) above the contact target, then
joins it with the outcomes from t2_failures.py. Writes runs/diagnostics/t2_failures/start_heights.csv
(one row per policy, seed, episode) and prints success and chest-contact share per height bin.

    python scripts/diagnose_timeouts/t2_start_heights.py
"""
import csv
import sys

import numpy as np

sys.path.insert(0, ".")
from wam.tasks import TaskEnv  # noqa: E402

DIR = "runs/diagnostics/t2_failures"
BINS = [(-9, -0.10), (-0.10, 0.0), (0.0, 0.10), (0.10, 0.20), (0.20, 9)]
GROUPS = {"G3 strong (1,3,5)": [f"G3_s{s}" for s in (1, 3, 5)],
          "G3 weak (2,4,6,7,8)": [f"G3_s{s}" for s in (2, 4, 6, 7, 8)],
          "scripted": [f"scripted_s{s}" for s in range(1, 9)]}

u = TaskEnv(task="T2", cabinet_spread=(0.1, np.radians(20.0)), conditioning="contact")
point = u.model.geom("link5_c2").id
heights = {}
for seed in range(1, 9):
    u.reset(seed=50_000 + seed)
    rows = []
    for _ in range(200):
        target, _ = u.contact_target()
        offset = u.data.geom_xpos[point] - target
        rows.append((offset[2], np.linalg.norm(offset[:2])))
        u.reset()
    heights[seed] = rows

with open(f"{DIR}/start_heights.csv", "w", newline="") as f:
    out = csv.writer(f)
    out.writerow(["run", "seed", "episode", "height_above_target_m", "horizontal_distance_m", "label", "chest_touched"])
    table = {}
    for group, runs in GROUPS.items():
        z, ok, chest = [], [], []
        for run in runs:
            seed = int(run.split("_s")[1])
            for k, e in enumerate(np.load(f"{DIR}/{run}.npy", allow_pickle=True)):
                dz, dxy = heights[seed][k]
                out.writerow([run, seed, k, f"{dz:.4f}", f"{dxy:.4f}", e["label"], int(e["chest"].any())])
                z.append(dz), ok.append(e["success"]), chest.append(e["chest"].any())
        table[group] = tuple(map(np.array, (z, ok, chest)))

print("| Start height vs target | " + " | ".join(table) + " |\n|---|" + "---|" * len(table))
for lo, hi in BINS:
    cells = []
    for z, ok, chest in table.values():
        m = (z >= lo) & (z < hi)
        cells.append(f"{ok[m].mean():.2f} (chest {chest[m].mean():.0%}, n {m.sum()})")
    print(f"| {lo:+.2f} to {hi:+.2f} m | " + " | ".join(cells) + " |")
