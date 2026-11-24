"""Training curves of the multi-task policies (RESULTS.md §10.4), split by training task.

analyze_runs.py pools every episode; a multi-task run interleaves four tasks, so this splits each
seed's episodes by the logged `task/index` (order: T0, T1, T4x, T4xneg) and reports, per task, the
success over that task's last 500 episodes and the training step at which its 200-episode rolling
success first exceeded 0.9.

    python scripts/analyze_multitask.py WamMultiTask-v0 WamMultiTaskLabel-v0
"""
import statistics as st
import sys
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = Path(__file__).parents[1] / "runs"
TASKS = ("T0", "T1", "T4x", "T4xneg")


def steps_to(success, steps, level=0.9, window=200):
    if len(success) < window:
        return "never"
    rolling = np.convolve(success, np.ones(window) / window, "valid")
    hit = np.flatnonzero(rolling > level)
    return f"{steps[hit[0] + window - 1] // 1000}k" if len(hit) else "never"


for env_id in sys.argv[1:]:
    print(f"\n## {env_id}")
    print("| Seed | " + " | ".join(f"{t} final / steps to 90%" for t in TASKS) + " | Pooled final |")
    print("|---|" + "---|" * (len(TASKS) + 1))
    finals = {t: [] for t in TASKS}
    for run in sorted(RUNS.glob(f"{env_id}__*"), key=lambda p: int(p.name.split("__")[2])):
        if not (run / "train_ppo.cleanrl_model").exists():
            continue
        events = EventAccumulator(str(run), size_guidance={"scalars": 0})
        events.Reload()
        success = np.array([e.value for e in events.Scalars("task/success")])
        steps = np.array([e.step for e in events.Scalars("task/success")])
        index = np.array([e.value for e in events.Scalars("task/index")]).astype(int)
        cells = []
        for k, task in enumerate(TASKS):
            mine = index == k
            final = success[mine][-500:].mean()
            finals[task].append(final)
            cells.append(f"{final:.3f} / {steps_to(success[mine], steps[mine])}")
        print(f"| {run.name.split('__')[2]} | " + " | ".join(cells) + f" | {success[-2000:].mean():.3f} |")
    print("| mean | " + " | ".join(f"{st.mean(finals[t]):.3f} (range {min(finals[t]):.3f}–{max(finals[t]):.3f})"
                                   for t in TASKS) + " | |")
