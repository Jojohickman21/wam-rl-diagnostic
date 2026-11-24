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


def summarize_run(run):
    """Per-task final success and steps to 90%, and pooled final success, for one finished run.

    `task/success` and `task/index` are written together for every finished episode (several can
    share a training step), so the two series are paired by position.
    """
    events = EventAccumulator(str(run), size_guidance={"scalars": 0})
    events.Reload()
    success = np.array([e.value for e in events.Scalars("task/success")])
    steps = np.array([e.step for e in events.Scalars("task/success")])
    index = np.array([e.value for e in events.Scalars("task/index")]).astype(int)
    if len(index) != len(success):
        raise ValueError(f"{run}: {len(success)} task/success events but {len(index)} task/index events")
    finals, reached = {}, {}
    for k, task in enumerate(TASKS):
        mine = index == k
        finals[task] = float(success[mine][-500:].mean()) if mine.any() else None
        reached[task] = steps_to(success[mine], steps[mine])
    return dict(seed=run.name.split("__")[2], finals=finals, reached=reached, pooled=float(success[-2000:].mean()))


def report(env_id, runs=RUNS):
    """The markdown table for one env id: a row per finished seed, then the per-task mean and range.
    A task with no episodes in a run shows as "no episodes" and is left out of the mean."""
    lines = [f"\n## {env_id}",
             "| Seed | " + " | ".join(f"{t} final / steps to 90%" for t in TASKS) + " | Pooled final |",
             "|---|" + "---|" * (len(TASKS) + 1)]
    finals = {t: [] for t in TASKS}
    for run in sorted(Path(runs).glob(f"{env_id}__*"), key=lambda p: int(p.name.split("__")[2])):
        if not (run / "train_ppo.cleanrl_model").exists():
            continue
        row = summarize_run(run)
        cells = []
        for task in TASKS:
            final = row["finals"][task]
            if final is None:
                cells.append("no episodes")
                continue
            finals[task].append(final)
            cells.append(f"{final:.3f} / {row['reached'][task]}")
        lines.append(f"| {row['seed']} | " + " | ".join(cells) + f" | {row['pooled']:.3f} |")
    lines.append("| mean | " + " | ".join(
        f"{st.mean(finals[t]):.3f} (range {min(finals[t]):.3f}–{max(finals[t]):.3f})" if finals[t] else "no episodes"
        for t in TASKS) + " | |")
    return "\n".join(lines)


if __name__ == "__main__":
    for env_id in sys.argv[1:]:
        print(report(env_id))
