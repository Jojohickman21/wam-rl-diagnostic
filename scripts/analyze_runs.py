"""Per-seed table and slide-angle bins for the tray envs, in the format RESULTS.md uses.

    python scripts/analyze_runs.py WamCloseCabinetTrayCoupledSeeRisk-v0 [more env ids]

Reads TensorBoard event files under runs/<env id>__*. Final metrics are means over each seed's
last 500 episodes; slide-angle bins pool the last 2,000 episodes of every seed.
"""
import sys
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = Path(__file__).parents[1] / "runs"
BINS = np.radians([15, 20, 25, 30, 35])


def scalars(events, tag):
    if tag not in events.Tags()["scalars"]:
        return None
    return np.array([e.value for e in events.Scalars(tag)])


def steps_to(success, steps, level, window=200):
    rolling = np.convolve(success, np.ones(window) / window, "valid")
    return f"{steps[np.argmax(rolling > level) + window - 1] // 1000}k" if np.any(rolling > level) else "never"


def report(env_id):
    print(f"\n## {env_id}")
    print("| Seed | Episodes | Success | Valid fraction | Spill | Timeout | Episode length | Steps to 50% / 90% | Mean coupling | Success 0-0.5M / 1.5-2M |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    pooled = []
    for run in sorted(RUNS.glob(f"{env_id}__*")):
        events = EventAccumulator(str(run), size_guidance={"scalars": 0})
        events.Reload()
        success = scalars(events, "task/success")
        steps = np.array([e.step for e in events.Scalars("task/success")])
        valid, spill, length, limit, tilt = (
            scalars(events, t) for t in
            ("task/valid_fraction", "task/spill", "charts/episodic_length", "task/tilt_limit", "task/plate_tilt")
        )
        coupling = scalars(events, "task/coupling_mean")
        last = slice(-500, None)
        early, late = success[steps < 500_000].mean(), success[steps >= 1_500_000].mean()
        print(
            f"| {run.name.split('__')[2]} | {len(success)} | {success[last].mean():.3f} | {valid[last].mean():.3f} "
            f"| {spill[last].mean():.3f} | {(length[last] >= 250).mean():.3f} | {length[last].mean():.0f} "
            f"| {steps_to(success, steps, 0.5)} / {steps_to(success, steps, 0.9)} "
            f"| {coupling[last].mean():.2f} | {early:.2f} / {late:.2f} |"
            if coupling is not None else
            f"| {run.name.split('__')[2]} | {len(success)} | {success[last].mean():.3f} | {valid[last].mean():.3f} "
            f"| {spill[last].mean():.3f} | {(length[last] >= 250).mean():.3f} | {length[last].mean():.0f} "
            f"| {steps_to(success, steps, 0.5)} / {steps_to(success, steps, 0.9)} | n/a | {early:.2f} / {late:.2f} |"
        )
        tail = slice(-2000, None)
        pooled.append((success[tail], spill[tail], length[tail], limit[tail], tilt[tail]))
    if not pooled:
        print("no runs found")
        return
    success, spill, length, limit, tilt = (np.concatenate(x) for x in zip(*pooled))
    bins = np.digitize(limit, BINS) - 1
    print("\n| Slide angle | 15–20° | 20–25° | 25–30° | 30–35° |\n|---|---|---|---|---|")
    for name, values in (("Success", success), ("Spill", spill), ("Timeout", (length >= 250).astype(float))):
        print(f"| {name} | " + " | ".join(f"{values[bins == i].mean():.3f}" for i in range(4)) + " |")
    print("| Final plate tilt in successful episodes, median | " + " | ".join(
        f"{np.degrees(np.median(tilt[(bins == i) & (success > 0)])):.1f}°" for i in range(4)) + " |")
    print("| Episodes | " + " | ".join(str(int((bins == i).sum())) for i in range(4)) + " |")


if __name__ == "__main__":
    for env_id in sys.argv[1:]:
        report(env_id)
