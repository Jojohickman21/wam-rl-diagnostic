"""Per-seed results for the end-effector-first envs, split by start type (`task/ee_first`).

    python scripts/analyze_ee_first.py WamCloseCabinetEEFirst-v0 WamCloseCabinetEEFirstUngated-v0

Only runs whose log has all 976 "SPS:" lines are used; unfinished runs are listed and skipped.
Means over each seed's last 1,000 episodes (about 500 of each start type). "Closed mostly by the
end-effector or other wrong contact" = a successful episode with valid fraction below 0.5.
"""
import sys
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = Path(__file__).parents[1] / "runs"
FULL = 976


def report(env_id):
    print(f"\n## {env_id}")
    print("| Seed | Start type | Episodes | Success | Valid fraction (successes) | Closed mostly by wrong contact (share of successes) | Spill | Timeout | Episodes with any wrong contact | Steps to 90% success (all starts) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for run in sorted(RUNS.glob(f"{env_id}__*")):
        seed = run.name.split("__")[2]
        log = RUNS / f"{env_id}-seed{seed}.log"
        lines = log.read_text().count("SPS:") if log.exists() else 0
        if lines < FULL:
            print(f"| {seed} | unfinished ({lines} of {FULL} updates), skipped | | | | | | | | |")
            continue
        events = EventAccumulator(str(run), size_guidance={"scalars": 0})
        events.Reload()
        get = lambda tag: np.array([e.value for e in events.Scalars(tag)])
        success, steps = get("task/success"), np.array([e.step for e in events.Scalars("task/success")])
        rolling = np.convolve(success, np.ones(200) / 200, "valid")
        to90 = f"{steps[np.argmax(rolling > 0.9) + 199] // 1000}k" if np.any(rolling > 0.9) else "never"
        last = slice(-1000, None)
        ee, valid, spill = get("task/ee_first")[last] > 0, get("task/valid_fraction")[last], get("task/spill")[last]
        length, wrong, ok = get("charts/episodic_length")[last], get("task/wrong_contact_steps")[last], success[last] > 0
        for name, sel in (("end-effector-first", ee), ("normal", ~ee)):
            won = sel & ok
            print(f"| {seed} | {name} | {sel.sum()} | {ok[sel].mean():.3f} | {valid[won].mean():.3f} | {np.mean(valid[won] < 0.5):.3f} "
                  f"| {spill[sel].mean():.3f} | {np.mean(length[sel] >= 250):.3f} | {np.mean(wrong[sel] > 0):.3f} | {to90} |")


if __name__ == "__main__":
    for env_id in sys.argv[1:]:
        report(env_id)
