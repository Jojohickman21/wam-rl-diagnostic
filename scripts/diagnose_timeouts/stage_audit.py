"""Timeout / Stage-1 audit of the staged scripted push (log.md 2026-11-27): where do non-spill failures happen?

Reads the per-step traces and outcomes written by scripted_baseline.py (staged mode, arg 8) for each task and seed,
and puts every timeout into one bucket (rules in log.md, first match wins, over the last W control steps):
  Stage 1 (never reached the standoff): handoff_miss, obstacle_block, clock_starved_approach, ik_stall
  Stage 2 (reached it, timed out): clock_starved_push, wedged, lost_contact, other
Per task it also reports the median time left when the standoff was reached (successes vs Stage-2 timeouts) and the
median progress of Stage-2 timeouts.

    python scripts/diagnose_timeouts/stage_audit.py <dir with <mode>_<task>_s<seed>.npy and .trace.npy> <task> [...] [--mode staged2]
"""
import glob
import sys

import numpy as np

W, LIMIT = 25, 250
STANDOFF_DONE, PUSHING, ERROR, SCENE, UPPER, L4, L5, L6, EE, DIST, MARGIN, LEAD = range(12)
STAGE1 = ("handoff_miss", "obstacle_block", "clock_starved_approach", "ik_stall")
STAGE2 = ("clock_starved_push", "wedged", "lost_contact", "other")


def bucket(t):
    last = t[-W:]
    forearm = last[:, L4] + last[:, L5] + last[:, L6]
    if not t[-1, STANDOFF_DONE]:
        touching = (last[:, [SCENE, UPPER, L4, L5, L6, EE]].sum(1) > 0).mean()
        closer = t[-W, ERROR] - t[-1, ERROR]
        if t[:, ERROR].min() < 0.06 and t[-1, ERROR] < 0.06:
            return "handoff_miss"
        if touching >= 0.3 and closer < 0.01:
            return "obstacle_block"
        if closer >= 0.01:
            return "clock_starved_approach"
        return "ik_stall"
    if t[-W, DIST] - t[-1, DIST] >= 0.02:
        return "clock_starved_push"
    if (forearm > 0).mean() >= 0.8:
        return "wedged"
    if (forearm > 0).mean() <= 0.2:
        return "lost_contact"
    return "other"


def main(directory, tasks, mode="staged"):
    for task in tasks:
        outcomes, traces = [], []
        for f in sorted(glob.glob(f"{directory}/{mode}_{task}_s?.npy")):
            outcomes.append(np.load(f))
            traces += list(np.load(f.replace(".npy", ".trace.npy"), allow_pickle=True))
        v = np.concatenate(outcomes)
        assert len(v) == len(traces)
        won, spill, timeout = v[:, 0] > 0, v[:, 2] > 0, v[:, 3] > 0
        labels = [bucket(t) if to else None for t, to in zip(traces, timeout)]
        n = len(v)
        print(f"\n{task}: {n} episodes | success {won.mean():.3f} | spill {spill.mean():.3f} | timeout {timeout.mean():.3f} "
              f"(stage 1 {np.mean([l in STAGE1 for l in labels]):.3f}, stage 2 {np.mean([l in STAGE2 for l in labels]):.3f})")
        for name in STAGE1 + STAGE2:
            idx = [i for i, l in enumerate(labels) if l == name]
            if not idx:
                continue
            extra = ""
            if name == "ik_stall":
                extra = f" | a joint within 0.05 rad of a stop: {np.mean([traces[i][-1, MARGIN] < 0.05 for i in idx]):.2f}"
            if name == "lost_contact":
                extra = f" | push phase never started: {np.mean([not traces[i][-1, PUSHING] for i in idx]):.2f}"
            if name in STAGE1:
                extra += f" | final distance to standoff median {np.median([traces[i][-1, ERROR] for i in idx]) * 100:.1f} cm" \
                         f" | target lead (j1-4) median {np.median([traces[i][-W:, LEAD].mean() for i in idx]) * 1000:.0f} mrad"
            print(f"  {name:24s} {len(idx):4d} ({len(idx) / n:.3f} of episodes){extra}")
        arrival = lambda t: LIMIT - int(np.argmax(t[:, STANDOFF_DONE] > 0))
        s2 = [i for i, l in enumerate(labels) if l in STAGE2]
        ok = [i for i in np.flatnonzero(won) if traces[i][-1, STANDOFF_DONE]]
        if ok:
            msg = f"  time left at standoff arrival, median steps: successes {np.median([arrival(traces[i]) for i in ok]):.0f}"
            if s2:
                msg += f", stage-2 timeouts {np.median([arrival(traces[i]) for i in s2]):.0f}" \
                       f" | stage-2 timeouts' progress median {np.median([1 - traces[i][-1, DIST] / max(traces[i][0, DIST], 1e-9) for i in s2]):.2f}" \
                       f" | successes' steps from standoff to done median {np.median([len(traces[i]) - (LIMIT - arrival(traces[i])) for i in ok]):.0f}"
            print(msg)


if __name__ == "__main__":
    args = sys.argv[2:]
    mode = args[args.index("--mode") + 1] if "--mode" in args else "staged"
    main(sys.argv[1], [a for a in args if a not in ("--mode", mode)], mode)
