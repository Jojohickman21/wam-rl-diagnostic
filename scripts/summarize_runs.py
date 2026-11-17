"""Print per-run means over the last episodes of each TensorBoard run under runs/.

    python scripts/summarize_runs.py [--last 200]
"""
import argparse
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = (
    "charts/episodic_return", "task/success", "task/valid_fraction", "task/spill",
    "task/wrong_contact_steps", "task/door_angle", "charts/episodic_length",
    "reward/progress", "reward/reach", "reward/wrong", "reward/tilt", "reward/force",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--last", type=int, default=200, help="episodes to average at the end of each run")
    args = parser.parse_args()

    print("run," + ",".join(tag.split("/")[1] for tag in TAGS) + ",episodes,first_step_success_over_half")
    for run in sorted(p for p in (Path(__file__).parents[1] / "runs").iterdir() if p.is_dir()):
        events = EventAccumulator(str(run), size_guidance={"scalars": 0})
        events.Reload()
        means = [np.mean([e.value for e in events.Scalars(tag)[-args.last :]]) for tag in TAGS]
        success = events.Scalars("task/success")
        values = np.array([e.value for e in success])
        rolling = np.convolve(values, np.ones(args.last) / args.last, mode="valid")
        crossed = np.argmax(rolling > 0.5) if np.any(rolling > 0.5) else None
        first = success[crossed + args.last - 1].step if crossed is not None else "never"
        print(f"{run.name}," + ",".join(f"{m:.3f}" for m in means) + f",{len(success)},{first}")


if __name__ == "__main__":
    main()
