"""Profile timeout episodes (length >= 250) from TensorBoard per-episode scalars, last 2,000 per seed.

    python scripts/diagnose_timeouts/profile.py <env ids>
"""
import glob, sys
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
TAGS = ["charts/episodic_length", "task/door_angle", "task/valid_fraction", "task/wrong_contact_steps",
        "force/door_link4", "force/door_link5", "force/door_link6", "force/door_ee", "force/door_upper", "force/scene",
        "reward/progress", "reward/reach", "task/plate_tilt", "task/tilt_limit", "reward/force"]
for env in sys.argv[1:]:
    rows = []
    for run in sorted(glob.glob(f"runs/{env}__*")):
        e = EventAccumulator(run, size_guidance={"scalars": 0}); e.Reload()
        rows.append(np.stack([[x.value for x in e.Scalars(t)][-2000:] for t in TAGS], 1))
    d = dict(zip(TAGS, np.concatenate(rows).T))
    to = d["charts/episodic_length"] >= 250
    print(f"\n## {env}: {to.sum()} timeouts of {len(to)}")
    if not to.any(): continue
    ang = d["task/door_angle"][to]
    arm = d["force/door_link4"][to] + d["force/door_link5"][to] + d["force/door_link6"][to]
    wrong = d["force/door_ee"][to] + d["force/door_upper"][to] + d["force/scene"][to]
    print("final door angle (rad) quartiles", np.round(np.percentile(ang, [10, 25, 50, 75, 90]), 2))
    print("door barely moved (progress reward < 5, i.e. < 0.1 rad net)", np.mean(d["reward/progress"][to] < 5).round(3))
    print("median progress reward", np.median(d["reward/progress"][to]).round(1), "median reach reward", np.median(d["reward/reach"][to]).round(2))
    print("last step: forearm (links 4-6) on door >1N", np.mean(arm > 1).round(3), "| any wrong contact", np.mean(wrong > 0).round(3),
          "| scene contact", np.mean(d["force/scene"][to] > 0).round(3), "| ee on door", np.mean(d["force/door_ee"][to] > 0).round(3),
          "| upper links on door", np.mean(d["force/door_upper"][to] > 0).round(3))
    print("median forearm force at end (N), when >1N", np.median(arm[arm > 1]).round(1) if (arm > 1).any() else "-")
    print("wrong-contact steps per timeout ep: median", np.median(d["task/wrong_contact_steps"][to]), "| >0:", np.mean(d["task/wrong_contact_steps"][to] > 0).round(3))
    print("valid fraction median", np.median(d["task/valid_fraction"][to]).round(3))
    print("plate tilt at end median deg", np.degrees(np.median(d["task/plate_tilt"][to])).round(1))
    print("force-cap penalty paid (any)", np.mean(d["reward/force"][to] < 0).round(3))
