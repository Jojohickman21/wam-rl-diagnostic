"""Feasibility of the procedural slide family (task "F") by push direction.

For one arc of push azimuths [lo, hi] degrees with a given home_yaw, runs the scripted forearm push
of task_feasibility.py (any of four forearm points, wrist leveler holding the plate) on N starts and
prints the share solved forearm-only (valid fraction > 0.9). Used to choose the training arcs and
the held-out band before any training.

    python scripts/family_feasibility.py <lo deg> <hi deg> <home_yaw rad> [n]
"""
import sys

import gymnasium as gym
import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import task_feasibility as tf  # noqa: E402
import wam  # noqa: F401,E402
from wam.tasks import TaskEnv  # noqa: E402

lo, hi, home_yaw = np.radians(float(sys.argv[1])), np.radians(float(sys.argv[2])), float(sys.argv[3])
n = int(sys.argv[4]) if len(sys.argv) > 4 else 20
OFFSET = 0.04 + 0.01  # capsule radius + margin; the contact target is already on the surface


def run(seed, point, steps=450):
    u = TaskEnv(task="F", family=dict(arcs=((lo, hi, home_yaw),)))
    env = gym.wrappers.TimeLimit(u, steps)
    env.reset(seed=seed)
    phase = 1
    for _ in range(steps):
        target, direction = u.contact_target()
        if phase == 1:
            goal = target - OFFSET * direction
        else:
            goal = u.data.geom_xpos[u.model.geom(point).id] + tf.PUSH_STEP / tf.GAIN * direction
        action, error = tf.drive(u, goal, point)
        if phase == 1 and error < tf.REACHED:
            phase = 2
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return info["task/success"] > 0 and info["task/valid_fraction"] > 0.9


solved = [any(run(seed, p) for p in tf.FOREARM_POINTS) for seed in range(n)]
print(f"azimuth {sys.argv[1]}..{sys.argv[2]} deg, home_yaw {home_yaw}: solved forearm-only {np.mean(solved):.2f} (n={n})", flush=True)
