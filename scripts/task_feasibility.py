"""Feasibility gate for the task set (TASK_SET.md): can a scripted forearm push solve the task?

A point on the forearm (one of FOREARM_POINTS) is driven by damped least squares on
joints 1-4, the wrist leveler holding the plate as in training, through three phases:
  1. approach: to `lift` metres above the pre-contact point (skipped when lift is 0);
  2. place: to the pre-contact point, `offset` metres behind the push point along the push
     direction, i.e. on the side the forearm has to push from;
  3. push: along the push direction, the direction the push point moves when the object moves
     toward its goal, recomputed every step.
Reports, over N starts: success, forearm-only success (valid fraction > 0.9), spill, and the share
of episodes where the end-effector touched the object. Each start is tried with every forearm
point; it counts as solved if any of them closes it forearm-only. Gate: >= 30% of starts solved.

    python scripts/task_feasibility.py T1 [n_starts]
"""
import sys

import gymnasium as gym
import mujoco
import numpy as np

sys.path.insert(0, ".")  # run from the repo root
import wam  # noqa: F401,E402
from wam.close_cabinet import ACTION_SCALE  # noqa: E402

# Per task: push-point offset from the forearm point's centre (capsule radius 0.04 + the object's
# half-extent along the push + 1 cm), and how high to come in from (T3 enters the handle gap from above).
SCRIPT = {
    "T1": dict(offset=0.04 + 0.015 + 0.01, lift=0.0, point="push_point"),
    "T2": dict(offset=0.04 + 0.01 + 0.01, lift=0.0, point="push_point"),
    "T3": dict(offset=0.04 + 0.012 + 0.01, lift=0.15, point="handle"),
    "T4x": dict(offset=0.04 + 0.05 + 0.01, lift=0.0, point="push_point"),
    "T4y": dict(offset=0.04 + 0.05 + 0.01, lift=0.0, point="push_point"),
    "T4xy": dict(offset=0.04 + 0.07 + 0.01, lift=0.0, point="push_point"),
}
GAIN, DAMPING, PUSH_STEP, REACHED = 0.5, 1e-3, 0.008, 0.015
FOREARM_POINTS = ("link4_c3", "link5_c1", "link5_c2", "link5_c3")  # elbow end to wrist end


def push_direction(u):
    """World direction the mover's origin moves when the object moves toward the goal."""
    s = u.data.qpos[u.qpos_adr]
    toward = u.goal - s
    world = np.zeros(3)
    for i, joint in enumerate(u.joint_ids):
        world += toward[i] * u.data.xaxis[joint]  # slide joints only: hinge tasks are not scripted here
    return world / max(np.linalg.norm(world), 1e-9)


def point_of(u, name):
    if name == "handle":
        return u.data.geom_xpos[u.model.geom("handle").id].copy()
    return u.data.site_xpos[u.model.site(name).id].copy()


def drive(u, target, geom_name):
    """One DLS step moving the forearm point toward `target`; returns the 4-dim action."""
    geom = u.model.geom(geom_name).id
    point = u.data.geom_xpos[geom]
    jac = np.zeros((3, u.model.nv))
    mujoco.mj_jac(u.model, u.data, jac, None, point, u.model.geom_bodyid[geom])
    j4 = jac[:, :4]
    dq = j4.T @ np.linalg.solve(j4 @ j4.T + DAMPING * np.eye(3), GAIN * (target - point))
    return np.clip(dq / ACTION_SCALE, -1.0, 1.0).astype(np.float32), float(np.linalg.norm(target - point))


def run(task, seed, geom_name="link5_c2", max_steps=400):
    env = gym.wrappers.TimeLimit(gym.make(f"WamTask{task}-v0").unwrapped, max_steps)
    u = env.unwrapped
    cfg = SCRIPT[task]
    env.reset(seed=seed)
    phase = 0 if cfg["lift"] > 0 else 1
    ee_touch = False
    for _ in range(max_steps):
        direction = push_direction(u)
        pre = point_of(u, cfg["point"]) - cfg["offset"] * direction
        if phase == 0:
            target = pre + np.array([0.0, 0.0, cfg["lift"]])
        elif phase == 1:
            target = pre
        else:
            target = u.data.geom_xpos[u.model.geom(geom_name).id] + PUSH_STEP / GAIN * direction
        action, error = drive(u, target, geom_name)
        if phase < 2 and error < REACHED:
            phase += 1
        _, _, terminated, truncated, info = env.step(action)
        ee_touch |= info["force/door_ee"] > 0
        if terminated or truncated:
            break
    return dict(success=info["task/success"] > 0, spill=info["task/spill"] > 0,
                valid=info["task/valid_fraction"], ee=ee_touch, phase=phase,
                distance_norm=info["task/distance_norm"])


if __name__ == "__main__":
    task = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    solved, by_point = [], {g: [] for g in FOREARM_POINTS}
    for seed in range(n):
        rows = {g: run(task, seed, g) for g in FOREARM_POINTS}
        for g, r in rows.items():
            by_point[g].append(r)
        solved.append(any(r["success"] and r["valid"] > 0.9 for r in rows.values()))
    for g, rows in by_point.items():
        ok = np.array([r["success"] for r in rows])
        print(f"  {task} via {g}: success {ok.mean():.2f} | forearm-only {np.mean(ok & np.array([r['valid'] > 0.9 for r in rows])):.2f} "
              f"| spill {np.mean([r['spill'] for r in rows]):.2f} | hand touched {np.mean([r['ee'] for r in rows]):.2f} "
              f"| reached push phase {np.mean([r['phase'] == 2 for r in rows]):.2f}")
    print(f"{task}: n={n} starts solved forearm-only by some point {np.mean(solved):.2f} "
          f"| gate (>= 0.30): {'PASS' if np.mean(solved) >= 0.30 else 'FAIL'}")
