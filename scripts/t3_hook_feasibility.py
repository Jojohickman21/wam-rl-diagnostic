"""T3 (drawer pull) feasibility with orientation: align the forearm with the handle, drop into the
handle gap from above, then draw the drawer toward the robot.

task_feasibility.py moves one forearm point and cannot turn the forearm, so it never fit the
forearm (about 8.5 cm thick) into the 10 cm gap behind T3's handle. Here damped least squares on
joints 1-4 tracks the point's position and, with weight W_ROT, turns link 5's z axis (the forearm
tube's axis) parallel to the handle. Phases: above the gap (lift), into the gap, pull along the
drawer's open direction. Same success rule as the gate: solved if valid fraction > 0.9.

    python scripts/t3_hook_feasibility.py [n_starts]
"""
import sys

import gymnasium as gym
import mujoco
import numpy as np

sys.path.insert(0, ".")
import wam  # noqa: F401,E402
from wam.close_cabinet import ACTION_SCALE  # noqa: E402

GAIN, DAMPING, W_ROT, LIFT, PULL_STEP, REACHED, ALIGNED = 0.5, 1e-3, 0.3, 0.15, 0.008, 0.015, np.radians(15)
POINTS = ("link4_c3", "link5_c1", "link5_c2", "link5_c3")


def step_action(u, point, target, handle_axis):
    geom = u.model.geom(point).id
    body = u.model.geom_bodyid[geom]
    jp, jr = np.zeros((3, u.model.nv)), np.zeros((3, u.model.nv))
    mujoco.mj_jac(u.model, u.data, jp, jr, u.data.geom_xpos[geom], body)
    fore = u.data.xmat[u.model.body("link5").id].reshape(3, 3)[:, 2]
    h = handle_axis if fore @ handle_axis >= 0 else -handle_axis
    rot_err = np.cross(fore, h)
    J = np.vstack([jp[:, :4], W_ROT * jr[:, :4]])
    err = np.concatenate([GAIN * (target - u.data.geom_xpos[geom]), W_ROT * GAIN * rot_err])
    dq = J.T @ np.linalg.solve(J @ J.T + DAMPING * np.eye(6), err)
    angle = np.arcsin(min(np.linalg.norm(rot_err), 1.0))
    return np.clip(dq / ACTION_SCALE, -1, 1).astype(np.float32), float(np.linalg.norm(target - u.data.geom_xpos[geom])), angle


def run(seed, point, max_steps=450):
    env = gym.wrappers.TimeLimit(gym.make("WamTaskT3-v0").unwrapped, max_steps)
    u = env.unwrapped
    env.reset(seed=seed)
    phase, best = 0, 0
    for _ in range(max_steps):
        rot = u.data.xmat[u.mover_body].reshape(3, 3)
        out_dir, handle_axis = rot[:, 0], rot[:, 1]  # drawer opens along local x; handle runs along local y
        handle = u.data.geom_xpos[u.model.geom("handle").id]
        inside = handle - (0.04 + 0.012 + 0.01) * out_dir  # in the gap, against the handle's inner side
        if phase == 0:
            target = inside + np.array([0, 0, LIFT])
        elif phase == 1:
            target = inside
        else:
            target = u.data.geom_xpos[u.model.geom(point).id] + PULL_STEP / GAIN * out_dir
        action, error, angle = step_action(u, point, target, handle_axis)
        if phase < 2 and error < REACHED and angle < ALIGNED:
            phase += 1
        best = max(best, phase)
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return info["task/success"] > 0 and info["task/valid_fraction"] > 0.9, best, info["task/distance_norm"]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    solved, phases, dist = [], [], []
    for seed in range(n):
        runs = [run(seed, p) for p in POINTS]
        solved.append(any(r[0] for r in runs))
        phases.append(max(r[1] for r in runs))
        dist.append(min(r[2] for r in runs))
    print(f"T3 hook: n={n} solved {np.mean(solved):.2f} | reached gap {np.mean(np.array(phases) >= 2):.2f} "
          f"| reached above-gap {np.mean(np.array(phases) >= 1):.2f} | best final distance (normalised) median {np.median(dist):.2f} "
          f"| gate (>= 0.30): {'PASS' if np.mean(solved) >= 0.30 else 'FAIL'}")
