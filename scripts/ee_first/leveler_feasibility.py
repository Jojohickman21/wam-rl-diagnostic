"""Forearm-only feasibility on end-effector-first starts in the leveler env (A24 rebuilt).

Drives joints 1-4 to home, holds up to 150 steps while the leveler's tuck swings the wrist back,
then runs the standard base-yaw sweep. Prints success, forearm-only success (valid fraction > 0.9)
and wrong-contact share by start type.

    python scripts/ee_first/leveler_feasibility.py [env id]
"""
import sys

import gymnasium as gym
import numpy as np

sys.path.insert(0, ".")  # run from the repo root
import wam  # noqa: F401,E402
from wam.checks import SETTLE_STEPS, SWEEP_RATE, SWEEP_STEPS  # noqa: E402
from wam.close_cabinet import ACTION_SCALE, FRAME_SKIP  # noqa: E402

env = gym.wrappers.TimeLimit(gym.make(sys.argv[1] if len(sys.argv) > 1 else "WamCloseCabinetEEFirstLeveler-v0").unwrapped, 500)
u = env.unwrapped
sweep = SWEEP_RATE * FRAME_SKIP * u.model.opt.timestep / ACTION_SCALE
res = {True: [], False: []}
for seed in range(60):
    _, info = env.reset(seed=seed)
    ee = bool(info["task/ee_first"])
    steps = hold = 0
    done = False
    while not done:
        d = u.home[:4] - u.q_target[:4]
        if steps == 0 and (np.abs(d).max() > 1e-3 or (hold < 150 and np.abs(u.home[4:7] - u.data.qpos[4:7]).max() > 0.1)):
            action = np.clip(d / ACTION_SCALE, -1, 1)
            hold += np.abs(d).max() <= 1e-3
        else:
            action = np.zeros(4)
            action[0] = sweep if steps < SWEEP_STEPS else 0.0
            steps += 1
        _, _, terminated, truncated, info = env.step(action.astype(np.float32))
        done = terminated or truncated or steps >= SWEEP_STEPS + SETTLE_STEPS
    res[ee].append((info["task/success"], info["task/valid_fraction"], info["task/wrong_contact_steps"] > 0))
for ee, rows in res.items():
    v = np.array(rows, float)
    ok = v[:, 0] > 0
    print(f"ee_first={ee}: n={len(v)} success {ok.mean():.2f} | forearm-only success {np.mean(ok & (v[:, 1] > 0.9)):.2f} | any wrong contact {v[:, 2].mean():.2f}")
