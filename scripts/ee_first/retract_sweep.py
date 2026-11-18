"""Forearm-only feasibility on end-effector-first starts: swing base yaw away, go home, sweep.
Wrist levelled by the A20 controller during the script (feasibility check only)."""
import sys
import numpy as np, gymnasium as gym
sys.path.insert(0, ".")
from wam.close_cabinet import CloseCabinetEnv, ACTION_SCALE, FRAME_SKIP
from wam.checks import SWEEP_RATE, SWEEP_STEPS, SETTLE_STEPS
kw = dict(start_spread=1.0, cabinet_spread=(0.1, np.radians(20)), ee_first_fraction=0.5)
env = gym.wrappers.TimeLimit(CloseCabinetEnv(**kw, wrist_leveler=True), 400)
u = env.unwrapped
sweep_a = SWEEP_RATE * FRAME_SKIP * u.model.opt.timestep / ACTION_SCALE
out = {}
for retract in (0.0, 0.4, 0.8):
    res = []
    for s in range(120):
        _, info = env.reset(seed=s)
        if not info["task/ee_first"]: continue
        home = u.home[:4]; phase, steps, done = 0, 0, False
        q1_goal = u.q_target[0] - retract  # sweep closes with +joint 1, so retract is -joint 1
        while not done:
            if phase == 0:
                a = np.zeros(4); a[0] = np.clip((q1_goal - u.q_target[0]) / ACTION_SCALE, -1, 1)
                if abs(q1_goal - u.q_target[0]) < 1e-3: phase = 1
            elif phase == 1:
                d = home - u.q_target[:4]
                a = np.clip(d / ACTION_SCALE, -1, 1)
                if np.abs(d).max() < 1e-3: phase = 2
            else:
                a = np.zeros(4); a[0] = sweep_a if steps < SWEEP_STEPS else 0.0; steps += 1
            _, _, term, trunc, info = env.step(a.astype(np.float32))
            done = term or trunc or steps >= SWEEP_STEPS + SETTLE_STEPS
        res.append((info["task/success"], info["task/spill"], info["task/valid_fraction"], info["task/wrong_contact_steps"] > 0))
    v = np.array(res, float)
    ok = v[:, 0] > 0
    print(f"retract {retract} rad: n={len(v)} success {ok.mean():.2f} | valid (forearm-only, valid fraction>0.9) success {np.mean(ok & (v[:,2] > 0.9)):.2f} | spill {v[:,1].mean():.2f} | any wrong contact {v[:,3].mean():.2f}")
