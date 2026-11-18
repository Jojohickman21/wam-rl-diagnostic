"""Forearm-only feasibility on end-effector-first starts, plain 7-joint env (no leveler).
Transit: joints 1-4 drive to home; the wrist levels the plate by damped least squares and uses the
remaining (null-space) freedom to move toward the home wrist pose. Then the standard base-yaw sweep."""
import sys
import mujoco, numpy as np, gymnasium as gym
sys.path.insert(0, ".")
from wam.close_cabinet import CloseCabinetEnv, ACTION_SCALE, FRAME_SKIP
from wam.checks import SWEEP_RATE, SWEEP_STEPS, SETTLE_STEPS
env = gym.wrappers.TimeLimit(CloseCabinetEnv(start_spread=1.0, cabinet_spread=(0.1, np.radians(20)), ee_first_fraction=0.5), 600)
u = env.unwrapped; scratch = mujoco.MjData(u.model); jac = np.zeros((3, u.model.nv))
sweep_a = SWEEP_RATE * FRAME_SKIP * u.model.opt.timestep / ACTION_SCALE
def wrist_step(arm_target, k=0.5):
    scratch.qpos[:] = u.data.qpos; scratch.qpos[:4] = arm_target; qw = u.q_target[4:].copy(); scratch.qpos[4:7] = qw
    mujoco.mj_kinematics(u.model, scratch); mujoco.mj_comPos(u.model, scratch)
    n = scratch.xmat[u.link7].reshape(3, 3)[:, 1]; e = np.cross(n, [0, 0, np.sign(n[2])])
    mujoco.mj_jacBody(u.model, scratch, None, jac, u.link7); J = jac[:, 4:7]
    Jp = J.T @ np.linalg.inv(J @ J.T + 1e-3 * np.eye(3))
    return Jp @ e + (np.eye(3) - Jp @ J) @ (k * (u.home[4:7] - qw))
res = {True: [], False: []}
for s in range(160):
    _, info = env.reset(seed=s); ee = bool(info["task/ee_first"]); steps = 0; done = False; homing = 0
    while not done:
        d = u.home[:7] - u.q_target
        w = wrist_step(u.q_target[:4] + np.clip(d[:4], -ACTION_SCALE, ACTION_SCALE))
        # Home phase ends when joints 1-4 are home and the wrist has settled on whatever level branch it reached.
        if steps == 0 and homing < 200 and (np.abs(d[:4]).max() > 2e-3 or np.abs(w).max() > 1e-3):
            homing += 1
            a = np.concatenate([np.clip(d[:4] / ACTION_SCALE, -1, 1), np.clip(w / ACTION_SCALE, -1, 1)])
        else:
            a = np.zeros(7); a[0] = sweep_a if steps < SWEEP_STEPS else 0.0; steps += 1
        _, _, term, trunc, info = env.step(a.astype(np.float32))
        done = term or trunc or steps >= SWEEP_STEPS + SETTLE_STEPS
    res[ee].append((info["task/success"], info["task/spill"], info["task/valid_fraction"], info["task/wrong_contact_steps"] > 0, info["force/door_ee"] > 0))
for k, v in res.items():
    v = np.array(v, float); ok = v[:, 0] > 0
    print(f"ee_first={k}: n={len(v)} success {ok.mean():.2f} | forearm-only success (valid>0.9) {np.mean(ok & (v[:,2] > .9)):.2f} | spill {v[:,1].mean():.2f} | any wrong contact {v[:,3].mean():.2f}")
