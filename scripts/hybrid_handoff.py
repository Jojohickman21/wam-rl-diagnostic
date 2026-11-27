"""Hybrid check (log 2026-11-27): the staged script's approach (Stage 1) hands off to a trained G policy (Stage 2).

Stage 1 is scripted_baseline.py's "staged" approach up to its standoff: the arm point nearest the goal is
driven by DLS (cap 0.5) to STANDOFF in front of the face centre, level with it and on the push normal. Once
within AT_STANDOFF, the policy takes over for the rest of the episode with its mean action, from whatever
state the script left (the policy's observation already carries the script's last actions and targets).
An episode that never reaches the standoff stays scripted to the end and counts as it ends.

Env, 250-step limit (Stage 1 steps count) and starts match eval_policy.py, so seed s sees the same starts as
the staged script's and the policy's own evals of seed s.

    python scripts/hybrid_handoff.py <train env id> <seed> [episodes] [outcomes.npy]
"""
import glob
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import task_feasibility as tf  # noqa: E402
from train_ppo import Agent  # noqa: E402
from wam.tasks import TaskEnv  # noqa: E402

train_env, seed = sys.argv[1], int(sys.argv[2])
episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 200
CAP, STANDOFF, AT_STANDOFF = 0.5, 0.15, 0.03  # as scripted_baseline.py's staged mode

u = TaskEnv(task="T2", cabinet_spread=(0.1, np.radians(20.0)), conditioning=gym.spec(train_env).kwargs["conditioning"])
env = gym.wrappers.TimeLimit(u, 250)


class Spaces:
    single_observation_space = env.observation_space
    single_action_space = env.action_space


path = glob.glob(f"runs/{train_env}__train_ppo__{seed}__*/train_ppo.cleanrl_model")[0]
agent = Agent(Spaces())
agent.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
stats = np.load(Path(path).with_name("obs_rms.npz"))
w = stats["count"][:, None]
mean = (w * stats["mean"]).sum(0) / w.sum()
var = (w * (stats["var"] + (stats["mean"] - mean) ** 2)).sum(0) / w.sum()

rows = []
obs, _ = env.reset(seed=50_000 + seed)
handed, handoff_step, steps = False, -1, 0
while len(rows) < episodes:
    if handed:
        x = np.clip((obs - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32)
        with torch.no_grad():
            action = agent.actor_mean(torch.tensor(x)[None])[0].numpy()
    else:
        target, direction = u.contact_target()
        goal = target - STANDOFF * direction
        point = min(tf.FOREARM_POINTS, key=lambda g: np.linalg.norm(u.data.geom_xpos[u.model.geom(g).id] - goal))
        action, error = tf.drive(u, goal, point)
        action *= min(1.0, CAP / max(np.abs(action).max(), 1e-9))
        if error < AT_STANDOFF:
            handed, handoff_step = True, steps + 1
    obs, _, terminated, truncated, info = env.step(np.clip(action, -1, 1))
    steps += 1
    if terminated or truncated:
        rows.append((info["task/success"], info["task/valid_fraction"], info["task/spill"],
                     float(truncated and not terminated), info["task/distance_norm"], float(handed), handoff_step))
        obs, _ = env.reset()
        handed, handoff_step, steps = False, -1, 0
v = np.array(rows)
if len(sys.argv) > 4:
    np.save(sys.argv[4], v)
won, h = v[:, 0] > 0, v[:, 5] > 0
print(f"hybrid ({train_env} seed {seed}) on T2 ({episodes} episodes): success {won.mean():.3f} | spill {v[:, 2].mean():.3f} "
      f"| timeout {v[:, 3].mean():.3f} | reached standoff {h.mean():.3f} (step median {np.median(v[h, 6]) if h.any() else float('nan'):.0f}) "
      f"| success after handoff {won[h].mean() if h.any() else float('nan'):.3f} | success without handoff {won[~h].mean() if (~h).any() else float('nan'):.3f}",
      flush=True)
