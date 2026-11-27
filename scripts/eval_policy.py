"""Evaluate a trained policy on any task, including tasks and fixture poses it never trained on.

Loads runs/<train env>__train_ppo__<seed>__*/train_ppo.cleanrl_model with its obs_rms.npz (all env
processes pooled) and runs it on a TaskEnv with actions sampled from the policy, as in training.
The normalisation statistics stay frozen at their trained values (applied by hand, as
NormalizeObservation does with epsilon 1e-8, then clipped to ±10), so unseen tasks cannot shift them.

    python scripts/eval_policy.py <train env id> <seed> <task> [episodes] [yaw spread, degrees] [det]

With `det`, the policy's mean action is used instead of a sample (output line says so).

The default yaw spread is training's 20 degrees; a larger one tests unseen fixture orientations.
"""
import glob
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import wam  # noqa: F401,E402
from train_ppo import Agent  # noqa: E402
from wam.tasks import TaskEnv  # noqa: E402

train_env, seed, task = sys.argv[1], int(sys.argv[2]), sys.argv[3]
episodes = int(sys.argv[4]) if len(sys.argv) > 4 else 200
yaw_spread = float(sys.argv[5]) if len(sys.argv) > 5 else 20.0
deterministic = len(sys.argv) > 6 and sys.argv[6] == "det"
outcome = "Outcome" in train_env
# A label-trained policy sees the label form of the observation, with a one-hot over its training
# tasks (all zeros for a task it never trained on); every other policy sees the full descriptor.
train_kwargs = gym.spec(train_env).kwargs
labels = train_kwargs["tasks"] if train_kwargs.get("task_labels") else None
conditioning = train_kwargs.get("conditioning", "descriptor")  # the family policies see contact targets etc.

raw = gym.wrappers.TimeLimit(
    TaskEnv(task=task, outcome_reward=outcome, cabinet_spread=(0.1, np.radians(yaw_spread)), labels=labels,
            conditioning=conditioning), 250
)
path = glob.glob(f"runs/{train_env}__train_ppo__{seed}__*/train_ppo.cleanrl_model")[0]
stats = np.load(Path(path).with_name("obs_rms.npz"))
weights = stats["count"][:, None]
total = weights.sum()
mean = (weights * stats["mean"]).sum(0) / total
var = (weights * (stats["var"] + (stats["mean"] - mean) ** 2)).sum(0) / total


class Spaces:
    single_observation_space = raw.observation_space
    single_action_space = raw.action_space


agent = Agent(Spaces())
agent.load_state_dict(torch.load(path, map_location="cpu"))
torch.manual_seed(seed)


def frozen(obs):
    return np.clip((obs - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32)


rows = []
raw_obs, _ = raw.reset(seed=50_000 + seed)
while len(rows) < episodes:
    with torch.no_grad():
        x = torch.tensor(frozen(raw_obs))[None]
        action = agent.actor_mean(x) if deterministic else agent.get_action_and_value(x)[0]
    raw_obs, _, terminated, truncated, info = raw.step(np.clip(action[0].numpy(), -1, 1))
    if terminated or truncated:
        rows.append((info["task/success"], info["task/valid_fraction"], info["task/spill"],
                     float(truncated and not terminated), info["task/distance_norm"]))
        raw_obs, _ = raw.reset()
v = np.array(rows)
won = v[:, 0] > 0
print(f"{train_env} seed {seed}{' (mean action)' if deterministic else ''} on {task} (yaw ±{yaw_spread:.0f}°, {episodes} episodes): "
      f"success {won.mean():.3f} | valid fraction of successes {v[won, 1].mean() if won.any() else float('nan'):.3f} "
      f"| spill {v[:, 2].mean():.3f} | timeout {v[:, 3].mean():.3f} | final distance (normalised) median {np.median(v[:, 4]):.2f}")
