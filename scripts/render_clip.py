"""Render one episode of a trained policy with a chosen outcome, as MP4 and a small GIF.

    python scripts/render_clip.py <train env> <seed> <task or -> <success|timeout|stop|spill> <out stem> [camera]

Replays (saved normalisation statistics, actions sampled as in training) on the training env, or on
<task> as scripts/eval_policy.py builds it, until an episode ends with the wanted outcome ("stop" =
a timeout with a joint within 0.05 rad of a hard limit). Gives up after 40 episodes.
"""
import glob
import sys
from pathlib import Path

import gymnasium as gym
import imageio
import mujoco
import numpy as np
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import wam  # noqa: F401,E402
from train_ppo import Agent  # noqa: E402
from wam.tasks import TaskEnv  # noqa: E402

env_id, seed, task, want, stem = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
CAMERA = sys.argv[6] if len(sys.argv) > 6 else "overview"  # "top" shows the arm under a fixture
if task == "-":
    raw = gym.make(env_id)
else:
    kwargs = gym.spec(env_id).kwargs
    labels = kwargs["tasks"] if kwargs.get("task_labels") else None
    raw = gym.wrappers.TimeLimit(TaskEnv(task=task, outcome_reward="Outcome" in env_id, labels=labels), 250)
path = glob.glob(f"runs/{env_id}__train_ppo__{seed}__*/train_ppo.cleanrl_model")[0]


class Spaces:
    single_observation_space = raw.observation_space
    single_action_space = raw.action_space


agent = Agent(Spaces())
agent.load_state_dict(torch.load(path, map_location="cpu"))
torch.manual_seed(seed)
stats = np.load(Path(path).with_name("obs_rms.npz"))
w = stats["count"][:, None]
mean = (w * stats["mean"]).sum(0) / w.sum()
var = (w * (stats["var"] + (stats["mean"] - mean) ** 2)).sum(0) / w.sum()
LIMITS = {1: (-2.059, 2.0944), 3: (-0.19198, 3.927), 5: (-1.69297, 3.14159)}

obs, _ = raw.reset(seed=90_000 + seed)
for attempt in range(40):
    u = raw.unwrapped
    renderer = mujoco.Renderer(u.model, 480, 640)
    frames = []
    while True:
        renderer.update_scene(u.data, CAMERA)
        frames.append(renderer.render())
        x = np.clip((obs - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32)
        with torch.no_grad():
            action, *_ = agent.get_action_and_value(torch.tensor(x)[None])
        obs, _, terminated, truncated, info = raw.step(np.clip(action[0].numpy(), -1, 1))
        if terminated or truncated:
            break
    q = u.data.qpos[:7]
    at_stop = any(min(q[j] - lo, hi - q[j]) < 0.05 for j, (lo, hi) in LIMITS.items())
    outcome = "success" if info["task/success"] else "spill" if info["task/spill"] else ("stop" if at_stop else "timeout")
    renderer.close()
    if outcome == want or (want == "timeout" and outcome == "stop"):
        Path(stem).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(f"{stem}.mp4", frames, fps=50)
        imageio.mimsave(f"{stem}.gif", [f[::2, ::2] for f in frames[::2]], duration=0.04, loop=0)
        print(f"{stem}: {outcome} after {attempt + 1} episodes, {len(frames)} frames")
        break
    obs, _ = raw.reset()
else:
    print(f"{stem}: no '{want}' episode in 40 tries")
