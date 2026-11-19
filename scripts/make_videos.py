"""Record short MP4 clips of a trained policy (exact replay with the run's saved obs_rms.npz).

Writes artifacts/videos/<label>_s<seed>_ep<k>_<outcome>.mp4 for the first N_EP episodes.
Outcome: success / spill / timeout; '+hand' when the hand pushed the door more than the forearm.
Usage: make_videos.py <env id> <seed> <label> [n_episodes]
"""
import glob, sys
from pathlib import Path
import gymnasium as gym, imageio, mujoco, numpy as np, torch
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import wam  # noqa
from train_ppo import Agent

env_id, seed, label = sys.argv[1], int(sys.argv[2]), sys.argv[3]
N_EP = int(sys.argv[4]) if len(sys.argv) > 4 else 4
raw = gym.make(env_id)
env = gym.wrappers.TransformObservation(gym.wrappers.NormalizeObservation(gym.wrappers.ClipAction(raw)), lambda o: np.clip(o, -10, 10))
u = raw.unwrapped
class Spaces: single_observation_space = raw.observation_space; single_action_space = raw.action_space
agent = Agent(Spaces()); path = glob.glob(f"runs/{env_id}__train_ppo__{seed}__*/train_ppo.cleanrl_model")[0]
agent.load_state_dict(torch.load(path, map_location="cpu")); torch.manual_seed(seed)
d = np.load(Path(path).with_name("obs_rms.npz")); w = d["count"][:, None]; n = w.sum()
mean = (w * d["mean"]).sum(0) / n
rms = env.get_wrapper_attr("obs_rms")
rms.mean, rms.var, rms.count = mean, (w * (d["var"] + (d["mean"] - mean) ** 2)).sum(0) / n, n
renderer = mujoco.Renderer(u.model, 480, 640)
fps = raw.metadata["render_fps"]
out = Path("artifacts/videos")

obs, _ = env.reset(seed=20_000 + seed)
for k in range(N_EP):
    frames, hand, forearm = [], 0.0, 0.0
    while True:
        renderer.update_scene(u.data, "overview"); frames.append(renderer.render())
        with torch.no_grad():
            a, *_ = agent.get_action_and_value(torch.tensor(obs, dtype=torch.float32)[None])
        obs, r, term, trunc, info = env.step(a[0].numpy())
        hand += info["force/door_ee"]; forearm += info["force/door_link4"] + info["force/door_link5"] + info["force/door_link6"]
        if term or trunc:
            break
    outcome = "success" if info["task/success"] else "spill" if info["task/spill"] else "timeout"
    if hand > forearm: outcome += "+hand"
    f = out / f"{label}_s{seed}_ep{k}_{outcome}.mp4"
    imageio.mimsave(f, frames, fps=fps); print(f, len(frames))
    obs, _ = env.reset()
