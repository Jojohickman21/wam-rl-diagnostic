"""What happens when a G policy's forearm contact with the T2 drawer first breaks?

Exact replay on eval_policy.py's starts (as t2_failures.py). Per step: the action, its change ‖Δa‖,
drawer speed toward closed, forearm force on the drawer, and the forearm point's offset from the
contact target (the face centre) in the face frame: n along the push direction (negative = in front
of the face), lateral, and vertical (face half-size 0.18 x 0.11 m, capsule radius 0.04 m). A "break"
is the first step after at least 3 steps of forearm contact with no forearm contact.

    python scripts/diagnose_timeouts/t2_contact_break.py <seed> [episode to print and render]

Prints, over level starts (|start height| < 0.10 m), how often contact breaks and what the step
looks like (‖Δa‖ there vs. elsewhere, where the forearm point is, whether the drawer had stopped);
with an episode index, the ±15-step trace around its break and an MP4 next to t2_failures' clips.
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
from train_ppo import Agent  # noqa: E402
from wam.tasks import TaskEnv  # noqa: E402

ENV = "WamFamilyContactFK-v0"
seed = int(sys.argv[1])
show = int(sys.argv[2]) if len(sys.argv) > 2 else None
u = TaskEnv(task="T2", cabinet_spread=(0.1, np.radians(20.0)), conditioning="contact_fk")
env = gym.wrappers.TimeLimit(u, 250)
point = u.model.geom("link5_c2").id


class Spaces:
    single_observation_space = env.observation_space
    single_action_space = env.action_space


path = glob.glob(f"runs/{ENV}__train_ppo__{seed}__*/train_ppo.cleanrl_model")[0]
agent = Agent(Spaces())
agent.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
torch.manual_seed(seed)
stats = np.load(Path(path).with_name("obs_rms.npz"))
w = stats["count"][:, None]
mean = (w * stats["mean"]).sum(0) / w.sum()
var = (w * (stats["var"] + (stats["mean"] - mean) ** 2)).sum(0) / w.sum()
renderer = mujoco.Renderer(u.model, 480, 640) if show is not None else None

episodes, frames = [], []
obs, _ = env.reset(seed=50_000 + seed)
rec, last = [], np.zeros(4)
while len(episodes) < 200:
    target, direction = u.contact_target()
    if not rec:
        start_height = u.data.geom_xpos[point][2] - target[2]
    if renderer is not None and len(episodes) == show:
        renderer.update_scene(u.data, "overview")
        frames.append(renderer.render())
    x = np.clip((obs - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32)
    with torch.no_grad():
        action = np.clip(agent.get_action_and_value(torch.tensor(x)[None])[0][0].numpy(), -1, 1)
    lateral = np.cross([0.0, 0.0, 1.0], direction)
    lateral /= np.linalg.norm(lateral)
    offset = u.data.geom_xpos[point] - target
    obs, _, terminated, truncated, info = env.step(action)
    rec.append(dict(action=action, da=float(np.linalg.norm(action - last)) if rec else 0.0,
                    speed=-float(u.data.qvel[u.qvel_adr[0]]),
                    force=info["force/door_link4"] + info["force/door_link5"] + info["force/door_link6"],
                    n=offset @ direction, lat=offset @ lateral, up=offset[2], tilt=np.degrees(info["task/plate_tilt"]),
                    dist=info["task/distance_norm"]))
    last = action
    if terminated or truncated:
        outcome = "success" if info["task/success"] else "spill" if info["task/spill"] else "timeout"
        episodes.append(dict(start_height=start_height, outcome=outcome, steps=rec))
        rec, last = [], np.zeros(4)
        obs, _ = env.reset()


def first_break(steps):
    run = 0
    for t, s in enumerate(steps):
        if s["force"] > 0:
            run += 1
        elif run >= 3:
            return t
        else:
            run = 0
    return None


level = [(k, e) for k, e in enumerate(episodes) if abs(e["start_height"]) < 0.10]
broken = [(k, e, first_break(e["steps"])) for k, e in level if first_break(e["steps"]) is not None]
print(f"{ENV} seed {seed}: {len(level)} level starts, success {np.mean([e['outcome'] == 'success' for _, e in level]):.2f}; "
      f"contact broke in {len(broken)} ({np.mean([e['outcome'] == 'success' for _, e, _ in broken]) if broken else float('nan'):.2f} of those still succeeded)")
if broken:
    da_break = [max(s["da"] for s in e["steps"][max(t - 2, 0): t + 1]) for _, e, t in broken]
    da_all = [s["da"] for _, e in level for s in e["steps"][1:]]
    at = [e["steps"][t - 1] for _, e, t in broken]
    print(f"  max ‖Δa‖ over the 3 steps up to the break: median {np.median(da_break):.2f} (all steps: median {np.median(da_all):.2f}, 90th pct {np.percentile(da_all, 90):.2f})")
    print(f"  forearm point at the last contact step: lateral |·| median {np.median([abs(s['lat']) for s in at]):.3f} m, "
          f"vertical median {np.median([s['up'] for s in at]):+.3f} m (|·| > 0.11 m: {np.mean([abs(s['up']) > 0.11 for s in at]):.2f}), "
          f"n median {np.median([s['n'] for s in at]):+.3f} m")
    print(f"  drawer speed at the last contact step median {np.median([s['speed'] for s in at]):.3f} m/s; push done at the break median "
          f"{np.median([1 - e['steps'][t]['dist'] / max(e['steps'][0]['dist'], 1e-9) for _, e, t in broken]):.2f}")
    print(f"  outcomes after a break: {dict(zip(*np.unique([e['outcome'] for _, e, _ in broken], return_counts=True)))}")
    print("  level-start episodes with a break: " + " ".join(str(k) for k, _, _ in broken[:30]))
if show is not None:
    e = episodes[show]
    t = first_break(e["steps"])
    print(f"\nepisode {show}: start height {e['start_height']:+.3f} m, outcome {e['outcome']}, first break at step {t}")
    print("step | action (j1-j4)            | ‖Δa‖ | drawer m/s | force N | n      lateral vertical | tilt° | left")
    for i in range(max((t or 0) - 15, 0), min((t or 0) + 16, len(e["steps"]))):
        s = e["steps"][i]
        mark = " <- break" if i == t else ""
        print(f"{i:4d} | {' '.join(f'{a:+.2f}' for a in s['action'])} | {s['da']:.2f} | {s['speed']:+.3f}     | {s['force']:6.1f}  | "
              f"{s['n']:+.3f} {s['lat']:+.3f} {s['up']:+.3f}  | {s['tilt']:4.1f}  | {s['dist']:.2f}{mark}")
    clip = f"runs/diagnostics/t2_failures/G3_s{seed}_ep{show:03d}_contact_break.mp4"
    imageio.mimsave(clip, frames, fps=50, macro_block_size=8)
    print(clip)
