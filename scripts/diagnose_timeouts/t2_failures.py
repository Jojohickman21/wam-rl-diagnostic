"""Why episodes fail on held-out T2: collisions with the chest, or tracking errors in free space?

Replays a G policy (exact: saved obs_rms.npz, sampled actions, torch seed = run seed) or the §11
scripted push (cap 0.5) on T2 with eval_policy.py's starts (reset seed 50_000 + seed), so success
rates reproduce the §12 / §11 numbers. Per step it records contact of any arm geom with the chest
(the fixture body), arm contact with the drawer by group, progress, the forearm point's distance to
the contact target, joint speed, and joints at a hard stop. Writes one .npy of episode dicts and, with
[clips], one 640x480 MP4 per failure label (first episode of each) plus one success.

    [EPISODES=n] python scripts/diagnose_timeouts/t2_failures.py <WamFamilyContactFK-v0 | scripted> <seed> <out stem> [clips]
    python scripts/diagnose_timeouts/t2_failures.py --summary <label> <out*.npy>

Labels (last 50 steps of a timeout, first matching rule wins):
  chest_stall     arm on the chest on >= 30% of steps and < 0.02 progress (normalised)
  joint_stop      a limited joint within 0.05 rad of its stop at the end
  pressing_stuck  forearm (links 4-6) on the drawer on >= 80% of steps and < 0.02 progress
  wrong_contact   hand or upper arm on the drawer on >= 30% of steps
  free_space      arm touches neither chest nor drawer on >= 80% of steps: an approach/tracking error
  other
Spills: pre_contact (forearm never touched the drawer), chest_impact (arm on the chest in the 10
steps before the spill), during_push.
"""
import glob
import os
import sys
from pathlib import Path

import gymnasium as gym
import imageio
import mujoco
import numpy as np
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import task_feasibility as tf  # noqa: E402
from wam.tasks import TaskEnv  # noqa: E402

LIMITS = {1: (-2.059, 2.0944), 3: (-0.19198, 3.927), 5: (-1.69297, 3.14159)}
CAP, OFFSET = 0.5, 0.05
EPISODES = int(os.environ.get("EPISODES", 200))  # fewer for a smoke test


def label(e):
    if e["success"]:
        return "success"
    if e["spill"]:
        if not np.any(e["forearm"] > 0):
            return "spill_pre_contact"
        return "spill_chest_impact" if np.any(e["chest"][-10:]) else "spill_during_push"
    last = slice(-50, None)
    progress = e["dist"][-50] - e["dist"][-1]
    chest, fa, wrong = e["chest"][last].mean(), (e["forearm"][last] > 0).mean(), (e["wrong"][last] > 0).mean()
    if chest >= 0.3 and progress < 0.02:
        return "chest_stall"
    if e["at_stop"]:
        return "joint_stop"
    if fa >= 0.8 and progress < 0.02:
        return "pressing_stuck"
    if wrong >= 0.3:
        return "wrong_contact"
    if np.mean(e["chest"][last] | (e["forearm"][last] > 0) | (e["wrong"][last] > 0)) <= 0.2:
        return "free_space"
    return "other"


def replay(policy, seed, stem, clips):
    conditioning = "contact" if policy == "scripted" else gym.spec(policy).kwargs["conditioning"]
    u = TaskEnv(task="T2", cabinet_spread=(0.1, np.radians(20.0)), conditioning=conditioning)
    env = gym.wrappers.TimeLimit(u, 250)
    chest = u.model.geom("chest").id
    arm = set(np.flatnonzero(u.geom_group >= 0))
    point = u.model.geom("link5_c2").id
    if policy != "scripted":
        from train_ppo import Agent

        class Spaces:
            single_observation_space = env.observation_space
            single_action_space = env.action_space

        path = glob.glob(f"runs/{policy}__train_ppo__{seed}__*/train_ppo.cleanrl_model")[0]
        agent = Agent(Spaces())
        agent.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        torch.manual_seed(seed)
        stats = np.load(Path(path).with_name("obs_rms.npz"))
        w = stats["count"][:, None]
        mean = (w * stats["mean"]).sum(0) / w.sum()
        var = (w * (stats["var"] + (stats["mean"] - mean) ** 2)).sum(0) / w.sum()
    renderer = mujoco.Renderer(u.model, 480, 640) if clips else None
    saved, episodes = set(), []
    obs, info = env.reset(seed=50_000 + seed)
    pushing = False
    rec = dict(chest=[], forearm=[], wrong=[], dist=[], track=[], speed=[], frames=[])
    while len(episodes) < EPISODES:
        if renderer:
            renderer.update_scene(u.data, "overview")
            rec["frames"].append(renderer.render())
        target, direction = u.contact_target()
        if policy == "scripted":
            goal = u.data.geom_xpos[point] + tf.PUSH_STEP / tf.GAIN * direction if pushing else target - OFFSET * direction
            action, error = tf.drive(u, goal, "link5_c2")
            action *= min(1.0, CAP / max(np.abs(action).max(), 1e-9))
        else:
            x = np.clip((obs - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32)
            with torch.no_grad():
                action = agent.get_action_and_value(torch.tensor(x)[None])[0][0].numpy()
        obs, _, terminated, truncated, info = env.step(np.clip(action, -1, 1))
        if policy == "scripted":
            pushing |= error < tf.REACHED or info["force/door_link5"] > 0
        touching = any((c.geom1 == chest and c.geom2 in arm) or (c.geom2 == chest and c.geom1 in arm)
                       for c in u.data.contact[: u.data.ncon])
        rec["chest"].append(touching)
        rec["forearm"].append(info["force/door_link4"] + info["force/door_link5"] + info["force/door_link6"])
        rec["wrong"].append(info["force/door_ee"] + info["force/door_upper"])
        rec["dist"].append(info["task/distance_norm"])
        rec["track"].append(float(np.linalg.norm(target - u.data.geom_xpos[point])))
        rec["speed"].append(float(np.abs(u.data.qvel[:4]).max()))
        if terminated or truncated:
            q = u.data.qpos[:7]
            frames = rec.pop("frames")
            e = {k: np.array(v) for k, v in rec.items()}
            e.update(success=bool(info["task/success"]), spill=bool(info["task/spill"]),
                     at_stop=bool(any(min(q[j] - lo, hi - q[j]) < 0.05 for j, (lo, hi) in LIMITS.items())),
                     stops=[j + 1 for j, (lo, hi) in LIMITS.items() if min(q[j] - lo, hi - q[j]) < 0.05])
            e["label"] = label(e)
            episodes.append(e)
            if renderer and e["label"] not in saved:
                saved.add(e["label"])
                imageio.mimsave(f"{stem}_ep{len(episodes) - 1:03d}_{e['label']}.mp4", frames, fps=50, macro_block_size=8)
            rec = dict(chest=[], forearm=[], wrong=[], dist=[], track=[], speed=[], frames=[])
            pushing = False
            obs, info = env.reset()
    np.save(f"{stem}.npy", np.array(episodes, dtype=object), allow_pickle=True)
    print(f"{policy} seed {seed}: success {np.mean([e['success'] for e in episodes]):.3f}", flush=True)


LABELS = ("success", "spill_pre_contact", "spill_chest_impact", "spill_during_push", "chest_stall",
          "joint_stop", "pressing_stuck", "wrong_contact", "free_space", "other")


def summary(name, files):
    eps = np.concatenate([np.load(f, allow_pickle=True) for f in files])
    fail = [e for e in eps if not e["success"]]
    counts = {k: sum(e["label"] == k for e in eps) for k in LABELS}
    print(f"\n{name}: {len(files)} seeds, {len(eps)} episodes, success {counts['success'] / len(eps):.3f}, {len(fail)} failures")
    print("  share of failures: " + " | ".join(f"{k} {counts[k] / max(len(fail), 1):.2f}" for k in LABELS[1:]))
    touched = np.mean([e["chest"].any() for e in fail]) if fail else float("nan")
    won = [e for e in eps if e["success"]]
    print(f"  arm touched the chest at some step: failures {touched:.2f}, successes "
          f"{np.mean([e['chest'].any() for e in won]) if won else float('nan'):.2f}")
    free = [e for e in fail if e["label"] == "free_space"]
    if free:
        print(f"  free_space timeouts: forearm-to-target distance at end median {np.median([e['track'][-1] for e in free]):.3f} m, "
              f"max joint speed (joints 1-4) over last 50 steps median {np.median([e['speed'][-50:].max() for e in free]):.2f} rad/s, "
              f"progress made before timing out median {np.median([1 - e['dist'][-1] / max(e['dist'][0], 1e-9) for e in free]):.2f}")
    stop = [j for e in fail if e["label"] == "joint_stop" for j in e["stops"]]
    if stop:
        print(f"  joint_stop: joints at the stop {dict(zip(*np.unique(stop, return_counts=True)))}")


if __name__ == "__main__":
    if sys.argv[1] == "--summary":
        summary(sys.argv[2], sys.argv[3:])
    else:
        replay(sys.argv[1], int(sys.argv[2]), sys.argv[3], len(sys.argv) > 4 and sys.argv[4] == "clips")
