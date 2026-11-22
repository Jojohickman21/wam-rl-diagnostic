"""Failure profile of a trained task-set policy (RESULTS.md §10), from an exact replay.

Replays runs/<env>__train_ppo__<seed>__* with its saved obs_rms.npz (pooled over env processes),
actions sampled as in training, and records per episode: outcome, distances, contacts, and how
hard the forearm pushes the object toward its goal. Writes one .npy of episode dicts.

    python scripts/diagnose_timeouts/task_profile.py <env id> <seed> <steps> <out.npy>
    python scripts/diagnose_timeouts/task_profile.py --summary <out*.npy>

"Push toward goal" generalises the door's hinge torque (§7.2): for each forearm (links 4-6)
contact, the force it puts on the mover is turned into a generalised force on each object joint
(torque about the axis for a hinge, force along the axis for a slide) and projected on the
direction from the joint state to the goal. Net/total near 0 with several contacts means pushes
that cancel, the §7.2 edge straddle. The sign is calibrated so successful pushing is positive.
"""
import glob
import sys
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import wam  # noqa: F401,E402

LIMITS = {1: (-2.059, 2.0944), 3: (-0.19198, 3.927), 5: (-1.69297, 3.14159)}


def replay(env_id, seed, steps, out):
    from train_ppo import Agent

    raw = gym.make(env_id)
    u = raw.unwrapped

    class Spaces:
        single_observation_space = raw.observation_space
        single_action_space = raw.action_space

    path = glob.glob(f"runs/{env_id}__train_ppo__{seed}__*/train_ppo.cleanrl_model")[0]
    agent = Agent(Spaces())
    agent.load_state_dict(torch.load(path, map_location="cpu"))
    torch.manual_seed(seed)
    stats = np.load(Path(path).with_name("obs_rms.npz"))
    w = stats["count"][:, None]
    mean = (w * stats["mean"]).sum(0) / w.sum()
    var = (w * (stats["var"] + (stats["mean"] - mean) ** 2)).sum(0) / w.sum()
    wrench = np.zeros(6)

    def push_toward_goal():
        """(net, total, count) of forearm-on-mover pushes projected on the toward-goal direction."""
        toward = u.goal - u.data.qpos[u.qpos_adr]
        norm = np.linalg.norm(toward)
        if norm < 1e-9:
            return 0.0, 0.0, 0
        toward = toward / norm
        net = total = 0.0
        count = 0
        for i in range(u.data.ncon):
            c = u.data.contact[i]
            on_mover = c.geom1 in u.mover_geom_set or c.geom2 in u.mover_geom_set
            arm = c.geom2 if c.geom1 in u.mover_geom_set else c.geom1
            if not on_mover or u.geom_group[arm] not in (1, 2, 3):
                continue
            mujoco.mj_contactForce(u.model, u.data, i, wrench)
            if wrench[0] < 1.0:
                continue
            force = wrench[0] * c.frame[:3] * (-1.0 if c.geom1 in u.mover_geom_set else 1.0)
            gen = np.zeros(len(u.joint_ids))
            for k, j in enumerate(u.joint_ids):
                axis = u.data.xaxis[j]
                if u.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                    gen[k] = np.cross(c.pos - u.data.xanchor[j], force) @ axis
                else:
                    gen[k] = force @ axis
            value = float(gen @ toward)
            net += value
            total += abs(value)
            count += 1
        return net, total, count

    episodes, record = [], None
    obs, info = raw.reset(seed=70_000 + seed)
    for _ in range(steps):
        if record is None:
            record = dict(start=info["task/distance_norm"], fa=[], wrong=[], ee=[], upper=[], dist=[], push=[], tilt=[])
        x = np.clip((obs - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32)
        with torch.no_grad():
            action, *_ = agent.get_action_and_value(torch.tensor(x)[None])
        obs, _, terminated, truncated, info = raw.step(np.clip(action[0].numpy(), -1, 1))
        record["fa"].append(info["force/door_link4"] + info["force/door_link5"] + info["force/door_link6"])
        record["ee"].append(info["force/door_ee"])
        record["upper"].append(info["force/door_upper"])
        record["wrong"].append(info["force/door_ee"] + info["force/door_upper"] + info["force/scene"])
        record["dist"].append(info["task/distance_norm"])
        record["tilt"].append(info["task/plate_tilt"])
        record["push"].append(push_toward_goal())
        if terminated or truncated:
            q = u.data.qpos[:7]
            record.update(
                success=bool(info["task/success"]), spill=bool(info["task/spill"]),
                timeout=bool(truncated and not terminated), valid=info["task/valid_fraction"],
                end=info["task/distance_norm"], length=len(record["dist"]),
                at_stop=bool(any(min(q[j] - lo, hi - q[j]) < 0.05 for j, (lo, hi) in LIMITS.items())),
            )
            for key in ("fa", "wrong", "ee", "upper", "dist", "tilt"):
                record[key] = np.array(record[key])
            record["push"] = np.array(record["push"])
            episodes.append(record)
            record = None
            obs, info = raw.reset()
    np.save(out, np.array(episodes, dtype=object), allow_pickle=True)
    print(f"{env_id} seed {seed}: {len(episodes)} episodes")


def summary(files):
    eps = np.concatenate([np.load(f, allow_pickle=True) for f in files])
    n = len(eps)
    won = [e for e in eps if e["success"]]
    pushes = np.concatenate([e["push"][(e["push"][:, 2] > 0)] for e in won if len(e["push"])] or [np.zeros((0, 3))])
    sign = np.sign(np.median(pushes[:, 0])) if len(pushes) else 1.0
    timeouts = [e for e in eps if e["timeout"]]
    spills = [e for e in eps if e["spill"]]
    print(f"{n} episodes | success {len(won)/n:.3f} spill {len(spills)/n:.3f} timeout {len(timeouts)/n:.3f}"
          f" | success push sign {sign:+.0f}")
    if spills:
        steps = np.array([e["length"] for e in spills])
        moved = np.array([e["start"] - e["end"] for e in spills])
        contact = np.array([np.mean(e["fa"] > 1) for e in spills])
        print(f"  spills: median step {np.median(steps):.0f} | object moved toward goal before spilling (normalised) median {np.median(moved):.2f}"
              f" | spilled with <10% of the way done {np.mean(moved < 0.1):.2f} | share of steps with forearm contact, median {np.median(contact):.2f}"
              f" | hand on object at some step {np.mean([np.any(e['ee'] > 0) for e in spills]):.2f}")
    if timeouts:
        last = slice(-50, None)
        contact = np.array([np.mean(e["fa"][last] > 1) for e in timeouts])
        moved = np.array([e["dist"][-50] - e["dist"][-1] for e in timeouts])
        end = np.array([e["end"] for e in timeouts])
        stuck = (contact > 0.8) & (np.abs(moved) < 0.01)
        off = contact < 0.2
        print(f"  timeouts: final distance (normalised) 25/50/75 {np.round(np.percentile(end, [25, 50, 75]), 2)}"
              f" | near-miss (<0.2 left) {np.mean(end < 0.2):.2f} | never started (>0.9 left) {np.mean(end > 0.9):.2f}")
        print(f"    pressing-but-stuck {stuck.mean():.2f} | forearm off the object {off.mean():.2f} | other {1 - stuck.mean() - off.mean():.2f}"
              f" | wrong contact in last 50 steps {np.mean([np.any(e['wrong'][last] > 0) for e in timeouts]):.2f}"
              f" | hand on object in last 50 {np.mean([np.any(e['ee'][last] > 0) for e in timeouts]):.2f}"
              f" | a joint at a hard stop {np.mean([e['at_stop'] for e in timeouts]):.2f}")
        stuck_eps = [e for e, s in zip(timeouts, stuck) if s]
        if stuck_eps:
            p = np.concatenate([e["push"][last][e["push"][last][:, 2] > 0] for e in stuck_eps])
            ratio = np.abs(p[:, 0]) / np.maximum(p[:, 1], 1e-9)
            print(f"    stuck pushes: contacts per step median {np.median(p[:, 2]):.0f} (>=2 on {np.mean(p[:, 2] >= 2):.2f})"
                  f" | net/total median {np.median(ratio):.2f} (straddle-like <0.2 on {np.mean(ratio < 0.2):.2f} of steps)"
                  f" | net push toward goal median {sign * np.median(p[:, 0]):.1f} | total median {np.median(p[:, 1]):.1f}")
    if len(pushes):
        ratio = np.abs(pushes[:, 0]) / np.maximum(pushes[:, 1], 1e-9)
        print(f"  successes (reference): contacts per step median {np.median(pushes[:, 2]):.0f} | net/total median {np.median(ratio):.2f}"
              f" | net push toward goal median {sign * np.median(pushes[:, 0]):.1f}")


if __name__ == "__main__":
    if sys.argv[1] == "--summary":
        summary(sys.argv[2:])
    else:
        replay(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
