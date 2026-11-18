"""Replay a saved PPO policy and record per-episode diagnostics.

Observation-normalisation stats were not saved, so they are rebuilt by running the policy through
the same NormalizeObservation wrapper for WARMUP steps (stats keep updating, as in training).
Usage: replay.py <env id> <seed> <out.npz> [record_steps]
"""
import glob, sys
import gymnasium as gym, mujoco, numpy as np, torch
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import wam  # noqa
from train_ppo import Agent

env_id, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
RECORD = int(sys.argv[4]) if len(sys.argv) > 4 else 100_000
WARMUP = 30_000
raw = gym.make(env_id)
env = gym.wrappers.TransformObservation(gym.wrappers.NormalizeObservation(gym.wrappers.ClipAction(raw)), lambda o: np.clip(o, -10, 10))
u = raw.unwrapped
class Spaces: single_observation_space = raw.observation_space; single_action_space = raw.action_space
agent = Agent(Spaces()); path = glob.glob(f"runs/{env_id}__train_ppo__{seed}__*/train_ppo.cleanrl_model")[0]
agent.load_state_dict(torch.load(path, map_location="cpu")); torch.manual_seed(seed)
door_jnt = int(np.where(u.model.jnt_qposadr == 7)[0][0])
GROUP = {name: i for i, name in enumerate(("upper", "link4", "link5", "link6", "ee"))}

def lever_arms():
    """Distance from the hinge axis of each forearm-door contact point (m)."""
    anchor, axis = u.data.xanchor[door_jnt], u.data.xaxis[door_jnt]
    arms = []
    for c in u.data.contact[:u.data.ncon]:
        if u.door_geom in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == u.door_geom else c.geom1
            if u.geom_group[other] in (1, 2, 3):
                r = c.pos - anchor; arms.append(np.linalg.norm(r - (r @ axis) * axis))
    return min(arms) if arms else np.nan

episodes, cur = [], None
obs, _ = env.reset(seed=10_000 + seed); steps = 0
def start():
    return dict(hinge=u.model.body_pos[u.cabinet_body][:2].copy(), yaw=u.cabinet_yaw, slide=u.tilt_max,
                q0=u.data.qpos[:7].copy(), door0=u.data.qpos[7], angles=[], tilt=[], lever=[], fa=[], wrong=[], qtrail=[])
cur = start()
while steps < WARMUP + RECORD:
    with torch.no_grad():
        a, *_ = agent.get_action_and_value(torch.tensor(obs, dtype=torch.float32)[None])
    obs, r, term, trunc, info = env.step(a[0].numpy()); steps += 1
    if steps > WARMUP:
        cur["angles"].append(info["task/door_angle"]); cur["tilt"].append(info["task/plate_tilt"])
        cur["fa"].append(info["force/door_link4"] + info["force/door_link5"] + info["force/door_link6"])
        cur["wrong"].append(info["force/door_ee"] + info["force/door_upper"] + info["force/scene"])
        cur["lever"].append(lever_arms()); cur["qtrail"].append(np.concatenate([u.data.qpos[:7], u.q_target]))
    if term or trunc:
        if steps > WARMUP + 1 and len(cur["angles"]) == len(cur["fa"]) and cur["angles"]:
            ep = {k: v for k, v in cur.items() if k not in ("angles", "tilt", "lever", "fa", "wrong", "qtrail")}
            ep.update(success=info["task/success"], spill=info["task/spill"], timeout=bool(trunc and not term),
                      length=len(cur["angles"]), door_end=cur["angles"][-1], qend=u.data.qpos[:7].copy(),
                      angles=np.array(cur["angles"]), fa=np.array(cur["fa"]), wrong=np.array(cur["wrong"]),
                      lever=np.array(cur["lever"]), tilt=np.array(cur["tilt"]), qtrail=np.array(cur["qtrail"]))
            episodes.append(ep)
        obs, _ = env.reset(); cur = start()
np.save(out, np.array(episodes, dtype=object), allow_pickle=True)
s = np.mean([e["success"] for e in episodes]); t = np.mean([e["timeout"] for e in episodes]); p = np.mean([e["spill"] for e in episodes])
print(f"{env_id} seed {seed}: {len(episodes)} episodes, success {s:.3f} spill {p:.3f} timeout {t:.3f}")
