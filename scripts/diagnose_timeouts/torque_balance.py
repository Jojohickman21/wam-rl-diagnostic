"""Per step, net vs total torque that forearm-door contacts put on the hinge, for A22 timeouts
(last 50 steps) and successes. Net << total means contacts on opposite faces cancel (straddling).
Sign is calibrated by successes: their median net torque is the closing direction.
Usage: torque_balance.py <env id> <seed> <record steps>"""
import glob, sys
from pathlib import Path
import gymnasium as gym, mujoco, numpy as np, torch
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import wam  # noqa
from train_ppo import Agent
env_id, seed, RECORD = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
raw = gym.make(env_id)
env = gym.wrappers.TransformObservation(gym.wrappers.NormalizeObservation(gym.wrappers.ClipAction(raw)), lambda o: np.clip(o, -10, 10))
u = raw.unwrapped
class Spaces: single_observation_space = raw.observation_space; single_action_space = raw.action_space
agent = Agent(Spaces()); path = glob.glob(f"runs/{env_id}__train_ppo__{seed}__*/train_ppo.cleanrl_model")[0]
agent.load_state_dict(torch.load(path, map_location="cpu")); torch.manual_seed(seed)
d = np.load(Path(path).with_name("obs_rms.npz")); w = d["count"][:, None]; n = w.sum(); mean = (w * d["mean"]).sum(0) / n
rms = env.get_wrapper_attr("obs_rms"); rms.mean, rms.var, rms.count = mean, (w * (d["var"] + (d["mean"] - mean) ** 2)).sum(0) / n, n
door_jnt = int(np.where(u.model.jnt_qposadr == 7)[0][0]); wrench = np.zeros(6)
def step_torques():
    net = tot = 0.0; count = 0
    for i in range(u.data.ncon):
        c = u.data.contact[i]
        if u.door_geom not in (c.geom1, c.geom2): continue
        other = c.geom2 if c.geom1 == u.door_geom else c.geom1
        if u.geom_group[other] not in (1, 2, 3): continue
        mujoco.mj_contactForce(u.model, u.data, i, wrench)
        if wrench[0] < 1.0: continue
        force_on_door = wrench[0] * c.frame[:3] * (-1.0 if c.geom1 == u.door_geom else 1.0)
        t = np.cross(c.pos - u.data.xanchor[door_jnt], force_on_door) @ u.data.xaxis[door_jnt]
        net += t; tot += abs(t); count += 1
    return net, tot, count
obs, _ = env.reset(seed=30_000 + seed); steps = 0; ep = []; res = {"timeout": [], "success": []}
while steps < RECORD:
    with torch.no_grad(): a, *_ = agent.get_action_and_value(torch.tensor(obs, dtype=torch.float32)[None])
    obs, _, term, trunc, info = env.step(a[0].numpy()); steps += 1
    ep.append(step_torques())
    if term or trunc:
        key = "timeout" if trunc and not term else ("success" if info["task/success"] else None)
        if key: res[key].append(ep[-50:] if key == "timeout" else ep)
        ep = []; obs, _ = env.reset()
close_sign = np.sign(np.median([s[0] for e in res["success"] for s in e if s[2] > 0]))
for key, eps in res.items():
    s = np.array([x for e in eps for x in e if x[2] > 0])
    if not len(s): print(key, "no contact steps"); continue
    net, tot, cnt = close_sign * s[:, 0], s[:, 1], s[:, 2]
    print(f"{key}: {len(eps)} eps, {len(s)} contact steps | contacts per step median {np.median(cnt):.0f}, share with >=2: {np.mean(cnt >= 2):.2f} | "
          f"net/total torque median {np.median(np.abs(net) / tot):.2f} | net closing torque median {np.median(net):.1f} N m (share opening: {np.mean(net < 0):.2f}) | total |torque| median {np.median(tot):.1f} N m")
