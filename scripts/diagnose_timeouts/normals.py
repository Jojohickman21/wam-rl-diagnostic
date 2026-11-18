"""For A22 timeouts: direction of forearm-door contact force during the last 50 steps.
Reports |n . hinge axis| (1 = pushing along the hinge axis, no closing torque) and the closing
torque the normal force would produce. Usage: normals.py <env id> <seed> <record steps>"""
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
def contacts():
    out = []
    for i in range(u.data.ncon):
        c = u.data.contact[i]
        if u.door_geom not in (c.geom1, c.geom2): continue
        other = c.geom2 if c.geom1 == u.door_geom else c.geom1
        if u.geom_group[other] not in (1, 2, 3): continue
        mujoco.mj_contactForce(u.model, u.data, i, wrench)
        normal = c.frame[:3]; axis = u.data.xaxis[door_jnt]; r = c.pos - u.data.xanchor[door_jnt]
        torque = np.cross(r, wrench[0] * normal) @ axis  # sign depends on contact orientation; magnitude is what matters
        out.append((wrench[0], abs(normal @ axis), abs(torque), c.pos[2] - u.data.xpos[u.door_body][2]))
    return out
obs, _ = env.reset(seed=20_000 + seed); steps = 0; ep = []; res = {"timeout": [], "success": []}
while steps < RECORD:
    with torch.no_grad(): a, *_ = agent.get_action_and_value(torch.tensor(obs, dtype=torch.float32)[None])
    obs, _, term, trunc, info = env.step(a[0].numpy()); steps += 1
    ep.append(contacts())
    if term or trunc:
        key = "timeout" if trunc and not term else ("success" if info["task/success"] else None)
        if key: res[key].append([c for step in (ep[-50:] if key == "timeout" else ep) for c in step])
        ep = []; obs, _ = env.reset()
for key, eps in res.items():
    c = np.array([x for e in eps for x in e if x[0] > 1.0])
    if not len(c): print(key, "no contacts"); continue
    print(f"{key}: {len(eps)} episodes, {len(c)} contact samples >1N | |normal . hinge axis| pct 50/90: {np.round(np.percentile(c[:,1], [50, 90]), 2)} | "
          f"force-weighted share along hinge axis: {np.sum(c[:,0]*c[:,1])/np.sum(c[:,0]):.2f} | |closing torque| median {np.median(c[:,2]):.1f} N m | contact height above door origin median {np.median(c[:,3]):.3f} m")
