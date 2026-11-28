"""One-step parity of wam/warp_env.py against the C TaskEnv (milestone 3; criteria in log.md 2026-11-28).

The C env (MuJoCo >= 4) runs family F with G3's setup, driven by a trained G3 policy's mean action. At every
SNAPSHOT_EVERY-th step the full C state is copied into one Warp world, both take the same action, and the next
observation, reward terms, contact gate and termination are compared. Runs on the Warp device given.

    python scripts/warp/parity.py [episodes] [policy seed] [device] [task]
"""
import glob
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, ".")
from wam.close_cabinet import F_MIN, REACH_SCALE, W_REACH  # noqa: E402
from wam.tasks import TaskEnv  # noqa: E402
from wam.warp_env import WarpFamilyEnv, start_pool  # noqa: E402

episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 48
policy_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
task = sys.argv[4] if len(sys.argv) > 4 else "F"
SNAPSHOT_EVERY = 15

path = glob.glob(f"runs/WamFamilyContactFK-v0__train_ppo__{policy_seed}__*/train_ppo.cleanrl_model")[0]
weights = torch.load(path, map_location="cpu", weights_only=True)
actor = nn.Sequential(nn.Linear(46, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 4))
actor.load_state_dict({k[len("actor_mean."):]: v for k, v in weights.items() if k.startswith("actor_mean.")})
stats = np.load(Path(path).with_name("obs_rms.npz"))
w = stats["count"][:, None]
mean = (w * stats["mean"]).sum(0) / w.sum()
var = (w * (stats["var"] + (stats["mean"] - mean) ** 2)).sum(0) / w.sum()

c = TaskEnv(task=task, conditioning="contact_fk")
g, j = c.family_geoms, c.joint_ids[0]
snaps = []
for e in range(episodes):
    obs, _ = c.reset(seed=20_000 + e)
    for t in range(250):
        with torch.no_grad():
            a = actor(torch.tensor(np.clip((obs - mean) / np.sqrt(var + 1e-8), -10, 10), dtype=torch.float32)).clamp(-1, 1).numpy()
        snap = None
        if t % SNAPSHOT_EVERY == 0:
            m, d = c.model, c.data
            snap = dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), warm=d.qacc_warmstart.copy(), q_target=c.q_target.copy(),
                        last_action=c.last_action.copy(), goal=c.goal.copy(), closing=c.closing, valid=c.valid_closing,
                        start=c.start_distance, link=c.link_distance, fixture_pos=m.body_pos[c.fixture_body].copy(),
                        fixture_quat=m.body_quat[c.fixture_body].copy(), friction=m.geom_friction[c.object_geom].copy(),
                        geom_size=m.geom_size[g].copy(), geom_pos=m.geom_pos[g].copy(), geom_rbound=m.geom_rbound[g].copy(),
                        geom_aabb=m.geom_aabb[g].reshape(-1, 2, 3).copy(), jnt_axis=m.jnt_axis[j].copy(),
                        action=a.copy(), returns=dict(c.returns))
        obs, r, term, trunc, info = c.step(a)
        if snap is not None:
            f = [info[f"force/door_{g}"] for g in ("upper", "link4", "link5", "link6", "ee")] + [info["force/scene"]]
            snap.update(ctrl_after=c.data.ctrl.copy(), task_q_next=c.data.qpos[c.qpos_adr[0]], task_v_next=c.data.qvel[c.qvel_adr[0]],
                        obs_next=obs.copy(), reward=r, reach=c.returns["reach"] - snap["returns"]["reach"],
                        link_next=c.link_distance, gate=sum(f[1:4]) > F_MIN and f[0] + f[4] + f[5] == 0.0,
                        success=bool(info["task/success"]), spill=bool(info["task/spill"]), term=term)
            snaps.append(snap)
        if term or t == 249:
            break
n = len(snaps)
print(f"{n} snapshot states from {episodes} episodes")

wenv = WarpFamilyEnv(n, start_pool(4, seed=30_000, task=task), device=device, task=task)
wenv.reset()
T = lambda key: torch.as_tensor(np.array([s[key] for s in snaps]), dtype=torch.float32, device=wenv.device)
wenv.qpos.copy_(T("qpos")), wenv.qvel.copy_(T("qvel")), wenv.qacc_warmstart.copy_(T("warm"))
wenv.q_target, wenv.last_action, wenv.goal = T("q_target"), T("last_action"), T("goal")[:, 0]
wenv.ctrl.copy_(wenv.q_target)
wenv.closing, wenv.valid_closing, wenv.start_distance = T("closing"), T("valid"), T("start")
ids = wenv.ids
wenv.body_pos[:, ids["fixture"]], wenv.body_quat[:, ids["fixture"]] = T("fixture_pos"), T("fixture_quat")
wenv.geom_friction[:, ids["obj"]] = T("friction")
fg = wenv.family_geoms
wenv.geom_size[:, fg], wenv.geom_pos[:, fg] = T("geom_size"), T("geom_pos")
wenv.geom_rbound[:, fg], wenv.geom_aabb[:, fg] = T("geom_rbound"), T("geom_aabb")
wenv.jnt_axis[:, ids["task_j"]] = T("jnt_axis")
wenv._kinematics()
link_prev_w = wenv._link_distance().clone()
wenv.link_distance = link_prev_w.clone()
obs_w, r_w, term_w, trunc_w, info_w = wenv.step(T("action"))
link_next_w = wenv.link_distance.clone()  # replaced by the reset value for terminal worlds; compared on live worlds only

obs_c, r_c = T("obs_next"), T("reward")
term_c = torch.tensor([s["term"] for s in snaps], device=wenv.device)
reach_c = T("reach")
reach_w = W_REACH * (torch.exp(-link_next_w / REACH_SCALE) - torch.exp(-link_prev_w / REACH_SCALE))
live = ~term_c & ~term_w
d_obs = (obs_w - obs_c).abs()[live]
blocks = {"q": slice(0, 7), "qvel": slice(7, 14), "q_target - q": slice(14, 21), "last action": slice(21, 28),
          "tilt": slice(28, 29), "contact vector": slice(29, 32), "push direction": slice(32, 35), "distance": slice(35, 36),
          "contact rate": slice(36, 37), "fk positions": slice(37, 46)}
print("1. observation |Δ|, 99th percentile per block (live states):")
for name, s in blocks.items():
    print(f"   {name:16s} {torch.quantile(d_obs[:, s].flatten(), 0.99).item():.2e}  (max {d_obs[:, s].max().item():.2e})")
r_wo = (r_w - reach_w) - (r_c - reach_c)
print(f"2. reward without reach |Δ|: mean {r_wo.abs().mean().item():.4f}, p99 {torch.quantile(r_wo.abs(), 0.99).item():.4f}, max {r_wo.abs().max().item():.4f}")
f = info_w["forces"]
gate_w = (f[:, 1:4].sum(1) > F_MIN) & (f[:, 0] + f[:, 4] + f[:, 5] == 0)
gate_c = torch.tensor([s["gate"] for s in snaps], device=wenv.device)
print(f"3. contact gate agreement {(gate_w == gate_c).float().mean().item():.3f} (gate on in C: {gate_c.float().mean().item():.2f} of states)")
succ_c = torch.tensor([s["success"] for s in snaps], device=wenv.device)
spill_c = torch.tensor([s["spill"] for s in snaps], device=wenv.device)
agree = ((info_w["success"] == succ_c) & (info_w["spill"] == spill_c)).float().mean().item()
print(f"4. termination agreement {agree:.3f} (C: {int(succ_c.sum())} successes, {int(spill_c.sum())} spills)")
link_c = T("link_next")
err = (link_next_w - link_c)[live].abs()
print(f"5. reach distance |Δ| (live): median {err.median().item() * 1000:.1f} mm, p95 {torch.quantile(err, 0.95).item() * 1000:.1f} mm "
      f"| reach-term |Δ| mean {(reach_w - reach_c)[live].abs().mean().item():.4f}, p95 {torch.quantile((reach_w - reach_c)[live].abs(), 0.95).item():.4f}")

# Physics only: same states, same joint targets (C's ctrl after its step), 20 Warp substeps, no env code involved.
import mujoco_warp as mjw  # noqa: E402
from wam.close_cabinet import FRAME_SKIP  # noqa: E402
wenv.qpos.copy_(T("qpos")), wenv.qvel.copy_(T("qvel")), wenv.qacc_warmstart.copy_(T("warm"))
wenv.body_pos[:, ids["fixture"]], wenv.body_quat[:, ids["fixture"]] = T("fixture_pos"), T("fixture_quat")
wenv.geom_friction[:, ids["obj"]] = T("friction")
fg = wenv.family_geoms
wenv.geom_size[:, fg], wenv.geom_pos[:, fg] = T("geom_size"), T("geom_pos")
wenv.geom_rbound[:, fg], wenv.geom_aabb[:, fg] = T("geom_rbound"), T("geom_aabb")
wenv.jnt_axis[:, ids["task_j"]] = T("jnt_axis")
wenv.ctrl.copy_(T("ctrl_after"))
wenv._kinematics()  # places static geoms after the per-world changes
for _ in range(FRAME_SKIP):
    mjw.step(wenv.m, wenv.d)
dq = (wenv.qpos[:, ids["task_q"]] - T("task_q_next")).abs() / wenv.range
dv = (wenv.qvel[:, ids["task_v"]] - T("task_v_next")).abs() / wenv.range
env_dq = d_obs[:, 35]
print(f"6. physics only, box: normalised distance |Δ| p99 {torch.quantile(dq[live], 0.99).item():.2e} (env run: {torch.quantile(env_dq, 0.99).item():.2e}); "
      f"normalised rate |Δ| p99 {torch.quantile(dv[live], 0.99).item():.2e} (env run: {torch.quantile(d_obs[:, 36], 0.99).item():.2e})")

# Control: the same state copy into a fresh C MjData, stepped in C. If this reproduces C, the copy is complete and the
# gap above is Warp vs C numerics.
import mujoco  # noqa: E402
fresh = mujoco.MjData(c.model)
cq, cv = [], []
for s in snaps:
    c.model.body_pos[c.fixture_body], c.model.body_quat[c.fixture_body] = s["fixture_pos"], s["fixture_quat"]
    c.model.geom_friction[c.object_geom] = s["friction"]
    c.model.geom_size[g], c.model.geom_pos[g], c.model.geom_rbound[g] = s["geom_size"], s["geom_pos"], s["geom_rbound"]
    c.model.geom_aabb[g] = s["geom_aabb"].reshape(len(g), 6)
    c.model.jnt_axis[j] = s["jnt_axis"]
    mujoco.mj_resetData(c.model, fresh)
    fresh.qpos[:], fresh.qvel[:], fresh.qacc_warmstart[:], fresh.ctrl[:] = s["qpos"], s["qvel"], s["warm"], s["ctrl_after"]
    for _ in range(FRAME_SKIP):
        mujoco.mj_step(c.model, fresh)
    cq.append(fresh.qpos[c.qpos_adr[0]]), cv.append(fresh.qvel[c.qvel_adr[0]])
cdq = (torch.tensor(cq) - T("task_q_next").cpu()).abs() / wenv.range
cdv = (torch.tensor(cv) - T("task_v_next").cpu()).abs() / wenv.range
print(f"7. C-to-C copy, box: normalised distance |Δ| max {cdq.max().item():.2e}, rate |Δ| max {cdv.max().item():.2e}")
