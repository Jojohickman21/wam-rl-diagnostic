"""Physics checks shared by scripts/verify_env.py and tests/test_physics.py."""
import mujoco
import numpy as np

from wam.close_cabinet import ACTION_SCALE, CONTACT_GROUPS, FRAME_SKIP, SUCCESS_ANGLE

SWEEP_RATE = 0.25  # rad/s of base yaw toward the cabinet
SWEEP_STEPS = 190
SETTLE_STEPS = 40


def run_random(env, n_actions, seed=0):
    """Bang-bang actions held for random durations: every joint at full rate, into every obstacle."""
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    stats = {"nonfinite": 0, "max_penetration": 0.0, "episodes": 1}
    step = 0
    while step < n_actions:
        action = rng.choice([-1.0, 1.0], 7).astype(np.float32)
        for _ in range(rng.integers(1, 26)):
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            stats["nonfinite"] += int(not (np.all(np.isfinite(obs)) and np.isfinite(reward)))
            stats["max_penetration"] = max(stats["max_penetration"], info["task/penetration"])
            if terminated or truncated:
                env.reset()
                stats["episodes"] += 1
            if step == n_actions:
                break
    stats["solver_warnings"] = int(sum(w.number for w in env.unwrapped.data.warning))
    return stats


def run_sweep(env, seed=0, on_step=None):
    """Open-loop forearm sweep: constant base-yaw rate, every other joint target held at home."""
    env.reset(seed=seed)
    sweep = np.zeros(7, np.float32)
    sweep[0] = SWEEP_RATE * FRAME_SKIP * env.unwrapped.model.opt.timestep / ACTION_SCALE
    keys = [f"force/door_{g}" for g in CONTACT_GROUPS] + ["force/scene"]
    forces, angles = [], [env.unwrapped.data.qpos[7]]
    tilt = penetration = 0.0
    for step in range(SWEEP_STEPS + SETTLE_STEPS):
        _, _, _, _, info = env.step(sweep if step < SWEEP_STEPS else np.zeros(7, np.float32))
        forces.append([info[key] for key in keys])
        angles.append(info["task/door_angle"])
        tilt = max(tilt, info["task/plate_tilt"])
        penetration = max(penetration, info["task/penetration"])
        if on_step is not None:
            on_step()
    forces, angles = np.array(forces), np.array(angles)
    pushing = slice(np.argmax(np.diff(angles) < -1e-4), np.argmax(angles < SUCCESS_ANGLE))
    forearm = forces[pushing, keys.index("force/door_link5")]
    return {
        "door_start": angles[0],
        "door_end": angles[-1],
        "max_reopen": float(np.max(np.diff(angles))),
        "contact_fraction": float(np.mean(forearm > 0.5)),
        "median_push_force": float(np.median(forearm)),
        "peak_force": dict(zip(keys, forces.max(axis=0))),
        "max_plate_tilt_deg": float(np.degrees(tilt)),
        "max_penetration": penetration,
        "solver_warnings": int(sum(w.number for w in env.unwrapped.data.warning)),
    }


def capsule_coverage(model, body_name, tolerance):
    """Fraction of a link's visual mesh vertices within `tolerance` of its collision capsules."""
    body = model.body(body_name).id
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body]
    mesh = next(g for g in geoms if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH)
    mesh_id = model.geom_dataid[mesh]
    start = model.mesh_vertadr[mesh_id]
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.geom_quat[mesh])
    verts = model.mesh_vert[start : start + model.mesh_vertnum[mesh_id]] @ rot.reshape(3, 3).T
    verts = verts + model.geom_pos[mesh]

    gap = np.full(len(verts), np.inf)
    for g in geoms:
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_CAPSULE:
            continue
        mujoco.mju_quat2Mat(rot, model.geom_quat[g])
        axis = rot.reshape(3, 3)[:, 2]
        radius, half = model.geom_size[g, :2]
        rel = verts - model.geom_pos[g]
        along = np.clip(rel @ axis, -half, half)
        gap = np.minimum(gap, np.linalg.norm(rel - np.outer(along, axis), axis=1) - radius)
    return float(np.mean(gap <= tolerance))
