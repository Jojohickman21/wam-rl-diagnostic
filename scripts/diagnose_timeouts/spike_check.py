"""Are plate impulses on the tray object a 50 Hz control-update artifact? (log.md 2026-11-27, before any env v3)

Runs scripts/scripted_baseline.py unchanged (staged, cap 0.5, slew 0.1, T2) for one seed with mujoco.mj_step wrapped
(reset settle excluded). Per physics substep k (0 = first substep after the first action; the target changes at
k % 20 == 0): the plate's normal force Fn and tangential force Ft on the cylinder, and its slip speed. On substeps
with Fn > SPIKE (3 mg), also ctrl - qpos and actuator force per joint. Saves per episode with the run's outcomes.

    python scripts/diagnose_timeouts/spike_check.py <seed> <out.npz>
    python scripts/diagnose_timeouts/spike_check.py --summary <out*.npz>
"""
import runpy
import sys

import mujoco
import numpy as np

sys.path.insert(0, ".")
import wam.close_cabinet as cc  # noqa: E402
from wam.close_cabinet import OBJECT_MASS  # noqa: E402

G = 9.81
SPIKE = 3 * OBJECT_MASS * G
SLIP = 0.02


def record(seed, out):
    episodes, spikes, cur, cur_spikes = [], [], [], []
    ids, wrench, vel = {}, np.zeros(6), np.zeros(6)
    state = {"reset": False, "k": 0}
    original, reset = mujoco.mj_step, cc.CloseCabinetEnv.reset

    def marked_reset(env, **kwargs):
        if cur:
            episodes.append(np.array(cur, np.float32))
            spikes.append(np.array(cur_spikes, np.float32).reshape(-1, 15))
            cur.clear(), cur_spikes.clear()
        state["reset"] = True
        result = reset(env, **kwargs)
        state["reset"], state["k"] = False, 0
        return result

    def step(model, data):
        original(model, data)
        if state["reset"]:
            return
        if not ids:
            ids.update(plate=model.geom("plate").id, obj=model.geom("tray_object").id, link7=model.body("link7").id,
                       qvel=model.jnt_dofadr[model.body("tray_object").jntadr[0]])
        normal = data.xmat[ids["link7"]].reshape(3, 3)[:, 1]
        up = normal if normal[2] >= 0 else -normal
        held = np.zeros(3)
        for i in range(data.ncon):
            c = data.contact[i]
            if {c.geom1, c.geom2} == {ids["plate"], ids["obj"]}:
                mujoco.mj_contactForce(model, data, i, wrench)
                held += c.frame.reshape(3, 3).T @ wrench[:3] * (1.0 if c.geom1 == ids["plate"] else -1.0)
        fn = held @ up
        ft = np.linalg.norm(held - fn * up)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, ids["plate"], vel, 0)
        r = (cc.PLATE_HALF_THICKNESS + cc.OBJECT_HALF_HEIGHT) * up
        point = vel[3:] + np.cross(vel[:3], r)
        obj = data.qvel[ids["qvel"]: ids["qvel"] + 3]
        rel = obj - point
        slip = np.linalg.norm(rel - (rel @ up) * up)
        k = state["k"]
        cur.append((k, fn, ft, slip, model.geom_friction[ids["obj"], 0]))
        if fn > SPIKE:
            cur_spikes.append((k, *(data.ctrl[:7] - data.qpos[:7]), *data.actuator_force[:7]))
        state["k"] = k + 1

    cc.CloseCabinetEnv.reset, mujoco.mj_step = marked_reset, step
    outcomes = f"{out}.outcomes.npy"
    sys.argv = ["scripted_baseline.py", "T2", str(seed), "200", "0.5", "staged", outcomes, "0.1"]
    runpy.run_path("scripts/scripted_baseline.py", run_name="__main__")
    mujoco.mj_step, cc.CloseCabinetEnv.reset = original, reset
    rows = np.load(outcomes)
    episodes, spikes = episodes[: len(rows)], spikes[: len(rows)]  # the final reset opens no episode
    np.savez(out, rows=rows, **{f"e{i}": e for i, e in enumerate(episodes)},
             **{f"s{i}": s for i, s in enumerate(spikes)})


def onset(e):
    slipping = e[:, 3] > SLIP
    t = len(e) - 1
    while t > 0 and slipping[t - 1]:
        t -= 1
    return t


def summary(files):
    groups = {"early spills": [], "other spills": [], "successes": []}
    rng = np.random.default_rng(0)
    successes = []
    for f in files:
        d = np.load(f)
        rows = d["rows"]
        for i, row in enumerate(rows):
            e, s = d[f"e{i}"], d[f"s{i}"]
            if row[2] > 0:
                groups["early spills" if onset(e) < 100 else "other spills"].append((e, s))
            elif row[0] > 0:
                successes.append((e, s))
    groups["successes"] = [successes[i] for i in rng.choice(len(successes), 50, replace=False)]
    print(f"spike = Fn > {SPIKE:.1f} N (3 mg); control update at phase 0 (substep k % 20 == 0)")
    for name, eps in groups.items():
        steps = sum(len(e) for e, _ in eps)
        ks = np.concatenate([e[e[:, 1] > SPIKE, 0] for e, _ in eps]) if eps else np.zeros(0)
        phase = np.bincount((ks % 20).astype(int), minlength=20)
        at_update = ks[ks % 20 < 2]
        first = np.sum((at_update >= 20) & (at_update < 22))
        later = np.sum(at_update >= 40)
        k0 = np.sum(ks < 2)
        eps_with = np.mean([np.any(e[:, 1] > SPIKE) for e, _ in eps])
        print(f"\n{name}: {len(eps)} episodes, {steps} substeps, spike substeps {len(ks)} ({len(ks) / max(steps, 1) * 1000:.1f} per 1000), "
              f"episodes with any spike {eps_with:.2f}")
        print("  by phase 0..19: " + " ".join(str(int(x)) for x in phase))
        print(f"  at the first action (k 0-1): {int(k0)} | first update (k 20-21): {int(first)} | later updates (k >= 40, phase 0-1): {int(later)} "
              f"| mid-step (phase 2-19): {int(np.sum(ks % 20 >= 2))}")
        late_eps = [e for e, _ in eps if len(e) > 200]
        if late_eps:
            rate = np.mean([np.mean(e[100:, 1] > SPIKE) * 1000 for e in late_eps])
            print(f"  spike rate after the first 100 ms: {rate:.2f} per 1000 substeps (over {len(late_eps)} episodes)")
        sp = np.concatenate([s for _, s in eps if len(s)]) if any(len(s) for _, s in eps) else np.zeros((0, 15))
        if len(sp):
            print("  at spikes, median |ctrl - qpos| per joint (mrad): " + " ".join(f"{x * 1000:.2f}" for x in np.median(np.abs(sp[:, 1:8]), 0))
                  + " | median |actuator force| (N·m): " + " ".join(f"{x:.1f}" for x in np.median(np.abs(sp[:, 8:15]), 0)))
    quiet = np.concatenate([e[(e[:, 1] <= SPIKE) & (e[:, 0] > 300)] for e, _ in groups["successes"]])
    print(f"\nreference, successes away from spikes (k > 300): Fn median {np.median(quiet[:, 1]):.2f} N (mg = {OBJECT_MASS * G:.2f})")
    slide = []
    for e, _ in groups["early spills"]:
        t0 = onset(e)
        seg = e[t0 + 10:]
        seg = seg[seg[:, 3] > SLIP]
        slide.append(seg)
    seg = np.concatenate(slide)
    touching = seg[seg[:, 1] > 1e-6]
    print(f"post-landing slide in early spills ({len(seg)} sliding substeps): in plate contact {len(touching) / len(seg):.2f}; "
          f"while touching, (Ft/Fn)/μ median {np.median(touching[:, 2] / touching[:, 1] / touching[:, 4]):.2f}, Ft/Fn median {np.median(touching[:, 2] / touching[:, 1]):.3f}, Ft/(mg) median {np.median(touching[:, 2] / (OBJECT_MASS * G)):.3f}, "
          f"Fn/(mg) median {np.median(touching[:, 1] / (OBJECT_MASS * G)):.2f}; over all sliding substeps, mean Fn/(mg) {np.mean(seg[:, 1]) / (OBJECT_MASS * G):.2f}")


if __name__ == "__main__":
    if sys.argv[1] == "--summary":
        summary(sys.argv[2:])
    else:
        record(int(sys.argv[1]), sys.argv[2])
