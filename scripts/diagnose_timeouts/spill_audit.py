"""1 kHz spill audit of the staged scripted push on T2: what makes the cylinder slip?

Runs scripts/scripted_baseline.py unchanged (staged mode, cap 0.5, seed s, 200 episodes) with mujoco.mj_step
wrapped, so every 1 ms physics substep is logged. Per substep: the specific force (acceleration minus gravity,
as an accelerometer reads it) at the cylinder's resting point on the plate, split into the part pressing
into the plate and the part along it; required friction μ_req = along / pressing; the cylinder's slip speed
relative to that point; plate tilt; arm contact force on the drawer, the chest, and anything else; and
whether the controller target changed this substep (an action update, every 20 ms). Also, from the
simulator only (not observable on hardware), the friction the plate actually applies to the cylinder
as a share of its limit μ·N: 1.0 means the cylinder is at the edge of sliding.

Measurement notes (checked 2026-11-27 on staged T2 episodes): mj_objectAcceleration read after mj_step mixes
the step's accelerations with the next state's kinematics and reads ~81 m/s² during drawer contact where the
finite difference of the plate velocity reads ~20; it is not used. The finite difference itself jitters
(median |specific force| 17–20 m/s² against 9.8 at rest, integrator noise); averaged over 10 ms it is ~10.1.
So μ_req uses the finite difference of the resting point's velocity, minus gravity, averaged over FILTER ms
(100 Hz bandwidth, 5x the control rate). The cylinder never lifts more than 2.7 mm off the plate.

For each episode that ended in a spill (from the run's own outcomes), slip onset t0 is the start of the
last continuous stretch with slip speed above SLIP (0.02 m/s). Over the 100 ms before t0 it labels the
cause, first rule that matches:
  startup       slip begins in the first 100 ms of the episode, before the cylinder has ever been pressed
                onto the plate (the env starts it 1 mm above the plate; the arm moves from the first step)
  contact_pop   the plate-cylinder contact reports more than POP (5 mm) of penetration in the window: a
                collision-detection glitch (the depth appears in one 1 ms substep with the arm nearly still;
                the solver then ejects the cylinder and kicks every joint)
  object_hit    something other than the plate (drawer, chest, an arm link) touches the cylinder in the window
  tilt          tan(tilt) alone reaches 0.9 μ at t0: the plate is tilted near the slide angle
  impact        an arm contact force on anything rises by > 20 N within 5 ms, and μ_req first exceeds μ
                within 15 ms after it
  action_step   μ_req first exceeds μ within 3 ms of an action update, no impact
  sustained     μ_req > μ for >= 20 ms in a row (smooth acceleration of the arm)
  brief         μ_req > μ, but under 20 ms and not tied to an impact or action update
  no_exceed     μ_req never exceeds μ in the window (slip not explained by the plate's motion)

    python scripts/diagnose_timeouts/spill_audit.py <seed> <out.npz> [slew]
    python scripts/diagnose_timeouts/spill_audit.py --summary <out*.npz>
"""
import runpy
import sys

import mujoco
import numpy as np

sys.path.insert(0, ".")
import wam.close_cabinet as cc  # noqa: E402
from wam.close_cabinet import OBJECT_HALF_HEIGHT, PLATE_HALF_THICKNESS  # noqa: E402

SLIP, WINDOW = 0.02, 100  # m/s; substeps (1 ms each)
EPISODES = int(__import__("os").environ.get("EPISODES", 200))  # fewer for a smoke test
LABELS = ("startup", "contact_pop", "object_hit", "tilt", "impact", "action_step", "sustained", "brief", "no_exceed")


def record(seed, out, slew=0.0):
    model_ids = {}
    episodes, current = [], []
    original = mujoco.mj_step
    vel = np.zeros(6)
    wrench = np.zeros(6)

    def logged_step(model, data):
        if in_reset[0]:  # the env's settle substeps before step 0 belong to no episode
            original(model, data)
            return
        if not model_ids:
            model_ids.update(plate=model.geom("plate").id, obj=model.geom("tray_object").id,
                             chest=model.geom("chest").id, link7=model.body("link7").id,
                             mover=set(g for g in range(model.ngeom) if model.body(model.geom_bodyid[g]).name == "mover"),
                             arm=set(g for g in range(model.ngeom) if model.body(model.geom_bodyid[g]).name.startswith("link")))
        if new_episode[0]:
            if current:
                episodes.append(np.array(current, dtype=np.float32))
                current.clear()
            new_episode[0] = False
        ctrl_before = data.ctrl.copy()
        prev = logged_step.ctrl
        logged_step.ctrl = ctrl_before
        original(model, data)
        ids = model_ids
        normal = data.xmat[ids["link7"]].reshape(3, 3)[:, 1]  # plate normal is link 7's y axis (a mesh geom's frame is not)
        up = normal if normal[2] >= 0 else -normal
        r = (PLATE_HALF_THICKNESS + OBJECT_HALF_HEIGHT) * up
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, ids["plate"], vel, 0)
        omega = vel[:3].copy()
        point_velocity = vel[3:] + np.cross(omega, r)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, ids["obj"], vel, 0)
        rel = vel[3:] - point_velocity
        slip = np.linalg.norm(rel - (rel @ up) * up)
        forces = np.zeros(3)  # drawer, chest, other
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if ids["obj"] in (g1, g2) or not ((g1 in ids["arm"]) ^ (g2 in ids["arm"])):
                continue
            other = g2 if g1 in ids["arm"] else g1
            mujoco.mj_contactForce(model, data, i, wrench)
            forces[0 if other in ids["mover"] else 1 if other == ids["chest"] else 2] += wrench[0]
        held = np.zeros(3)  # plate's force on the cylinder, world frame
        mu_contact = 0.0
        hit = np.zeros(4)  # normal force on the cylinder from: drawer, chest, arm (not the plate), anything else
        depth = 0.0  # deepest plate-cylinder penetration this substep, m (positive = overlapping)
        for i in range(data.ncon):
            c = data.contact[i]
            if {c.geom1, c.geom2} == {ids["obj"], ids["plate"]}:
                depth = max(depth, -c.dist)
                mujoco.mj_contactForce(model, data, i, wrench)
                held += c.frame.reshape(3, 3).T @ wrench[:3] * (1.0 if c.geom1 == ids["plate"] else -1.0)
                mu_contact = c.friction[0]
            elif ids["obj"] in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == ids["obj"] else c.geom1
                mujoco.mj_contactForce(model, data, i, wrench)
                hit[0 if other in ids["mover"] else 1 if other == ids["chest"] else 2 if other in ids["arm"] else 3] += wrench[0]
        normal = held @ up
        util = np.linalg.norm(held - normal * up) / (mu_contact * normal) if normal > 1e-6 and mu_contact else 0.0
        tilt = np.arccos(np.clip(abs(data.xmat[ids["link7"]].reshape(3, 3)[2, 1]), 0.0, 1.0))
        update = float(prev is not None and np.abs(ctrl_before - prev).max() > 0)
        offset = data.geom_xpos[ids["obj"]] - data.geom_xpos[ids["plate"]]
        radial = np.linalg.norm(offset - (offset @ up) * up)
        current.append((*point_velocity, *up, slip, tilt, *forces, update, model.geom_friction[ids["obj"], 0],
                        np.linalg.norm(omega), util, *hit, radial, offset @ up, depth))

    logged_step.ctrl = None
    in_reset, new_episode = [False], [False]
    reset = cc.CloseCabinetEnv.reset

    def marked_reset(env, **kwargs):
        in_reset[0] = True
        out = reset(env, **kwargs)
        in_reset[0], new_episode[0] = False, True
        logged_step.ctrl = None
        return out

    cc.CloseCabinetEnv.reset = marked_reset
    mujoco.mj_step = logged_step
    outcomes = f"{out}.outcomes.npy"
    sys.argv = ["scripted_baseline.py", "T2", str(seed), str(EPISODES), "0.5", "staged", outcomes, str(slew)]
    runpy.run_path("scripts/scripted_baseline.py", run_name="__main__")
    mujoco.mj_step = original
    if current:
        episodes.append(np.array(current, dtype=np.float32))
    rows = np.load(outcomes)
    assert len(episodes) == len(rows), (len(episodes), len(rows))
    spills = [e for e, row in zip(episodes, rows) if row[2] > 0]
    controls = [e for e, row in zip(episodes, rows) if row[0] > 0][: len(spills)]
    np.savez(out, *spills, n_spills=len(spills), n=len(rows))
    np.savez(f"{out}.success_controls.npz", *controls)


COLUMNS = dict(slip=6, tilt=7, f_drawer=8, f_chest=9, f_other=10, update=11, mu=12, omega=13, util=14,
               hit_drawer=15, hit_chest=16, hit_arm=17, hit_other=18, radial=19, height=20, depth=21)
POP = 0.005  # m; a plate-cylinder penetration deeper than this is a collision-detection glitch (rest: < 1 mm)
FILTER = 10  # ms; causal moving average of the specific force before μ_req


def mu_required(e, k=FILTER):
    """μ_req per substep from the specific force (finite difference of the resting point's velocity, minus
    gravity) averaged over the last k substeps, in the current plate frame."""
    kernel = np.ones(k) / k
    accel = np.vstack([np.zeros(3), np.diff(e[:, 0:3], axis=0) / 0.001])
    specific = accel + np.array([0.0, 0.0, 9.81])
    specific = np.stack([np.convolve(specific[:, i], kernel)[: len(e)] for i in range(3)], 1)
    up = e[:, 3:6]
    pressing = np.einsum("ij,ij->i", specific, up)
    along = np.linalg.norm(specific - pressing[:, None] * up, axis=1)
    return along / np.maximum(pressing, 1e-3)


def classify(e):
    c = COLUMNS
    e = np.column_stack([e, mu_required(e)])
    c = dict(c, mu_req=e.shape[1] - 1)
    slipping = e[:, c["slip"]] > SLIP
    end = len(e)
    t0 = end - 1
    while t0 > 0 and slipping[t0 - 1]:
        t0 -= 1
    lo = max(t0 - WINDOW, 0)
    w = e[lo:t0 + 1]
    mu = float(e[0, c["mu"]])
    over = w[:, c["mu_req"]] > mu
    force = w[:, c["f_drawer"]] + w[:, c["f_chest"]] + w[:, c["f_other"]]
    rise = np.array([force[i] - force[max(i - 5, 0)] for i in range(len(w))])
    longest = run = 0
    for flag in over:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    first = int(np.argmax(over)) if over.any() else None
    info = dict(t0=t0, steps_before_end=end - t0, mu=mu, peak_ratio=float(w[:, c["mu_req"]].max() / mu),
                longest_ms=longest, peak_force=float(force.max()), peak_rise=float(rise.max()),
                tilt_ratio=float(np.tan(e[t0, c["tilt"]]) / mu), in_contact=bool(w[-1, c["f_drawer"]] > 0),
                chest=bool(w[:, c["f_chest"]].max() > 0), omega_peak=float(w[:, c["omega"]].max()),
                util_peak=float(w[:, c["util"]].max()), settled=bool((e[:t0, c["util"]] > 0).any()),
                hit=float(w[:, [c["hit_drawer"], c["hit_chest"], c["hit_arm"], c["hit_other"]]].max()))
    info["pop"] = float(w[:, c["depth"]].max())
    if t0 < 100 and not info["settled"]:
        label = "startup"
    elif info["pop"] > POP:
        label = "contact_pop"
    elif info["hit"] > 0:
        label = "object_hit"
    elif info["tilt_ratio"] >= 0.9:
        label = "tilt"
    elif first is not None and (rise[max(first - 15, 0): first + 1] > 20).any():
        label = "impact"
    elif first is not None and w[max(first - 3, 0): first + 1, c["update"]].any():
        label = "action_step"
    elif longest >= 20:
        label = "sustained"
    elif first is not None:
        label = "brief"
    else:
        label = "no_exceed"
    return label, info


def summary(files):
    spills, n = [], 0
    controls = []
    for f in files:
        d = np.load(f)
        n += int(d["n"])
        spills += [d[f"arr_{i}"] for i in range(int(d["n_spills"]))]
        c = np.load(f.replace(".npz", ".npz.success_controls.npz") if not f.endswith(".success_controls.npz") else f)
        controls += [c[k] for k in c.files]
    rows = [classify(e) for e in spills]
    print(f"{len(spills)} spills in {n} episodes ({len(spills) / n:.3f}); slip onset after episode start, ms: "
          f"quartiles {np.percentile([i['t0'] for _, i in rows], [25, 50, 75]).round(0)}")
    for label in LABELS:
        sub = [i for l, i in rows if l == label]
        if not sub:
            continue
        print(f"  {label:12s} {len(sub):3d} ({len(sub) / len(rows):.2f}) | peak μ_req/μ median {np.median([i['peak_ratio'] for i in sub]):.2f} "
              f"| longest μ_req>μ median {np.median([i['longest_ms'] for i in sub]):.0f} ms | tilt tan/μ median {np.median([i['tilt_ratio'] for i in sub]):.2f} "
              f"| arm on drawer at onset {np.mean([i['in_contact'] for i in sub]):.2f} | chest touched in window {np.mean([i['chest'] for i in sub]):.2f} "
              f"| peak contact rise median {np.median([i['peak_rise'] for i in sub]):.0f} N/5 ms | slide angle median {np.degrees(np.arctan(np.median([i['mu'] for i in sub]))):.0f}°"
              f" | contact friction use peak median {np.median([i['util_peak'] for i in sub]):.2f}")
    ratio = [np.max(mu_required(c)[FILTER:] / c[0, COLUMNS["mu"]]) for c in controls]
    longest = []
    for c in controls:
        over = mu_required(c)[FILTER:] > c[0, COLUMNS["mu"]]
        best = run = 0
        for flag in over:
            run = run + 1 if flag else 0
            best = max(best, run)
        longest.append(best)
    print(f"  deepest plate-cylinder penetration over whole successful episodes: median {np.median([c[:, COLUMNS['depth']].max() for c in controls]) * 1000:.1f} mm, "
          f"share with a pop (> {POP * 1000:.0f} mm) {np.mean([c[:, COLUMNS['depth']].max() > POP for c in controls]):.2f}")
    print(f"  reference, {len(controls)} successful episodes (same runs): peak μ_req/μ median {np.median(ratio):.2f}, "
          f"episodes with any μ_req > μ {np.mean(np.array(ratio) > 1):.2f}, longest stretch median {np.median(longest):.0f} ms, "
          f"contact friction use peak median {np.median([c[:, COLUMNS['util']].max() for c in controls]):.2f}")
    return rows


if __name__ == "__main__":
    if sys.argv[1] == "--summary":
        summary(sys.argv[2:])
    else:
        record(int(sys.argv[1]), sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.0)
