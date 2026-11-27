"""How often does the plate-cylinder contact "pop" under a given collision setting? (log 2026-11-27)

Runs scripts/scripted_baseline.py unchanged (staged, cap 0.5, T2) for one seed, with mujoco.mj_step wrapped:
on the first call the named setting is applied to the model's options; every substep the deepest plate-cylinder
penetration is recorded. An episode "pops" if that penetration ever exceeds 5 mm (rest: ~0.1 mm). Prints pops,
spill and success, so settings can be compared on identical starts.

    python scripts/diagnose_timeouts/pop_rate.py <setting> <seed> [episodes]

Settings: baseline (primitive cylinders, multiccd on, native CCD), prism64 / prism32 (both collide as convex
prisms with that many sides, options as baseline), no_multiccd, libccd (native CCD off), ccd_strict (300 iterations,
tolerance 1e-9), ccd_iter100 (100 iterations), ccd_tol (tolerance 1e-9 only), libccd_no_multiccd.
"""
import runpy
import sys

sys.path.insert(0, ".")

import mujoco
import numpy as np

SETTINGS = {
    "baseline": lambda o: None,
    "no_multiccd": lambda o: setattr(o, "enableflags", o.enableflags & ~int(mujoco.mjtEnableBit.mjENBL_MULTICCD)),
    "libccd": lambda o: setattr(o, "disableflags", o.disableflags | int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)),
    "ccd_strict": lambda o: (setattr(o, "ccd_iterations", 300), setattr(o, "ccd_tolerance", 1e-9)),
    "ccd_iter100": lambda o: setattr(o, "ccd_iterations", 100),
    "ccd_tol": lambda o: setattr(o, "ccd_tolerance", 1e-9),
    "libccd_no_multiccd": lambda o: (setattr(o, "enableflags", o.enableflags & ~int(mujoco.mjtEnableBit.mjENBL_MULTICCD)),
                                     setattr(o, "disableflags", o.disableflags | int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD))),
}
# Collision shape, set before the env is built: number of prism sides (0 = primitive cylinders).
GEOMETRY = {"prism64": 64, "prism32": 32}
setting, seed = sys.argv[1], sys.argv[2]
import wam.close_cabinet as cc  # noqa: E402

cc.TRAY_PRISM_SIDES = GEOMETRY.get(setting, 0)
SETTINGS.update({name: SETTINGS["baseline"] for name in GEOMETRY})
episodes = sys.argv[3] if len(sys.argv) > 3 else "200"
original = mujoco.mj_step
deepest, ids = [], {}


reset = cc.CloseCabinetEnv.reset


def marked_reset(env, **kwargs):
    deepest.append(0.0)  # one entry per episode, its reset settle included
    return reset(env, **kwargs)


def step(model, data):
    if not ids:
        SETTINGS[setting](model.opt)
        ids.update(plate=model.geom("plate").id, obj=model.geom("tray_object").id)
    original(model, data)
    for i in range(data.ncon):
        c = data.contact[i]
        if {c.geom1, c.geom2} == {ids["plate"], ids["obj"]}:
            deepest[-1] = max(deepest[-1], -c.dist)


mujoco.mj_step = step
cc.CloseCabinetEnv.reset = marked_reset
out = f"/tmp/pop_{setting}_{seed}.npy"
sys.argv = ["scripted_baseline.py", "T2", seed, episodes, "0.5", "staged", out]
runpy.run_path("scripts/scripted_baseline.py", run_name="__main__")
rows = np.load(out)
d = np.array(deepest[: len(rows)])  # the run resets once more after its last episode
print(f"{setting} seed {seed}: pops {int((d > 0.005).sum())} / {len(d)} episodes | deepest penetration p50 {np.median(d) * 1000:.2f} mm, "
      f"p99 {np.percentile(d, 99) * 1000:.1f} mm | spill {rows[:, 2].mean():.3f} | success {rows[:, 0].mean():.3f}", flush=True)
