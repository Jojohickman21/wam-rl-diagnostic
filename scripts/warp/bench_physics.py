"""GPU throughput of the v2 task physics in MuJoCo Warp: milestone 1 of the port (log.md 2026-11-28).

Builds the C TaskEnv for <task> (needs MuJoCo 4.x, as Warp does), resets it once, and copies that state to <nworld>
worlds. Each control step writes new joint targets (a random walk of up to ACTION_SCALE per step, clipped to the
position limits; no wrist leveler yet) and runs FRAME_SKIP physics steps, captured once as a CUDA graph. Reports
control steps per second summed over worlds, peak contacts and constraints per world, overflows, and whether
every world stayed finite.

    python scripts/warp/bench_physics.py <task> <nworld> [control steps] [nconmax] [njmax]
"""
import sys
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from mujoco_warp._src import collision_convex

# Warp's EPA horizon buffer (24 edges) overflows on ~5% of worlds with the cylinder tray contacts, which can corrupt
# contact depth (the v1 "pops" were a depth error). The kernel bounds by the array's size, so allocate it larger.
EPA_HORIZON = 64
collision_convex.MJ_MAX_EPAHORIZON = EPA_HORIZON

sys.path.insert(0, ".")
from wam.close_cabinet import ACTION_SCALE, FRAME_SKIP  # noqa: E402
from wam.tasks import TaskEnv  # noqa: E402

task, nworld = sys.argv[1], int(sys.argv[2])
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 100
nconmax = int(sys.argv[4]) if len(sys.argv) > 4 else 48
njmax = int(sys.argv[5]) if len(sys.argv) > 5 else 256

env = TaskEnv(task=task, conditioning="contact")
env.reset(seed=1)
m = env.model
mw = mjw.put_model(m)
dw = mjw.put_data(m, env.data, nworld=nworld, nconmax=nconmax, njmax=njmax)

rng = np.random.default_rng(0)
lo, hi = env.ctrl_range[:, 0], env.ctrl_range[:, 1]
target = np.tile(env.q_target, (nworld, 1))

with wp.ScopedCapture() as capture:
    for _ in range(FRAME_SKIP):
        mjw.step(mw, dw)
graph = capture.graph


def control_step():
    global target
    target = np.clip(target + ACTION_SCALE * rng.uniform(-1, 1, target.shape), lo, hi)
    dw.ctrl.assign(target.astype(np.float32))
    wp.capture_launch(graph)


for _ in range(5):  # warm-up
    control_step()
wp.synchronize()
peak_con = peak_efc = 0
t0 = time.time()
for i in range(steps):
    control_step()
    if i % 10 == 0:
        peak_con = max(peak_con, int(dw.nacon.numpy()[0]) / nworld)
        peak_efc = max(peak_efc, int(dw.nefc.numpy().max()))
wp.synchronize()
elapsed = time.time() - t0
qpos = dw.qpos.numpy()
overflow = dw.overflow.numpy()
flags = {f.name: int(np.count_nonzero(overflow & f.value)) for f in mjw.OverflowType
         if f.value and not (f.value & (f.value - 1)) and np.any(overflow & f.value)} if hasattr(mjw, "OverflowType") else {}
print(f"{task} nworld {nworld}: {nworld * steps / elapsed:,.0f} control steps/s ({nworld * steps * FRAME_SKIP / elapsed:,.0f} physics steps/s) "
      f"| {elapsed / steps * 1000:.1f} ms per control step | contacts per world, mean at peak {peak_con:.1f} (nconmax {nconmax}) "
      f"| constraints per world, peak {peak_efc} (njmax {njmax}) | worlds with overflow {int(np.count_nonzero(overflow))} {flags} "
      f"| all finite {bool(np.all(np.isfinite(qpos)))} | device {wp.get_device()}", flush=True)
