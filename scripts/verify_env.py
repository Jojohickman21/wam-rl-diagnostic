"""Check the CloseCabinet env end to end and write artifacts/verify_scene.mp4.

    python scripts/verify_env.py

Exits non-zero if a physics check fails. tests/test_physics.py asserts the same checks.
"""
import os
import time
from pathlib import Path

import gymnasium as gym
import imageio
import mujoco
import numpy as np

import wam  # noqa: F401  (registers WamCloseCabinet-v0; sets MUJOCO_GL=egl on Linux)
from wam import checks

ARTIFACTS = Path(__file__).parents[1] / "artifacts"
DIAGNOSTIC_STEPS = 2_000_000


def main():
    env = gym.make("WamCloseCabinet-v0")

    stress = checks.run_random(env, n_actions=10_000)
    print(f"random actions: {stress}")
    assert stress["nonfinite"] == 0 and stress["solver_warnings"] == 0
    assert stress["max_penetration"] < 0.002

    model, data = env.unwrapped.model, env.unwrapped.data
    renderer = mujoco.Renderer(model, 480, 640)
    show_collision = mujoco.MjvOption()
    show_collision.geomgroup[3] = 1
    frames = []

    def grab():
        views = []
        for camera in ("overview", "top"):
            renderer.update_scene(data, camera, show_collision)
            views.append(renderer.render())
        frames.append(np.concatenate(views, axis=1))

    sweep = checks.run_sweep(env, on_step=grab)
    print(f"forearm sweep: {sweep}")
    force = sweep["peak_force"]
    assert force["force/door_link5"] > 0 and sweep["contact_fraction"] > 0.9
    assert all(force[f"force/door_{g}"] == 0 for g in ("upper", "link4", "link6", "ee"))
    assert sweep["door_end"] < 0.05 and sweep["max_penetration"] < 0.002

    ARTIFACTS.mkdir(exist_ok=True)
    video = ARTIFACTS / "verify_scene.mp4"
    imageio.mimsave(video, frames, fps=env.metadata["render_fps"])
    print(f"wrote {video} ({len(frames)} frames)")

    env.reset(seed=0)
    n = 5_000
    start = time.perf_counter()
    for _ in range(n):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()
    sps = n / (time.perf_counter() - start)
    cores = os.cpu_count()
    print(f"throughput: {sps:.0f} env steps/s on one core")
    print(
        f"{DIAGNOSTIC_STEPS:,} env steps: {DIAGNOSTIC_STEPS / sps / 60:.1f} min on one core, "
        f"{DIAGNOSTIC_STEPS / sps / cores / 60:.1f} min on {cores} cores if scaling is linear "
        "(simulation only; PPO updates add to this)"
    )


if __name__ == "__main__":
    main()
