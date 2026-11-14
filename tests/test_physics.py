import gymnasium as gym
import numpy as np
import pytest

import wam  # noqa: F401  (registers WamCloseCabinet-v0)
from wam import checks


@pytest.fixture
def env():
    env = gym.make("WamCloseCabinet-v0")
    yield env
    env.close()


def test_random_actions_stay_finite_and_shallow(env):
    stats = checks.run_random(env, n_actions=10_000)
    assert stats["nonfinite"] == 0
    assert stats["solver_warnings"] == 0
    assert stats["max_penetration"] < 0.002


def test_forearm_sweep_closes_door_without_end_effector_contact(env):
    stats = checks.run_sweep(env)
    force = stats["peak_force"]
    assert force["force/door_link5"] > 0
    for other in ("upper", "link4", "link6", "ee"):
        assert force[f"force/door_{other}"] == 0
    assert force["force/scene"] == 0
    assert stats["door_start"] > 1.0
    assert stats["door_end"] < 0.05
    assert stats["max_reopen"] < 0.01
    assert stats["contact_fraction"] > 0.9
    assert stats["max_plate_tilt_deg"] < 25
    assert stats["max_penetration"] < 0.002
    assert stats["solver_warnings"] == 0


def test_reset_starts_out_of_contact(env):
    for seed in range(20):
        _, info = env.reset(seed=seed)
        assert env.unwrapped.data.ncon == 0
        assert 1.0 <= info["task/door_angle"] <= 1.4


def test_observation_matches_declared_space(env):
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape == (30,)
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    assert obs.dtype == np.float32
    assert reward == 0.0 and not terminated and not truncated


@pytest.mark.parametrize("link", ["link4", "link5", "link6"])
def test_contact_link_capsules_cover_visual_mesh(env, link):
    assert checks.capsule_coverage(env.unwrapped.model, link, tolerance=0.005) > 0.89
