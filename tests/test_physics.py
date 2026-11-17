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
    assert obs.shape == env.observation_space.shape == (34,)
    obs, reward, _, _, _ = env.step(env.action_space.sample())
    assert obs.dtype == np.float32
    assert np.isfinite(reward)


def test_forearm_sweep_earns_full_progress_and_success_bonus(env):
    stats = checks.run_sweep(env)
    returns = stats["returns"]
    assert returns["reward/progress"] == pytest.approx(50 * (stats["door_start"] - stats["door_end"]), rel=0.02)
    assert returns["reward/success"] == pytest.approx(20, rel=0.02)
    assert 3.5 < returns["reward/reach"] < 4.5
    assert returns["reward/wrong"] == 0 and returns["reward/spill"] == 0
    assert returns["reward/smooth"] > -5


@pytest.mark.parametrize("env_id,pays", [("WamCloseCabinet-v0", False), ("WamCloseCabinetUngated-v0", True)])
def test_door_closing_without_arm_contact_pays_only_when_ungated(env_id, pays):
    env = gym.make(env_id)
    env.reset(seed=0)
    env.unwrapped.data.qvel[7] = -3.0
    _, _, _, _, info = env.step(np.zeros(7, np.float32))
    assert info["task/door_angle"] < env.unwrapped.start_angle - 0.01
    assert (info["reward/progress"] > 0) == pays
    assert info["task/valid_fraction"] == 0
    env.close()


@pytest.mark.parametrize("link", ["link4", "link5", "link6"])
def test_contact_link_capsules_cover_visual_mesh(env, link):
    assert checks.capsule_coverage(env.unwrapped.model, link, tolerance=0.005) > 0.89


@pytest.fixture
def random_start_env():
    env = gym.make("WamCloseCabinetRandomStart-v0")
    yield env
    env.close()


def test_random_start_is_wide_level_and_out_of_contact(random_start_env):
    env = random_start_env.unwrapped
    starts = []
    for seed in range(50):
        random_start_env.reset(seed=seed)
        assert env.data.ncon == 0
        assert np.degrees(env._plate_tilt()) < 2
        assert env._outside_door()
        starts.append(env.data.qpos[:4].copy())
    assert np.all(np.ptp(starts, axis=0) > 0.8)


def test_scripted_sweep_solves_most_random_starts(random_start_env):
    results = [checks.run_sweep(random_start_env, seed=seed) for seed in range(60)]
    assert np.mean([r["success"] for r in results]) >= 0.85
    assert all(r["wrong_contact_steps"] == 0 for r in results)


def test_random_cabinet_pose_varies_and_is_observed():
    env = gym.make("WamCloseCabinetRandomPose-v0")
    cabinet = []
    for seed in range(30):
        obs, _ = env.reset(seed=seed)
        assert env.unwrapped.data.ncon == 0 and env.unwrapped._outside_door()
        cabinet.append(obs[-4:])
    cabinet = np.array(cabinet)
    assert np.all(np.ptp(cabinet[:, :2], axis=0) > 0.15)
    assert np.ptp(np.arctan2(cabinet[:, 3], cabinet[:, 2])) > np.radians(30)
    env.close()


def test_level_gate_withholds_progress_while_plate_is_tilted():
    results = {}
    for env_id in ("WamCloseCabinetRandomPoseBarrier-v0", "WamCloseCabinetRandomPoseCoupled-v0"):
        env = gym.make(env_id)
        env.reset(seed=0)
        env.unwrapped.data.qpos[5] += 0.6
        env.unwrapped.data.qvel[7] = -3.0
        _, _, _, _, info = env.step(np.zeros(7, np.float32))
        assert np.degrees(info["task/plate_tilt"]) > 10
        assert info["task/door_angle"] < env.unwrapped.start_angle - 0.01
        results[env_id] = info["reward/progress"]
        env.close()
    assert results["WamCloseCabinetRandomPoseBarrier-v0"] > 0
    assert results["WamCloseCabinetRandomPoseCoupled-v0"] == 0
