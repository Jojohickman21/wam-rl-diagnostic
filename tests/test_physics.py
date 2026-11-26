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
    assert obs.shape == env.observation_space.shape == (35,)
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
        cabinet.append(obs[-5:-1])
    cabinet = np.array(cabinet)
    assert np.all(np.ptp(cabinet[:, :2], axis=0) > 0.15)
    assert np.ptp(np.arctan2(cabinet[:, 3], cabinet[:, 2])) > np.radians(30)
    env.close()


def test_level_gate_withholds_progress_while_plate_is_tilted():
    from wam.close_cabinet import CloseCabinetEnv

    results = {}
    for level_gate in (False, True):
        env = CloseCabinetEnv(gated=False, tilt_barrier=True, level_gate=level_gate)
        env.reset(seed=0)
        # Joint 5 tilts the plate about 17 degrees from home; joint 6 is vertical there.
        env.data.qpos[4] += 0.3
        env.q_target[4] += 0.3
        env.data.qvel[7] = -3.0
        _, _, _, _, info = env.step(np.zeros(7, np.float32))
        assert 10 < np.degrees(info["task/plate_tilt"]) < 25
        assert info["task/door_angle"] < env.start_angle - 0.01
        results[level_gate] = info["reward/progress"]
    assert results[False] > 0
    assert results[True] == 0


@pytest.fixture
def tray_env():
    env = gym.make("WamCloseCabinetRandomPoseTray-v0")
    yield env
    env.close()


def test_tray_object_starts_on_plate_with_observed_slide_angle(tray_env):
    env = tray_env.unwrapped
    angles = []
    for seed in range(30):
        obs, info = tray_env.reset(seed=seed)
        assert obs.shape == tray_env.observation_space.shape == (41,)
        assert env.data.ncon == 0 and not env._spilled()
        assert env.data.geom_xpos[env.plate_geom][2] > 0.15
        assert obs[-7] == pytest.approx(info["task/tilt_limit"])
        assert np.tan(obs[-7]) == pytest.approx(env.model.geom_friction[env.object_geom, 0], rel=1e-5)
        angles.append(np.degrees(obs[-7]))
    assert 15 <= min(angles) and max(angles) <= 35 and np.ptp(angles) > 10


@pytest.mark.parametrize("seed", [0, 1])
def test_tray_object_slides_at_its_drawn_angle(seed):
    from wam.close_cabinet import CloseCabinetEnv

    env = CloseCabinetEnv(tray_object=True)
    assert checks.hold_tray_at(env, seed, 0.95) < 1.0
    assert np.isinf(checks.hold_tray_at(env, seed, 1.05))


def test_tray_random_actions_stay_finite_and_shallow(tray_env):
    stats = checks.run_random(tray_env, n_actions=5_000)
    assert stats["nonfinite"] == 0
    assert stats["solver_warnings"] == 0
    # The plate strikes the loose 0.2 kg cylinder under bang-bang actions; arm-scene pairs stay
    # under 2 mm (verified 2026-11-17), the plate-cylinder pair reaches 2.8 mm.
    assert stats["max_penetration"] < 0.003


def test_tray_mode_has_no_tilt_penalty_and_spills_only_when_object_leaves(tray_env):
    env = tray_env.unwrapped
    tray_env.reset(seed=0)
    for _ in range(25):
        _, _, terminated, _, info = tray_env.step(np.zeros(7, np.float32))
    assert info["reward/tilt"] == 0.0 and not terminated
    env.data.qpos[env.object_qpos : env.object_qpos + 3] += [0.0, 0.0, -1.0]
    _, _, terminated, _, info = tray_env.step(np.zeros(7, np.float32))
    assert terminated and info["task/spill"] == 1.0 and info["reward/spill"] == -15.0


@pytest.fixture
def coupled_env():
    env = gym.make("WamCloseCabinetTrayFrictionCoupled-v0")
    yield env
    env.close()


def test_coupled_env_observes_only_what_hardware_measures(coupled_env):
    from wam.close_cabinet import TILT_MAX

    for seed in range(5):
        obs, info = coupled_env.reset(seed=seed)
        assert obs.shape == coupled_env.observation_space.shape == (35,)
        assert obs[-1] == pytest.approx(TILT_MAX)
        assert info["task/tilt_limit"] != pytest.approx(TILT_MAX)


def test_required_friction_is_tan_tilt_on_a_still_plate(coupled_env):
    env = coupled_env.unwrapped
    coupled_env.reset(seed=0)
    env.q_target[4] += np.radians(14)
    env.data.ctrl[:] = env.q_target
    for _ in range(60):
        _, _, _, _, info = coupled_env.step(np.zeros(7, np.float32))
    assert info["task/plate_tilt"] > np.radians(3)
    assert info["task/friction_required"] == pytest.approx(np.tan(info["task/plate_tilt"]), abs=0.01)


def test_friction_coupling_scales_door_progress():
    from wam.close_cabinet import CloseCabinetEnv

    progress = {}
    for jolt in (False, True):
        env = CloseCabinetEnv(gated=False, tray_object=True, observe_object=False, friction_coupling=True)
        env.reset(seed=0)
        env.data.qvel[7] = -3.0
        if jolt:
            # A previous-step velocity of 1 m/s sideways reads as a 50 m/s^2 jolt this step.
            env.object_point_velocity = env.object_point_velocity + np.array([1.0, 0.0, 0.0])
        _, _, _, _, info = env.step(np.zeros(7, np.float32))
        progress[jolt] = info["reward/progress"]
        assert info["task/door_angle"] < env.start_angle - 0.01
    assert progress[False] > 0
    assert 0 <= progress[True] < 0.01 * progress[False]


def test_energy_penalty_replaces_smoothness_terms(coupled_env):
    coupled_env.reset(seed=0)
    _, _, _, _, info = coupled_env.step(np.ones(7, np.float32))
    assert info["reward/smooth"] == 0.0
    assert info["reward/energy"] < 0.0


@pytest.mark.parametrize("env_id,size,slide_angle_seen,risk_seen", [
    ("WamCloseCabinetTrayCoupledSeeObject-v0", 41, False, False),
    ("WamCloseCabinetTrayCoupledSeeRisk-v0", 36, False, True),
    ("WamCloseCabinetTrayEnergy-v0", 41, True, False),
])
def test_separating_envs_observe_one_change_each(env_id, size, slide_angle_seen, risk_seen):
    from wam.close_cabinet import TILT_MAX

    env = gym.make(env_id)
    obs, info = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape == (size,)
    angle = obs[34]
    assert (angle == pytest.approx(info["task/tilt_limit"])) == slide_angle_seen
    assert (angle == pytest.approx(TILT_MAX)) != slide_angle_seen
    obs, _, _, _, info = env.step(np.zeros(7, np.float32))
    if risk_seen:
        assert obs[-1] == pytest.approx(info["task/friction_required"])
    env.close()


@pytest.mark.parametrize("env_id,actions,size,risk_seen", [
    ("WamCloseCabinetTrayWristLeveler-v0", 4, 35, False),
    ("WamCloseCabinetTraySeeRisk-v0", 7, 36, True),
])
def test_section7_envs_observe_only_what_hardware_measures(env_id, actions, size, risk_seen):
    from wam.close_cabinet import TILT_MAX

    env = gym.make(env_id)
    obs, info = env.reset(seed=1)
    assert env.action_space.shape == (actions,)
    assert obs.shape == env.observation_space.shape == (size,)
    assert obs[34] == pytest.approx(TILT_MAX)
    assert info["task/tilt_limit"] != pytest.approx(TILT_MAX)
    env.unwrapped.data.qvel[7] = -3.0
    obs, _, _, _, info = env.step(np.zeros(actions, np.float32))
    assert info["task/coupling_mean"] == 1.0
    assert info["reward/smooth"] == 0.0
    if risk_seen:
        assert obs[-1] == pytest.approx(info["task/friction_required"])
    env.close()


@pytest.fixture
def leveler_env():
    env = gym.make("WamCloseCabinetTrayWristLeveler-v0")
    yield env
    env.close()


def test_wrist_leveler_keeps_plate_level_under_random_arm_actions(leveler_env):
    rng = np.random.default_rng(0)
    leveler_env.reset(seed=0)
    tilts = []
    for _ in range(1_000):
        _, reward, terminated, truncated, info = leveler_env.step(rng.uniform(-1, 1, 4).astype(np.float32))
        assert np.isfinite(reward)
        tilts.append(info["task/plate_tilt"])
        if terminated or truncated:
            leveler_env.reset()
    # Measured 2026-11-18: 98.9% of 2,000 steps under 5 degrees, median 1.8.
    assert np.mean(np.degrees(tilts) < 5) >= 0.95


def test_wrist_leveler_moves_only_the_wrist_toward_level(leveler_env):
    env = leveler_env.unwrapped
    leveler_env.reset(seed=0)
    target = env.q_target.copy()
    leveler_env.step(np.array([1.0, 0, 0, 0], np.float32))
    assert env.q_target[0] == pytest.approx(target[0] + 0.04)
    assert np.allclose(env.q_target[1:4], target[1:4])
    assert np.abs(env.q_target[4:] - env.data.qpos[4:7]).max() <= 0.1 + 1e-9


def test_joint1_sweep_with_wrist_leveler_solves_some_random_pose_starts(leveler_env):
    results = [checks.run_sweep(leveler_env, seed=seed) for seed in range(30)]
    # Measured 2026-11-18: 13 of 30.
    assert np.mean([r["success"] for r in results]) >= 0.3


def test_joint_margin_is_observed_only_where_requested():
    from wam.close_cabinet import MARGIN_CAP, TILT_MAX

    plain = gym.make("WamCloseCabinetTrayWristLeveler-v0")
    env = gym.make("WamCloseCabinetTrayWristLevelerMargin-v0")
    obs, info = env.reset(seed=1)
    assert plain.reset(seed=1)[0].shape == (35,)
    assert env.action_space.shape == (4,)
    assert obs.shape == env.observation_space.shape == (38,)
    assert list(env.unwrapped.limited_joints) == [1, 3, 5]
    assert obs[34] == pytest.approx(TILT_MAX)
    assert info["task/tilt_limit"] != pytest.approx(TILT_MAX)
    assert np.all((obs[-3:] >= 0) & (obs[-3:] <= MARGIN_CAP))
    plain.close()
    env.close()


def test_joint_margin_falls_to_zero_at_the_elbow_stop():
    from wam.close_cabinet import MARGIN_CAP

    env = gym.make("WamCloseCabinetTrayWristLevelerMargin-v0")
    env.reset(seed=0)
    unwrapped = env.unwrapped
    high = unwrapped.ctrl_range[3, 1]
    for offset, expected in ((0.0, 0.0), (0.2, 0.2), (1.0, MARGIN_CAP)):
        unwrapped.data.qpos[3] = high - offset
        assert unwrapped._obs()[-2] == pytest.approx(expected, abs=1e-6)
    env.close()


@pytest.mark.parametrize("env_id,charged", [
    ("WamCloseCabinetTrayWristLevelerMargin-v0", False),
    ("WamCloseCabinetTrayWristLevelerLimitPenalty-v0", True),
])
def test_joint_limit_penalty_charges_only_near_a_stop(env_id, charged):
    from wam.close_cabinet import W_LIMIT

    env = gym.make(env_id)
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(4, np.float32))
    assert info["reward/limit"] == 0.0
    unwrapped = env.unwrapped
    # Hold joint 4 0.05 rad from its upper stop: depth 0.75, so 0.5625 * W_LIMIT this step.
    high = unwrapped.ctrl_range[3, 1]
    unwrapped.data.qpos[3] = unwrapped.q_target[3] = high - 0.05
    terms, _ = unwrapped._reward(np.zeros(7), np.zeros(6), 0.0)
    expected = -W_LIMIT * 0.75 ** 2 if charged else 0.0
    assert terms["limit"] == pytest.approx(expected, rel=1e-6)
    env.close()


def test_ee_first_starts_put_the_end_effector_nearer_the_door():
    from wam.close_cabinet import EE_LEAD

    env = gym.make("WamCloseCabinetEEFirst-v0")
    flags = []
    for seed in range(40):
        _, info = env.reset(seed=seed)
        unwrapped = env.unwrapped
        assert unwrapped.data.ncon == 0 and unwrapped._outside_door()
        if info["task/ee_first"]:
            assert unwrapped._ee_lead() > EE_LEAD
        flags.append(info["task/ee_first"])
    assert 0.3 < np.mean(flags) < 0.7
    env.close()


def test_ee_first_option_leaves_other_envs_untouched():
    env = gym.make("WamCloseCabinetRandomPose-v0")
    _, info = env.reset(seed=0)
    assert "task/ee_first" not in info
    # No end-effector-first start exists in 400 random-pose starts (measured 2026-11-18).
    assert env.unwrapped._ee_lead() < 0
    env.close()


@pytest.mark.parametrize("env_id,actions", [
    ("WamCloseCabinetTrayWristLevelerOutcome-v0", 4),
    ("WamCloseCabinetEEFirstOutcome-v0", 7),
])
def test_outcome_reward_pays_closing_but_not_shaping(env_id, actions):
    env = gym.make(env_id)
    env.reset(seed=0)
    # Door spun shut by hand: no arm contact, so the old gate would pay nothing.
    env.unwrapped.data.qvel[7] = -3.0
    _, _, _, _, info = env.step(np.zeros(actions, np.float32))
    assert info["task/door_angle"] < env.unwrapped.start_angle - 0.01
    assert info["reward/progress"] > 0
    assert info["task/valid_fraction"] == 0
    for _ in range(20):
        _, _, terminated, truncated, info = env.step(np.ones(actions, np.float32))
        if terminated or truncated:
            break
    assert info["reward/reach"] == 0.0 and info["reward/wrong"] == 0.0
    env.close()


def test_outcome_reward_success_bonus_needs_valid_closing():
    from wam.close_cabinet import CloseCabinetEnv

    env = CloseCabinetEnv(outcome_reward=True)
    env.reset(seed=0)
    env.data.qpos[7] = 0.06  # one small step from success, all of it without arm contact
    env.data.qvel[7] = -3.0
    _, _, terminated, _, info = env.step(np.zeros(7, np.float32))
    assert terminated and info["task/success"] == 1.0
    assert info["reward/success"] == 0.0


def test_outcome_reward_sweep_still_earns_full_success_bonus():
    env = gym.make("WamCloseCabinetEEFirstOutcome-v0")
    for seed in range(40):
        if not env.reset(seed=seed)[1]["task/ee_first"]:
            stats = checks.run_sweep(env, seed=seed)
            if stats["success"] and stats["valid_fraction"] > 0.95:
                assert stats["returns"]["reward/success"] == pytest.approx(20 * stats["valid_fraction"], rel=0.05)
                break
    else:
        pytest.fail("no valid forearm sweep among 40 starts")
    env.close()


@pytest.mark.parametrize("env_id", [
    "WamCloseCabinetEEFirstLeveler-v0",
    "WamCloseCabinetEEFirstLevelerUngated-v0",
    "WamCloseCabinetEEFirstLevelerOutcome-v0",
])
def test_ee_first_leveler_envs_hold_the_plate_and_start_hand_first(env_id):
    from wam.close_cabinet import EE_LEAD

    env = gym.make(env_id)
    rng = np.random.default_rng(0)
    flags, tilts = [], []
    for seed in range(20):
        obs, info = env.reset(seed=seed)
        assert env.action_space.shape == (4,) and obs.shape == (35,)
        if info["task/ee_first"]:
            assert env.unwrapped._ee_lead() > EE_LEAD
        flags.append(info["task/ee_first"])
        for _ in range(25):
            _, _, terminated, truncated, info = env.step(rng.uniform(-1, 1, 4).astype(np.float32))
            tilts.append(info["task/plate_tilt"])
            if terminated or truncated:
                break
    assert 0.2 < np.mean(flags) < 0.8
    assert np.mean(np.degrees(tilts) < 5) >= 0.95
    env.close()


def test_ee_start_pool_is_seeded_and_holds_only_valid_hand_first_starts():
    from wam.close_cabinet import EE_LEAD, CloseCabinetEnv

    kwargs = dict(start_spread=1.0, cabinet_spread=(0.1, np.radians(20)), ee_first_fraction=1.0,
                  wrist_leveler=True, leveler_tuck=0.2, ee_start_pool=8)
    a, b = CloseCabinetEnv(**kwargs), CloseCabinetEnv(**kwargs)
    a.reset(seed=3)
    b.reset(seed=3)
    assert len(a.ee_pool) == 8
    assert all(np.array_equal(x[3], y[3]) for x, y in zip(a.ee_pool, b.ee_pool))
    for _ in range(10):
        a.reset()
        assert a.data.ncon == 0 and a._outside_door() and a._ee_lead() > EE_LEAD


def test_task_t0_reproduces_a20_step_for_step():
    a20, t0 = gym.make("WamCloseCabinetTrayWristLeveler-v0"), gym.make("WamTaskT0-v0")
    a20.reset(seed=4)
    t0.reset(seed=4)
    rng = np.random.default_rng(4)
    for _ in range(150):
        action = rng.uniform(-1, 1, 4).astype(np.float32)
        _, reward_a, term_a, trunc_a, _ = a20.step(action)
        _, reward_b, term_b, trunc_b, _ = t0.step(action)
        assert np.array_equal(a20.unwrapped.data.qpos, t0.unwrapped.data.qpos)
        assert reward_a == reward_b and term_a == term_b and trunc_a == trunc_b
        if term_a or trunc_a:
            a20.reset()
            t0.reset()


@pytest.mark.parametrize("task", ["T1", "T2", "T3", "T4x", "T4xneg", "T4y", "T4xy"])
def test_task_envs_start_out_of_contact_with_descriptor(task):
    from wam.tasks import SLOT_SIZE, TASKS

    env = gym.make(f"WamTask{task}-v0")
    u = env.unwrapped
    for seed in range(5):
        obs, info = env.reset(seed=seed)
        assert obs.shape == env.observation_space.shape == (59,)
        assert env.action_space.shape == (4,)
        assert u.data.ncon == 0 and u._outside_door()
        slots = u.descriptor().reshape(2, SLOT_SIZE)
        assert slots[0, 0] == 1.0 and slots[0, 2] == 1.0  # slot 0 holds a slide joint
        assert slots[1, 0] == float(len(TASKS[task]["joints"]) == 2)
        assert info["task/distance"] > TASKS[task]["tol"]
    env.close()


@pytest.mark.parametrize("task", ["T1", "T2", "T4x"])
def test_task_movers_stop_within_a_few_control_steps_like_the_door(task):
    env = gym.make(f"WamTask{task}-v0")
    u = env.unwrapped
    env.reset(seed=0)
    u.data.qvel[u.qvel_adr[0]] = 0.5
    for _ in range(3):
        env.step(np.zeros(4, np.float32))
    assert abs(u.data.qvel[u.qvel_adr[0]]) < 0.05
    env.close()


def test_task_progress_pays_like_the_door_per_unit_of_range():
    from wam.close_cabinet import W_PROGRESS
    from wam.tasks import TaskEnv

    env = TaskEnv(task="T4x", gated=False)
    env.reset(seed=0)
    before = env._distance()
    env.data.qvel[env.qvel_adr[0]] = 0.3 * np.sign(env.goal[0])
    _, _, _, _, info = env.step(np.zeros(4, np.float32))
    closed = before - env._distance()
    assert closed > 0
    assert info["reward/progress"] == pytest.approx(W_PROGRESS * 1.2 / 0.25 * closed, rel=1e-6)


def test_multitask_env_draws_every_training_task_with_one_observation_shape():
    env = gym.make("WamMultiTask-v0")
    seen = set()
    obs, info = env.reset(seed=0)
    for _ in range(40):
        obs, info = env.reset()
        assert obs.shape == (59,)
        seen.add(int(info["task/index"]))
    assert seen == {0, 1, 2, 3}
    env.close()


def test_label_baseline_swaps_geometry_for_a_one_hot_and_blanks_unseen_tasks():
    from wam.tasks import TaskEnv

    labels = ("T0", "T1", "T4x", "T4xneg")
    desc, lab = TaskEnv(task="T1"), TaskEnv(task="T1", labels=labels)
    a, _ = desc.reset(seed=2)
    b, _ = lab.reset(seed=2)
    assert a.shape == (59,) and b.shape == (47,)
    assert np.array_equal(a[:33], b[:33])
    slot = a[33:46]
    assert np.allclose(b[33:38], slot[[0, 9, 10, 11, 12]])
    assert list(b[-4:]) == [0.0, 1.0, 0.0, 0.0]
    held, _ = TaskEnv(task="T2", labels=labels).reset(seed=0)
    assert not held[-4:].any()


@pytest.mark.parametrize("env_id,size", [
    ("WamFamilyDesc-v0", 59), ("WamFamilyContact-v0", 37), ("WamFamilyContactFK-v0", 46),
])
def test_family_variants_have_their_observation_sizes(env_id, size):
    env = gym.make(env_id)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape == (size,)
    assert env.action_space.shape == (4,)
    env.close()


def test_family_never_trains_on_the_held_out_band_and_varies_the_box():
    from wam.tasks import TaskEnv

    env = TaskEnv(task="F", conditioning="contact")
    azimuths, sizes = [], []
    for seed in range(60):
        env.reset(seed=seed)
        d = env.push_direction()
        azimuths.append(np.degrees(np.arctan2(d[1], d[0])))
        sizes.append(env.model.geom_size[env.box_geom][0])
        assert env.data.ncon == 0 and env._outside_door()
    azimuths = np.array(azimuths)
    assert not np.any((azimuths >= 10) & (azimuths <= 50))
    assert np.any(azimuths < 10) and np.any(azimuths > 50) and np.any(azimuths < -90)  # both arcs drawn
    assert np.ptp(sizes) > 0.01


@pytest.mark.parametrize("task,geom", [("F", "box"), ("T1", "bar"), ("T2", "front"), ("T4x", "box"), ("T4xneg", "box")])
def test_contact_target_is_on_the_push_side_of_the_part_to_push(task, geom):
    from wam.tasks import TaskEnv

    env = TaskEnv(task=task, conditioning="contact")
    env.reset(seed=1)
    target, direction = env.contact_target()
    g = env.model.geom(geom).id
    assert np.dot(target - env.data.geom_xpos[g], direction) < 0  # in front of the part, on the push side
    nearest = min(env.mover_geoms, key=lambda k: np.linalg.norm(env.data.geom_xpos[k] - target))
    assert nearest == g


def test_contact_observation_carries_target_offset_direction_and_fk_points():
    from wam.tasks import TaskEnv

    env = TaskEnv(task="T4x", conditioning="contact_fk")
    obs, _ = env.reset(seed=3)
    target, direction = env.contact_target()
    forearm = env.data.geom_xpos[env.forearm_geom]
    assert np.allclose(obs[29:32], target - forearm, atol=1e-5)
    assert np.allclose(obs[32:35], direction, atol=1e-6)
    assert obs[35] == pytest.approx(env._distance() / 0.25, rel=1e-5)
    assert np.allclose(obs[37:40], env.data.xpos[env.model.body("link4").id], atol=1e-5)
    assert np.allclose(obs[43:46], env.data.xpos[env.model.body("link6").id], atol=1e-5)
