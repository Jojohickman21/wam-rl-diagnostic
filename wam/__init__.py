import os
import sys

# MuJoCo picks its GL backend at import time; a headless Linux node has no GLX display.
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np

gym.register(
    id="WamCloseCabinet-v0",
    entry_point="wam.close_cabinet:CloseCabinetEnv",
    max_episode_steps=250,
)
gym.register(
    id="WamCloseCabinetUngated-v0",
    entry_point="wam.close_cabinet:CloseCabinetEnv",
    max_episode_steps=250,
    kwargs={"gated": False},
)
gym.register(
    id="WamCloseCabinetRandomStart-v0",
    entry_point="wam.close_cabinet:CloseCabinetEnv",
    max_episode_steps=250,
    kwargs={"start_spread": 0.5},
)
gym.register(
    id="WamCloseCabinetRandomStartUngated-v0",
    entry_point="wam.close_cabinet:CloseCabinetEnv",
    max_episode_steps=250,
    kwargs={"start_spread": 0.5, "gated": False},
)
gym.register(
    id="WamCloseCabinetRandomPose-v0",
    entry_point="wam.close_cabinet:CloseCabinetEnv",
    max_episode_steps=250,
    kwargs={"start_spread": 0.5, "cabinet_spread": (0.1, np.radians(20))},
)
gym.register(
    id="WamCloseCabinetRandomPoseUngated-v0",
    entry_point="wam.close_cabinet:CloseCabinetEnv",
    max_episode_steps=250,
    kwargs={"start_spread": 0.5, "cabinet_spread": (0.1, np.radians(20)), "gated": False},
)
gym.register(
    id="WamCloseCabinetRandomPoseBarrier-v0",
    entry_point="wam.close_cabinet:CloseCabinetEnv",
    max_episode_steps=250,
    kwargs={"start_spread": 0.5, "cabinet_spread": (0.1, np.radians(20)), "tilt_barrier": True},
)
gym.register(
    id="WamCloseCabinetRandomPoseCoupled-v0",
    entry_point="wam.close_cabinet:CloseCabinetEnv",
    max_episode_steps=250,
    kwargs={
        "start_spread": 0.5, "cabinet_spread": (0.1, np.radians(20)),
        "tilt_barrier": True, "level_gate": True,
    },
)
gym.register(
    id="WamCloseCabinetRandomPoseTray-v0",
    entry_point="wam.close_cabinet:CloseCabinetEnv",
    max_episode_steps=250,
    kwargs={"start_spread": 0.5, "cabinet_spread": (0.1, np.radians(20)), "tray_object": True},
)
gym.register(
    id="WamCloseCabinetTrayFrictionCoupled-v0",
    entry_point="wam.close_cabinet:CloseCabinetEnv",
    max_episode_steps=250,
    kwargs={
        "start_spread": 0.5, "cabinet_spread": (0.1, np.radians(20)), "tray_object": True,
        "observe_object": False, "friction_coupling": True, "energy_penalty": True,
    },
)
# Section 6: each differs from one earlier env by a single change.
gym.register(
    id="WamCloseCabinetTrayCoupledSeeObject-v0",
    entry_point="wam.close_cabinet:CloseCabinetEnv",
    max_episode_steps=250,
    kwargs={
        "start_spread": 0.5, "cabinet_spread": (0.1, np.radians(20)), "tray_object": True,
        "observe_object": True, "observe_slide_angle": False,
        "friction_coupling": True, "energy_penalty": True,
    },
)
gym.register(
    id="WamCloseCabinetTrayCoupledSeeRisk-v0",
    entry_point="wam.close_cabinet:CloseCabinetEnv",
    max_episode_steps=250,
    kwargs={
        "start_spread": 0.5, "cabinet_spread": (0.1, np.radians(20)), "tray_object": True,
        "observe_object": False, "observe_risk": True,
        "friction_coupling": True, "energy_penalty": True,
    },
)
gym.register(
    id="WamCloseCabinetTrayEnergy-v0",
    entry_point="wam.close_cabinet:CloseCabinetEnv",
    max_episode_steps=250,
    kwargs={
        "start_spread": 0.5, "cabinet_spread": (0.1, np.radians(20)), "tray_object": True,
        "energy_penalty": True,
    },
)
# Section 7.
gym.register(
    id="WamCloseCabinetTrayWristLeveler-v0",
    entry_point="wam.close_cabinet:CloseCabinetEnv",
    max_episode_steps=250,
    kwargs={
        "start_spread": 0.5, "cabinet_spread": (0.1, np.radians(20)), "tray_object": True,
        "observe_object": False, "observe_slide_angle": False,
        "energy_penalty": True, "wrist_leveler": True,
    },
)
gym.register(
    id="WamCloseCabinetTraySeeRisk-v0",
    entry_point="wam.close_cabinet:CloseCabinetEnv",
    max_episode_steps=250,
    kwargs={
        "start_spread": 0.5, "cabinet_spread": (0.1, np.radians(20)), "tray_object": True,
        "observe_object": False, "observe_risk": True, "energy_penalty": True,
    },
)
gym.register(
    id="WamCloseCabinetTrayWristLevelerMargin-v0",
    entry_point="wam.close_cabinet:CloseCabinetEnv",
    max_episode_steps=250,
    kwargs={
        "start_spread": 0.5, "cabinet_spread": (0.1, np.radians(20)), "tray_object": True,
        "observe_object": False, "observe_slide_angle": False,
        "energy_penalty": True, "wrist_leveler": True, "observe_joint_margin": True,
    },
)
gym.register(
    id="WamCloseCabinetTrayWristLevelerLimitPenalty-v0",
    entry_point="wam.close_cabinet:CloseCabinetEnv",
    max_episode_steps=250,
    kwargs={
        "start_spread": 0.5, "cabinet_spread": (0.1, np.radians(20)), "tray_object": True,
        "observe_object": False, "observe_slide_angle": False,
        "energy_penalty": True, "wrist_leveler": True, "observe_joint_margin": True,
        "joint_limit_penalty": True,
    },
)
# End-effector-first mix: half the starts put the end-effector nearer the door than the forearm.
gym.register(
    id="WamCloseCabinetEEFirst-v0",
    entry_point="wam.close_cabinet:CloseCabinetEnv",
    max_episode_steps=250,
    kwargs={"start_spread": 1.0, "cabinet_spread": (0.1, np.radians(20)), "ee_first_fraction": 0.5},
)
gym.register(
    id="WamCloseCabinetEEFirstUngated-v0",
    entry_point="wam.close_cabinet:CloseCabinetEnv",
    max_episode_steps=250,
    kwargs={"start_spread": 1.0, "cabinet_spread": (0.1, np.radians(20)), "ee_first_fraction": 0.5, "gated": False},
)
# Outcome-only reward: no reach shaping, no per-step gate or wrong-contact penalty.
gym.register(
    id="WamCloseCabinetTrayWristLevelerOutcome-v0",
    entry_point="wam.close_cabinet:CloseCabinetEnv",
    max_episode_steps=250,
    kwargs={
        "start_spread": 0.5, "cabinet_spread": (0.1, np.radians(20)), "tray_object": True,
        "observe_object": False, "observe_slide_angle": False,
        "energy_penalty": True, "wrist_leveler": True, "outcome_reward": True,
    },
)
gym.register(
    id="WamCloseCabinetEEFirstOutcome-v0",
    entry_point="wam.close_cabinet:CloseCabinetEnv",
    max_episode_steps=250,
    kwargs={
        "start_spread": 1.0, "cabinet_spread": (0.1, np.radians(20)), "ee_first_fraction": 0.5,
        "outcome_reward": True,
    },
)
# End-effector-first starts with the A20 wrist leveler plus tuck (no tray): the rebuilt A24, A25, A27.
# The 7-joint versions above (A24v1 etc.) could not hold the plate from the wider starts.
_EE_LEVELER = {
    "start_spread": 1.0, "cabinet_spread": (0.1, np.radians(20)), "ee_first_fraction": 0.5,
    "wrist_leveler": True, "leveler_tuck": 0.2, "ee_start_pool": 300,
}
gym.register(
    id="WamCloseCabinetEEFirstLeveler-v0",
    entry_point="wam.close_cabinet:CloseCabinetEnv",
    max_episode_steps=250,
    kwargs=_EE_LEVELER,
)
gym.register(
    id="WamCloseCabinetEEFirstLevelerUngated-v0",
    entry_point="wam.close_cabinet:CloseCabinetEnv",
    max_episode_steps=250,
    kwargs={**_EE_LEVELER, "gated": False},
)
gym.register(
    id="WamCloseCabinetEEFirstLevelerOutcome-v0",
    entry_point="wam.close_cabinet:CloseCabinetEnv",
    max_episode_steps=250,
    kwargs={**_EE_LEVELER, "outcome_reward": True},
)
# Task set (TASK_SET.md): A20's structure on every task, constraint descriptor in the observation.
from wam.tasks import TASKS as _TASKS  # noqa: E402

for _task in _TASKS:
    gym.register(id=f"WamTask{_task}-v0", entry_point="wam.tasks:TaskEnv", max_episode_steps=250,
                 kwargs={"task": _task})
    gym.register(id=f"WamTask{_task}Outcome-v0", entry_point="wam.tasks:TaskEnv", max_episode_steps=250,
                 kwargs={"task": _task, "outcome_reward": True})
