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
