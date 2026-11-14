import os
import sys

# MuJoCo picks its GL backend at import time; a headless Linux node has no GLX display.
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym

gym.register(
    id="WamCloseCabinet-v0",
    entry_point="wam.close_cabinet:CloseCabinetEnv",
    max_episode_steps=250,
)
