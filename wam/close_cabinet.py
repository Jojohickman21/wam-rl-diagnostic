"""WACL CloseCabinet on an xArm 7: push a cabinet door closed with links 4-6 while holding a plate.

The reward is not implemented yet (see REWARD_DESIGN.md); step() returns 0.0 and reports the
quantities the reward will be built from in `info`.
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

SCENE = Path(__file__).parents[1] / "assets" / "close_cabinet.xml"

FRAME_SKIP = 20  # 1 kHz physics, 50 Hz control
ACTION_SCALE = 0.04  # rad per control step at |a|=1, i.e. 2 rad/s
# The position target may lead the measured joint by at most this much. Without the clamp the
# target integrates without bound while a link is blocked and takes many steps to unwind.
MAX_LEAD = 0.1  # rad
SUCCESS_ANGLE = 0.05  # rad

CONTACT_GROUPS = ("upper", "link4", "link5", "link6", "ee")
BODY_GROUP = {
    "link_base": "upper", "link1": "upper", "link2": "upper", "link3": "upper",
    "link4": "link4", "link5": "link5", "link6": "link6", "link7": "ee",
}


class CloseCabinetEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.renderer = None

        self.home = self.model.key("home").qpos.copy()
        self.ctrl_range = self.model.actuator_ctrlrange.copy()
        self.door_geom = self.model.geom("door").id
        self.link7 = self.model.body("link7").id
        self.geom_group = np.full(self.model.ngeom, -1)
        for geom in range(self.model.ngeom):
            body = self.model.body(self.model.geom_bodyid[geom]).name
            if body in BODY_GROUP:
                self.geom_group[geom] = CONTACT_GROUPS.index(BODY_GROUP[body])

        self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (30,), np.float32)
        self.q_target = self.home[:7].copy()
        self.last_action = np.zeros(7)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = self.home[:7] + self.np_random.uniform(-0.02, 0.02, 7)
        self.data.qpos[7] = self.np_random.uniform(1.0, 1.4)
        self.q_target = self.data.qpos[:7].copy()
        self.data.ctrl[:] = self.q_target
        self.last_action = np.zeros(7)
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), self._info(self._contact_forces(), penetration=0.0)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        q = self.data.qpos[:7]
        self.q_target = np.clip(self.q_target + ACTION_SCALE * action, q - MAX_LEAD, q + MAX_LEAD)
        self.q_target = np.clip(self.q_target, self.ctrl_range[:, 0], self.ctrl_range[:, 1])
        self.data.ctrl[:] = self.q_target

        forces = np.zeros(len(CONTACT_GROUPS) + 1)
        penetration = 0.0
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
            forces += self._contact_forces()
            if self.data.ncon:
                penetration = max(penetration, -self.data.contact.dist.min())
        self.last_action = action
        return self._obs(), 0.0, False, False, self._info(forces / FRAME_SKIP, penetration)

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, 480, 640)
        self.renderer.update_scene(self.data, "overview")
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _obs(self):
        q, qvel = self.data.qpos, self.data.qvel
        return np.concatenate(
            [q[:7], qvel[:7], self.q_target - q[:7], q[7:8], qvel[7:8], self.last_action]
        ).astype(np.float32)

    def _contact_forces(self):
        """Normal force (N) on the door from each arm group, then arm force on the rest of the scene."""
        forces = np.zeros(len(CONTACT_GROUPS) + 1)
        wrench = np.zeros(6)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            mujoco.mj_contactForce(self.model, self.data, i, wrench)
            if contact.geom1 == self.door_geom:
                forces[self.geom_group[contact.geom2]] += wrench[0]
            elif contact.geom2 == self.door_geom:
                forces[self.geom_group[contact.geom1]] += wrench[0]
            else:
                forces[-1] += wrench[0]
        return forces

    def _info(self, forces, penetration):
        """`forces` is the mean of _contact_forces over the control step's physics substeps."""
        plate_normal_z = self.data.xmat[self.link7].reshape(3, 3)[2, 1]
        info = {f"force/door_{name}": forces[i] for i, name in enumerate(CONTACT_GROUPS)}
        info["force/scene"] = forces[-1]
        info["task/door_angle"] = float(self.data.qpos[7])
        info["task/success"] = float(self.data.qpos[7] < SUCCESS_ANGLE)
        info["task/plate_tilt"] = float(np.arccos(np.clip(abs(plate_normal_z), 0.0, 1.0)))
        info["task/penetration"] = float(penetration)
        return info
