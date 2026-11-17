"""WACL CloseCabinet on an xArm 7: push a cabinet door closed with links 4-6 while holding a plate.

The reward follows REWARD_DESIGN.md term by term. `gated=False` is ablation A0: closing progress
pays whatever touches the door and wrong contact is free. The `task/` metrics in `info` always
use the real contact gate, so the two modes are scored the same way.
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

F_MIN, F_MAX = 1.0, 60.0  # N
REACH_SCALE = 0.1  # m
TILT_FREE, TILT_MAX = np.radians(10.0), np.radians(25.0)
W_PROGRESS = 50.0  # per rad
W_REACH = 5.0
W_WRONG = 0.5  # per step
W_TILT = 2.0  # per rad per step
W_FORCE = 0.5  # per step
W_ACTION_RATE, W_JOINT_VEL, W_TORQUE = 0.01, 0.001, 0.01
SUCCESS_BONUS = 20.0
SPILL_PENALTY = 15.0
REWARD_TERMS = ("progress", "reach", "wrong", "tilt", "force", "smooth", "success", "spill")

CONTACT_GROUPS = ("upper", "link4", "link5", "link6", "ee")
BODY_GROUP = {
    "link_base": "upper", "link1": "upper", "link2": "upper", "link3": "upper",
    "link4": "link4", "link5": "link5", "link6": "link6", "link7": "ee",
}


class CloseCabinetEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, gated=True):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.gated = gated
        self.renderer = None

        self.home = self.model.key("home").qpos.copy()
        self.ctrl_range = self.model.actuator_ctrlrange.copy()
        self.door_geom = self.model.geom("door").id
        self.link7 = self.model.body("link7").id
        self.torque_limit = self.model.actuator_forcerange[:, 1].copy()
        self.geom_group = np.full(self.model.ngeom, -1)
        for geom in range(self.model.ngeom):
            body = self.model.body(self.model.geom_bodyid[geom]).name
            if body in BODY_GROUP:
                self.geom_group[geom] = CONTACT_GROUPS.index(BODY_GROUP[body])
        self.allowed_capsules = [
            geom for geom in range(self.model.ngeom)
            if self.model.geom(geom).name.startswith(("link4_c", "link5_c", "link6_c"))
        ]

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

        self.start_angle = self.data.qpos[7]
        self.link_distance = self._link_distance()
        self.closing = self.valid_closing = 0.0
        self.wrong_contact_steps = 0
        self.returns = dict.fromkeys(REWARD_TERMS, 0.0)
        return self._obs(), self._info(self._contact_forces(), penetration=0.0)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        q = self.data.qpos[:7]
        self.q_target = np.clip(self.q_target + ACTION_SCALE * action, q - MAX_LEAD, q + MAX_LEAD)
        self.q_target = np.clip(self.q_target, self.ctrl_range[:, 0], self.ctrl_range[:, 1])
        self.data.ctrl[:] = self.q_target

        angle_before = self.data.qpos[7]
        forces = np.zeros(len(CONTACT_GROUPS) + 1)
        penetration = 0.0
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
            forces += self._contact_forces()
            if self.data.ncon:
                penetration = max(penetration, -self.data.contact.dist.min())
        forces /= FRAME_SKIP

        terms, terminated = self._reward(action, forces, angle_before - self.data.qpos[7])
        for name, value in terms.items():
            self.returns[name] += value
        self.last_action = action
        info = self._info(forces, penetration)
        return self._obs(), float(sum(terms.values())), terminated, False, info

    def _reward(self, action, forces, closed):
        allowed = forces[1:4].sum()
        wrong = forces[0] + forces[4] + forces[5]
        contact_goal = allowed > F_MIN and wrong == 0.0
        gate = contact_goal or not self.gated
        self.closing += max(closed, 0.0)
        self.valid_closing += contact_goal * max(closed, 0.0)
        self.wrong_contact_steps += wrong > 0.0

        distance_before, self.link_distance = self.link_distance, self._link_distance()
        tilt = self._plate_tilt()
        angle = self.data.qpos[7]
        terms = dict.fromkeys(REWARD_TERMS, 0.0)
        terms["progress"] = W_PROGRESS * (gate * max(closed, 0.0) - max(-closed, 0.0))
        terms["reach"] = W_REACH * (
            np.exp(-self.link_distance / REACH_SCALE) - np.exp(-distance_before / REACH_SCALE)
        )
        terms["wrong"] = -W_WRONG * (self.gated and wrong > 0.0)
        terms["tilt"] = -W_TILT * max(tilt - TILT_FREE, 0.0)
        terms["force"] = -W_FORCE * min(max(allowed + wrong - F_MAX, 0.0) / F_MAX, 1.0)
        terms["smooth"] = -(
            W_ACTION_RATE * np.sum((action - self.last_action) ** 2)
            + W_JOINT_VEL * np.sum(self.data.qvel[:7] ** 2)
            + W_TORQUE * np.sum((self.data.qfrc_actuator[:7] / self.torque_limit) ** 2)
        )
        if angle < SUCCESS_ANGLE:
            earned = self.valid_closing if self.gated else self.closing
            terms["success"] = SUCCESS_BONUS * min(earned / (self.start_angle - angle), 1.0)
            return terms, True
        if tilt > TILT_MAX:
            terms["spill"] = -SPILL_PENALTY
            return terms, True
        return terms, False

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

    def _link_distance(self):
        """Smallest distance (m) from the link 4-6 collision capsules to the door, 0 in contact."""
        fromto = np.zeros(6)
        return max(0.0, min(
            mujoco.mj_geomDistance(self.model, self.data, geom, self.door_geom, 1.0, fromto)
            for geom in self.allowed_capsules
        ))

    def _plate_tilt(self):
        plate_normal_z = self.data.xmat[self.link7].reshape(3, 3)[2, 1]
        return float(np.arccos(np.clip(abs(plate_normal_z), 0.0, 1.0)))

    def _info(self, forces, penetration):
        """`forces` is the mean of _contact_forces over the control step's physics substeps.

        `reward/*` are episode sums so far; `task/valid_fraction` is the share of all closing
        that happened with links 4-6 pushing and nothing else touching.
        """
        info = {f"force/door_{name}": forces[i] for i, name in enumerate(CONTACT_GROUPS)}
        info["force/scene"] = forces[-1]
        info["task/door_angle"] = float(self.data.qpos[7])
        info["task/success"] = float(self.data.qpos[7] < SUCCESS_ANGLE)
        info["task/valid_fraction"] = self.valid_closing / self.closing if self.closing > 1e-6 else 0.0
        info["task/wrong_contact_steps"] = float(self.wrong_contact_steps)
        info["task/spill"] = float(self._plate_tilt() > TILT_MAX)
        info["task/plate_tilt"] = self._plate_tilt()
        info["task/penetration"] = float(penetration)
        info.update({f"reward/{name}": float(value) for name, value in self.returns.items()})
        return info
