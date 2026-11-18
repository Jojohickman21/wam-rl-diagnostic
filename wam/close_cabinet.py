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
W_BARRIER = 1.0  # per step at the spill limit
W_FORCE = 0.5  # per step
W_ACTION_RATE, W_JOINT_VEL, W_TORQUE = 0.01, 0.001, 0.01
W_ENERGY = 0.004  # per W, Deep Whole-Body Control (Fu, Cheng, Pathak, CoRL 2022) Table 1, arm term
# A18: the door reward is scaled by exp(-K_COUPLING * excess) where excess is how far the friction
# a resting object would need exceeds MU_DESIGN; 0.6 at the 15 degree slide angle, 0.2 at 20.
MU_DESIGN = np.tan(np.radians(12.0))
K_COUPLING = 10.0
MU_REQ_CAP = 2.0  # stands in for "the object would lift off the plate"
GRAVITY = np.array([0.0, 0.0, -9.81])
SUCCESS_BONUS = 20.0
SPILL_PENALTY = 15.0
START_TILT = np.radians(2.0)

# A7 tray object: a squat cylinder that slides before it tips (tip angle atan(2r/h) = 56 deg).
OBJECT_RADIUS, OBJECT_HALF_HEIGHT, OBJECT_MASS = 0.03, 0.02, 0.2  # m, m, kg
PLATE_RADIUS, PLATE_HALF_THICKNESS = 0.09, 0.004  # must match the plate geom in xarm7_wam.xml
SLIDE_ANGLE_RANGE = (np.radians(15.0), np.radians(35.0))
# Below this the arm can pin the cylinder against the floor and crush it 15 mm into it.
MIN_TRAY_HEIGHT = 0.15  # m, plate centre at reset
REWARD_TERMS = ("progress", "reach", "wrong", "tilt", "force", "smooth", "energy", "success", "spill")

CONTACT_GROUPS = ("upper", "link4", "link5", "link6", "ee")
BODY_GROUP = {
    "link_base": "upper", "link1": "upper", "link2": "upper", "link3": "upper",
    "link4": "link4", "link5": "link5", "link6": "link6", "link7": "ee",
}


class CloseCabinetEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, gated=True, start_spread=0.02, cabinet_spread=(0.0, 0.0),
                 level_gate=False, tilt_barrier=False, tray_object=False, observe_object=True,
                 observe_slide_angle=None, observe_risk=False, friction_coupling=False,
                 energy_penalty=False, wrist_leveler=False):
        """`start_spread` is the half-width (rad) of the uniform draw around home on every joint.

        `level_gate` (A4) pays closing progress only while the plate is within TILT_FREE.
        `tilt_barrier` (A5) replaces the linear tilt penalty with a quadratic one that reaches
        W_BARRIER per step at the spill limit.
        `tray_object` (A7) puts a free cylinder on the plate. The plate constraint becomes physical:
        the episode spills when the cylinder leaves the plate, and there is no tilt penalty. The
        cylinder's friction is drawn each episode so that it starts to slide on a static plate at
        an angle between 15 and 35 degrees; that angle replaces the 25 degree limit in the
        observation, and the cylinder's position and velocity in the plate frame are appended.
        `observe_object=False` leaves both out, so the policy sees only what encoders and a door
        camera give on hardware. `observe_slide_angle` (default: same as observe_object) controls
        the slide angle alone; hardware cannot measure it, the object's position on the plate
        it can, with a camera.
        `observe_risk` appends the previous control step's required friction (encoder-derived).
        `friction_coupling` (A18) scales the positive door reward by how safely the plate is
        moving: see _required_friction, which uses only the arm's own kinematics.
        `energy_penalty` replaces the action-rate, joint-velocity and torque terms with
        W_ENERGY times the absolute mechanical power of the arm joints.
        `wrist_leveler` (A20) hands joints 5-7 to a feedback controller that keeps the plate level
        (see _wrist_targets); the action shrinks to joints 1-4. The observation keeps its shape:
        the last-action slots for joints 5-7 hold the controller's own target change, in action
        units, which a real controller also knows.

        `cabinet_spread` is (metres, radians): the cabinet's hinge is shifted uniformly within a
        square of that half-width in x and y and turned about z by up to that angle.
        """
        spec = mujoco.MjSpec.from_file(str(SCENE))
        if tray_object:
            body = spec.worldbody.add_body(name="tray_object")
            body.add_freejoint()
            body.add_geom(
                name="tray_object", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[OBJECT_RADIUS, OBJECT_HALF_HEIGHT, 0], mass=OBJECT_MASS, rgba=[0.9, 0.6, 0.1, 1],
                contype=1, conaffinity=1, condim=3, solref=[0.002, 1], solimp=[0.95, 0.99, 0.001, 0.5, 2],
                # Higher priority makes the contact use this geom's friction, not the max of both.
                priority=1,
            )
            # Without both, soft-contact creep slides the cylinder off at about 60% of its
            # friction angle; with them it holds at 90% and slides at 110% (scripts/calibrate_tray.py).
            spec.option.noslip_iterations = 10
            spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.gated = gated
        self.level_gate = level_gate
        self.tilt_barrier = tilt_barrier
        self.tray_object = tray_object
        self.observe_object = tray_object and observe_object
        if observe_slide_angle is None:
            observe_slide_angle = observe_object
        self.observe_slide_angle = tray_object and observe_slide_angle
        self.observe_risk = observe_risk
        self.friction_coupling = friction_coupling
        self.energy_penalty = energy_penalty
        self.wrist_leveler = wrist_leveler
        self.tilt_max = TILT_MAX
        self.start_spread = start_spread
        self.cabinet_spread = cabinet_spread
        self.renderer = None
        # Scratch state for the wrist controller's IK, so it never touches the simulated arm.
        self.ik_data = mujoco.MjData(self.model)

        self.home = self.model.key("home").qpos.copy()
        self.ctrl_range = self.model.actuator_ctrlrange.copy()
        self.door_geom = self.model.geom("door").id
        self.link7 = self.model.body("link7").id
        self.plate_geom = self.model.geom("plate").id
        self.object_geom = self.model.geom("tray_object").id if tray_object else -1
        if tray_object:
            joint = self.model.joint(self.model.body("tray_object").jntadr[0])
            self.object_qpos, self.object_qvel = joint.qposadr[0], joint.dofadr[0]
        self.door_body = self.model.body("door").id
        self.cabinet_body = self.model.body("cabinet").id
        self.cabinet_home = self.model.body_pos[self.cabinet_body].copy()
        self.torque_limit = self.model.actuator_forcerange[:, 1].copy()
        self.geom_group = np.full(self.model.ngeom, -1)
        for geom in range(self.model.ngeom):
            body = self.model.body(self.model.geom_bodyid[geom]).name
            if body in BODY_GROUP:
                self.geom_group[geom] = CONTACT_GROUPS.index(BODY_GROUP[body])
        self.distal_geoms = [
            geom for geom in range(self.model.ngeom)
            if self.geom_group[geom] > 0 and self.model.geom_contype[geom]
        ]
        self.allowed_capsules = [
            geom for geom in range(self.model.ngeom)
            if self.model.geom(geom).name.startswith(("link4_c", "link5_c", "link6_c"))
        ]

        self.action_space = gym.spaces.Box(-1.0, 1.0, (4 if wrist_leveler else 7,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (35 + 6 * self.observe_object + self.observe_risk,), np.float32)
        self.q_target = self.home[:7].copy()
        self.last_action = np.zeros(7)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        shift, turn = self.cabinet_spread
        self.cabinet_yaw = self.np_random.uniform(-turn, turn)
        if self.tray_object:
            self.tilt_max = self.np_random.uniform(*SLIDE_ANGLE_RANGE)
            self.model.geom_friction[self.object_geom, 0] = np.tan(self.tilt_max)
        self.model.body_pos[self.cabinet_body] = self.cabinet_home + np.append(
            self.np_random.uniform(-shift, shift, 2), 0.0
        )
        mujoco.mju_axisAngle2Quat(self.model.body_quat[self.cabinet_body], [0.0, 0.0, 1.0], self.cabinet_yaw)
        while True:
            self.data.qpos[7] = self.np_random.uniform(1.0, 1.4)
            spread = self.np_random.uniform(-self.start_spread, self.start_spread, 7)
            self.data.qpos[:7] = np.clip(self.home[:7] + spread, self.ctrl_range[:, 0], self.ctrl_range[:, 1])
            self._level_plate()
            if self.tray_object:
                self._place_object()
            mujoco.mj_forward(self.model, self.data)
            tray_clear = not self.tray_object or self.data.geom_xpos[self.plate_geom][2] > MIN_TRAY_HEIGHT
            if self.data.ncon == 0 and self._plate_tilt() < START_TILT and self._outside_door() and tray_clear:
                break
        self.q_target = self.data.qpos[:7].copy()
        self.data.ctrl[:] = self.q_target
        self.last_action = np.zeros(7)

        self.start_angle = self.data.qpos[7]
        self.link_distance = self._link_distance()
        self.closing = self.valid_closing = 0.0
        self.wrong_contact_steps = 0
        self.returns = dict.fromkeys(REWARD_TERMS, 0.0)
        self.object_point_velocity = self._object_point_velocity()
        self.friction_required, self.coupling_sum, self.steps = 0.0, 0.0, 0
        return self._obs(), self._info(self._contact_forces(), penetration=0.0)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        q = self.data.qpos[:7]
        if self.wrist_leveler:
            arm = np.clip(self.q_target[:4] + ACTION_SCALE * action, q[:4] - MAX_LEAD, q[:4] + MAX_LEAD)
            arm = np.clip(arm, self.ctrl_range[:4, 0], self.ctrl_range[:4, 1])
            wrist = self._wrist_targets(arm)
            action = np.concatenate([action, (wrist - self.q_target[4:]) / ACTION_SCALE])
            self.q_target = np.concatenate([arm, wrist])
        else:
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
        tilt = self._plate_tilt()
        gate = (contact_goal or not self.gated) and (tilt < TILT_FREE or not self.level_gate)
        self.closing += max(closed, 0.0)
        self.valid_closing += contact_goal * max(closed, 0.0)
        self.wrong_contact_steps += wrong > 0.0

        distance_before, self.link_distance = self.link_distance, self._link_distance()
        self.friction_required = self._required_friction()
        coupling = 1.0
        if self.friction_coupling:
            coupling = np.exp(-K_COUPLING * max(self.friction_required - MU_DESIGN, 0.0))
        self.coupling_sum += coupling
        self.steps += 1
        angle = self.data.qpos[7]
        terms = dict.fromkeys(REWARD_TERMS, 0.0)
        # Only the positive door reward is scaled: reopening and retreating cost the full amount.
        terms["progress"] = W_PROGRESS * (coupling * gate * max(closed, 0.0) - max(-closed, 0.0))
        reach = W_REACH * (
            np.exp(-self.link_distance / REACH_SCALE) - np.exp(-distance_before / REACH_SCALE)
        )
        terms["reach"] = coupling * reach if reach > 0.0 else reach
        terms["wrong"] = -W_WRONG * (self.gated and wrong > 0.0)
        excess = max(tilt - TILT_FREE, 0.0)
        if self.tilt_barrier:
            terms["tilt"] = -W_BARRIER * (excess / (self.tilt_max - TILT_FREE)) ** 2
        elif not self.tray_object:
            terms["tilt"] = -W_TILT * excess
        terms["force"] = -W_FORCE * min(max(allowed + wrong - F_MAX, 0.0) / F_MAX, 1.0)
        if self.energy_penalty:
            terms["energy"] = -W_ENERGY * np.sum(np.abs(self.data.qfrc_actuator[:7] * self.data.qvel[:7]))
        else:
            terms["smooth"] = -(
                W_ACTION_RATE * np.sum((action - self.last_action) ** 2)
                + W_JOINT_VEL * np.sum(self.data.qvel[:7] ** 2)
                + W_TORQUE * np.sum((self.data.qfrc_actuator[:7] / self.torque_limit) ** 2)
            )
        if angle < SUCCESS_ANGLE:
            earned = self.valid_closing if self.gated else self.closing
            terms["success"] = coupling * SUCCESS_BONUS * min(earned / (self.start_angle - angle), 1.0)
            return terms, True
        if self._spilled():
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
        hinge = self.model.body_pos[self.cabinet_body][:2]
        obs = [
            q[:7], qvel[:7], self.q_target - q[:7], q[7:8], qvel[7:8], self.last_action,
            hinge, [np.cos(self.cabinet_yaw), np.sin(self.cabinet_yaw)],
            [self.tilt_max if self.observe_slide_angle else TILT_MAX],
        ]
        if self.observe_object:
            to_plate = self.data.geom_xmat[self.plate_geom].reshape(3, 3).T
            offset = self.data.geom_xpos[self.object_geom] - self.data.geom_xpos[self.plate_geom]
            velocity = qvel[self.object_qvel : self.object_qvel + 3]
            obs += [to_plate @ offset, to_plate @ velocity]
        if self.observe_risk:
            obs.append([self.friction_required])
        return np.concatenate(obs).astype(np.float32)

    def _contact_forces(self):
        """Normal force (N) on the door from each arm group, then arm force on the rest of the scene.

        Contacts with the tray object are not arm contacts and are left out.
        """
        forces = np.zeros(len(CONTACT_GROUPS) + 1)
        wrench = np.zeros(6)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if self.object_geom in (contact.geom1, contact.geom2):
                continue
            mujoco.mj_contactForce(self.model, self.data, i, wrench)
            if contact.geom1 == self.door_geom:
                forces[self.geom_group[contact.geom2]] += wrench[0]
            elif contact.geom2 == self.door_geom:
                forces[self.geom_group[contact.geom1]] += wrench[0]
            else:
                forces[-1] += wrench[0]
        return forces

    def _level_plate(self):
        """Turn the three wrist joints until the plate normal is vertical; the other joints stay."""
        jac = np.zeros((3, self.model.nv))
        for _ in range(20):
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            normal = self.data.xmat[self.link7].reshape(3, 3)[:, 1]
            error = np.cross(normal, [0.0, 0.0, np.sign(normal[2])])
            mujoco.mj_jacBody(self.model, self.data, None, jac, self.link7)
            wrist = jac[:, 4:7]
            step = wrist.T @ np.linalg.solve(wrist @ wrist.T + 1e-4 * np.eye(3), error)
            self.data.qpos[4:7] = np.clip(
                self.data.qpos[4:7] + step, self.ctrl_range[4:, 0], self.ctrl_range[4:, 1]
            )

    def _wrist_targets(self, arm):
        """A20 wrist controller: joint 5-7 targets that level the plate once joints 1-4 reach `arm`.

        Damped least squares on the plate normal (link 7's y axis), as _level_plate does at reset,
        started from the measured wrist angles. Inputs are encoder readings and commanded targets
        only. The result may lead the measured wrist by at most MAX_LEAD, like the policy's joints.
        """
        data, jac = self.ik_data, np.zeros((3, self.model.nv))
        data.qpos[:] = self.data.qpos
        data.qpos[:4] = arm
        q = self.data.qpos[4:7]
        for _ in range(3):
            mujoco.mj_kinematics(self.model, data)
            mujoco.mj_comPos(self.model, data)
            normal = data.xmat[self.link7].reshape(3, 3)[:, 1]
            error = np.cross(normal, [0.0, 0.0, np.sign(normal[2])])
            mujoco.mj_jacBody(self.model, data, None, jac, self.link7)
            wrist = jac[:, 4:7]
            step = wrist.T @ np.linalg.solve(wrist @ wrist.T + 1e-4 * np.eye(3), error)
            data.qpos[4:7] = np.clip(data.qpos[4:7] + step, q - MAX_LEAD, q + MAX_LEAD)
        return np.clip(data.qpos[4:7], self.ctrl_range[4:, 0], self.ctrl_range[4:, 1])

    def _plate_up(self):
        """Unit normal of the plate's upper face (the plate is levelled with either face up)."""
        axis = self.data.geom_xmat[self.plate_geom].reshape(3, 3)[:, 2]
        return axis if axis[2] >= 0.0 else -axis

    def _object_point_velocity(self):
        """World velocity of the point where a resting object's centre sits, fixed to link 7."""
        point = self.data.geom_xpos[self.plate_geom] + (
            PLATE_HALF_THICKNESS + OBJECT_HALF_HEIGHT
        ) * self._plate_up()
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jac, None, point, self.link7)
        return jac[:, :7] @ self.data.qvel[:7]

    def _required_friction(self):
        """Friction coefficient a resting object would need to stay put (Heins & Schoellig, 2023).

        Built from arm kinematics only, as on hardware: the object point's acceleration is the
        change in its encoder-derived velocity over one control step, and the object itself is
        never measured. Static and tilted by a: tan(a). Accelerating: grows with the tangential
        component of gravity minus acceleration in the plate frame.
        """
        velocity = self._object_point_velocity()
        accel = (velocity - self.object_point_velocity) / (FRAME_SKIP * self.model.opt.timestep)
        self.object_point_velocity = velocity
        up = self._plate_up()
        g_eff = GRAVITY - accel
        pressing = -(g_eff @ up)
        if pressing < 0.1 * 9.81:
            return MU_REQ_CAP
        return min(np.linalg.norm(g_eff + pressing * up) / pressing, MU_REQ_CAP)

    def _place_object(self):
        """Stand the cylinder on the plate's centre, 1 mm clear so the start is out of contact."""
        mujoco.mj_kinematics(self.model, self.data)
        up = self._plate_up()
        lift = PLATE_HALF_THICKNESS + OBJECT_HALF_HEIGHT + 0.001
        self.data.qpos[self.object_qpos : self.object_qpos + 3] = (
            self.data.geom_xpos[self.plate_geom] + lift * up
        )
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.geom_xmat[self.plate_geom])
        self.data.qpos[self.object_qpos + 3 : self.object_qpos + 7] = quat

    def _spilled(self):
        """Tray mode: the cylinder's centre has left the cylinder above the plate. Otherwise: tilt."""
        if not self.tray_object:
            return self._plate_tilt() > self.tilt_max
        up = self._plate_up()
        offset = self.data.geom_xpos[self.object_geom] - self.data.geom_xpos[self.plate_geom]
        height = offset @ up
        return bool(np.linalg.norm(offset - height * up) > PLATE_RADIUS or not 0.0 < height < 0.1)

    def _outside_door(self):
        """True when links 4-7 and the plate are all on the side of the door that closes it.

        A start between the door and the cabinet would have to go around the door's free edge
        first, which is a different and harder task than the one being measured.
        """
        outward = -self.data.xmat[self.door_body].reshape(3, 3)[:, 1]
        offsets = self.data.geom_xpos[self.distal_geoms] - self.data.xpos[self.door_body]
        return bool(np.all(offsets @ outward > 0.0))

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
        info["task/spill"] = float(self._spilled())
        info["task/tilt_limit"] = float(self.tilt_max)
        info["task/plate_tilt"] = self._plate_tilt()
        info["task/friction_required"] = float(self.friction_required)
        info["task/coupling_mean"] = self.coupling_sum / self.steps if self.steps else 1.0
        info["task/penetration"] = float(penetration)
        info.update({f"reward/{name}": float(value) for name, value in self.returns.items()})
        return info
