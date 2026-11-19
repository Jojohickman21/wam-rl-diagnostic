"""The task set of TASK_SET.md: one environment for every articulated object, on A20's structure.

`TaskEnv` keeps everything `CloseCabinetEnv` does for the arm (wrist leveler, tray object, spill,
energy term, contact groups) and replaces the door with a task: a fixture body, a moving body
("mover") on one or two joints, a start range, a goal and a range that normalises reward.

What changes from the door, term by term:
- Progress is the decrease of d = |s − s*|, the distance of the mover's joint coordinates s from
  the goal, at W_PROGRESS · (1.2 / range) per unit, so every task's full progress is worth what
  the door's is (50 per rad over the door's 1.2 rad). Reopening is charged at the same rate.
- Contact goal, wrong contact, reach shaping and the success bonus are the door's, with "the door"
  replaced by every geom on the mover.
- The observation drops the door angle and hinge pose and adds the constraint descriptor: for each
  of two joint slots, [valid, hinge, slide, axis (3), anchor (3), s, ṡ, s*, range], with the axis
  and anchor in the world frame. Everything in it comes from the object's articulation model and
  its joint state, which on hardware means a perceived articulation, as the cabinet pose is now.

T0 is the door itself; with the same seed it draws the same starts as A20.
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from wam.close_cabinet import (
    ACTION_SCALE, BODY_GROUP, CONTACT_GROUPS, FRAME_SKIP, F_MAX, F_MIN, MAX_LEAD, MIN_TRAY_HEIGHT, REACH_SCALE,
    REWARD_TERMS, SPILL_PENALTY, START_TILT, SUCCESS_BONUS, TILT_MAX, TILT_FREE, W_ACTION_RATE,
    W_ENERGY, W_FORCE, W_JOINT_VEL, W_PROGRESS, W_REACH, W_TILT, W_TORQUE, W_WRONG, SLIDE_ANGLE_RANGE,
    CloseCabinetEnv,
)

ASSETS = Path(__file__).parents[1] / "assets"
DOOR_RANGE = 1.2  # rad; W_PROGRESS is per rad of door, so a task's progress weight is W_PROGRESS · 1.2 / range
SLOTS = 2
SLOT_SIZE = 13

# outward: unit vector in the mover's frame; at reset every distal arm geom must lie on that side
# of the plane through the mover's origin, so the arm starts on the side it has to push from.
# goal: ("fixed", values) or ("axis", lo, hi, sign) for a distance along the single joint, or
# ("planar", lo, hi, angle_lo, angle_hi) for a direction in the fixture's xy plane.
TASKS = {
    "T0": dict(scene="close_cabinet.xml", fixture="cabinet", mover="door", joints=("door_hinge",),
               start=(1.0, 1.4), goal=("fixed", (0.0,)), range=DOOR_RANGE, tol=0.05, outward=(0, -1, 0)),
    "T1": dict(scene="task_slide_door.xml", fixture="fixture", mover="mover", joints=("slide",),
               start=(-0.30, -0.20), goal=("fixed", (0.0,)), range=0.25, tol=0.015, outward=(-1, 0, 0)),
    "T2": dict(scene="task_drawer_push.xml", fixture="fixture", mover="mover", joints=("slide",),
               start=(0.15, 0.30), goal=("fixed", (0.0,)), range=0.225, tol=0.015, outward=(1, 0, 0)),
    "T3": dict(scene="task_drawer_pull.xml", fixture="fixture", mover="mover", joints=("slide",),
               start=(0.0, 0.02), goal=("fixed", (0.25,)), range=0.25, tol=0.015, outward=(1, 0, 0)),
    "T4x": dict(scene="task_box_x.xml", fixture="fixture", mover="mover", joints=("slide_x",),
                start=(0.0, 0.0), goal=("axis", 0.20, 0.30, 1.0), range=0.25, tol=0.02, outward=None),
    "T4y": dict(scene="task_box_y.xml", fixture="fixture", mover="mover", joints=("slide_y",),
                start=(0.0, 0.0), goal=("axis", 0.20, 0.30, -1.0), range=0.25, tol=0.02, outward=None),
    "T4xy": dict(scene="task_box_xy.xml", fixture="fixture", mover="mover", joints=("slide_x", "slide_y"),
                 start=(0.0, 0.0), goal=("planar", 0.25, 0.25, -np.pi / 2, 0.0), range=0.25, tol=0.02,
                 outward=None),
}


class TaskEnv(CloseCabinetEnv):

    def __init__(self, task="T0", render_mode=None, gated=True, start_spread=0.5,
                 cabinet_spread=(0.1, np.radians(20)), tray_object=True, observe_object=False,
                 energy_penalty=True, wrist_leveler=True, outcome_reward=False):
        spec_ = TASKS[task]
        self.task = task
        self.spec_ = spec_
        spec = mujoco.MjSpec.from_file(str(ASSETS / spec_["scene"]))
        if tray_object:
            body = spec.worldbody.add_body(name="tray_object")
            body.add_freejoint()
            body.add_geom(
                name="tray_object", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[0.03, 0.02, 0], mass=0.2, rgba=[0.9, 0.6, 0.1, 1],
                contype=1, conaffinity=1, condim=3, solref=[0.002, 1], solimp=[0.95, 0.99, 0.001, 0.5, 2],
                priority=1,
            )
            spec.option.noslip_iterations = 10
            spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.gated = gated
        self.level_gate = self.tilt_barrier = self.friction_coupling = False
        self.observe_risk = self.observe_joint_margin = self.joint_limit_penalty = False
        self.tray_object = tray_object
        self.observe_object = tray_object and observe_object
        self.observe_slide_angle = self.observe_object
        self.energy_penalty = energy_penalty
        self.wrist_leveler = wrist_leveler
        self.outcome_reward = outcome_reward
        self.leveler_tuck = 0.0
        self.ee_first_fraction, self.ee_start_pool, self.ee_pool, self.ee_first = 0.0, 0, None, False
        self.tilt_max = TILT_MAX
        self.start_spread = start_spread
        self.cabinet_spread = cabinet_spread
        self.renderer = None
        self.ik_data = mujoco.MjData(self.model)

        self.home = self.model.key("home").qpos.copy()
        self.ctrl_range = self.model.actuator_ctrlrange.copy()
        self.link7 = self.model.body("link7").id
        self.plate_geom = self.model.geom("plate").id
        self.object_geom = self.model.geom("tray_object").id if tray_object else -1
        if tray_object:
            joint = self.model.joint(self.model.body("tray_object").jntadr[0])
            self.object_qpos, self.object_qvel = joint.qposadr[0], joint.dofadr[0]
        self.fixture_body = self.cabinet_body = self.model.body(spec_["fixture"]).id
        self.mover_body = self.door_body = self.model.body(spec_["mover"]).id
        self.cabinet_home = self.model.body_pos[self.fixture_body].copy()
        quat = self.model.body_quat[self.fixture_body]
        self.fixture_yaw0 = 2.0 * np.arctan2(quat[3], quat[0])  # fixtures are only turned about z
        self.joint_ids = [self.model.joint(name).id for name in spec_["joints"]]
        self.qpos_adr = np.array([self.model.jnt_qposadr[j] for j in self.joint_ids])
        self.qvel_adr = np.array([self.model.jnt_dofadr[j] for j in self.joint_ids])
        self.mover_geoms = [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == self.mover_body]
        self.mover_geom_set = set(self.mover_geoms)
        self.door_geom = self.mover_geoms[0]
        self.torque_limit = self.model.actuator_forcerange[:, 1].copy()
        self.geom_group = np.full(self.model.ngeom, -1)
        for geom in range(self.model.ngeom):
            body = self.model.body(self.model.geom_bodyid[geom]).name
            if body in BODY_GROUP:
                self.geom_group[geom] = CONTACT_GROUPS.index(BODY_GROUP[body])
        self.distal_geoms = [g for g in range(self.model.ngeom) if self.geom_group[g] > 0 and self.model.geom_contype[g]]
        self.ee_geoms = [g for g in range(self.model.ngeom)
                         if self.geom_group[g] == CONTACT_GROUPS.index("ee") and self.model.geom_contype[g]]
        self.allowed_capsules = [g for g in range(self.model.ngeom)
                                 if self.model.geom(g).name.startswith(("link4_c", "link5_c", "link6_c"))]

        self.action_space = gym.spaces.Box(-1.0, 1.0, (4 if wrist_leveler else 7,), np.float32)
        self.limited_joints = np.flatnonzero(np.ptp(self.ctrl_range, axis=1) < 4 * np.pi - 0.01)
        size = 28 + 5 + SLOTS * SLOT_SIZE + 6 * self.observe_object
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (size,), np.float32)
        self.q_target = self.home[:7].copy()
        self.last_action = np.zeros(7)
        self.goal = np.zeros(len(self.joint_ids))
        self.outward_local = np.zeros(3)

    # ---------- start state ----------

    def _draw_start(self):
        shift, turn = self.cabinet_spread
        self.cabinet_yaw = self.np_random.uniform(-turn, turn)
        if self.tray_object:
            self.tilt_max = self.np_random.uniform(*SLIDE_ANGLE_RANGE)
            self.model.geom_friction[self.object_geom, 0] = np.tan(self.tilt_max)
        self.model.body_pos[self.fixture_body] = self.cabinet_home + np.append(
            self.np_random.uniform(-shift, shift, 2), 0.0
        )
        mujoco.mju_axisAngle2Quat(self.model.body_quat[self.fixture_body], [0.0, 0.0, 1.0],
                                  self.fixture_yaw0 + self.cabinet_yaw)
        self._draw_goal()
        lo, hi = self.spec_["start"]
        while True:
            for adr in self.qpos_adr:
                self.data.qpos[adr] = self.np_random.uniform(lo, hi) if hi > lo else lo
            spread = self.np_random.uniform(-self.start_spread, self.start_spread, 7)
            self.data.qpos[:7] = np.clip(self.home[:7] + spread, self.ctrl_range[:, 0], self.ctrl_range[:, 1])
            self._level_plate()
            if self.tray_object:
                self._place_object()
            mujoco.mj_forward(self.model, self.data)
            tray_clear = not self.tray_object or self.data.geom_xpos[self.plate_geom][2] > MIN_TRAY_HEIGHT
            if self.data.ncon == 0 and self._plate_tilt() < START_TILT and self._outside_door() and tray_clear:
                break
        return (self.model.body_pos[self.fixture_body].copy(), self.cabinet_yaw, self.tilt_max,
                self.data.qpos.copy())

    def _draw_goal(self):
        """Fixed goals draw nothing, so T0 keeps A20's random sequence."""
        kind = self.spec_["goal"]
        if kind[0] == "fixed":
            self.goal = np.array(kind[1], float)
            direction = None
        elif kind[0] == "axis":
            _, lo, hi, sign = kind
            self.goal = np.array([sign * self.np_random.uniform(lo, hi)])
            direction = sign * self.model.jnt_axis[self.joint_ids[0]]
        else:
            _, lo, hi, a_lo, a_hi = kind
            distance = self.np_random.uniform(lo, hi) if hi > lo else lo
            angle = self.np_random.uniform(a_lo, a_hi)
            self.goal = distance * np.array([np.cos(angle), np.sin(angle)])
            direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        outward = self.spec_["outward"]
        self.outward_local = np.array(outward, float) if outward is not None else -direction

    def _outside_door(self):
        """Every distal arm geom lies on the push side of the plane through the mover's origin."""
        outward = self.data.xmat[self.mover_body].reshape(3, 3) @ self.outward_local
        offsets = self.data.geom_xpos[self.distal_geoms] - self.data.xpos[self.mover_body]
        return bool(np.all(offsets @ outward > 0.0))

    def _finish_reset(self):
        self.start_distance = self._distance()
        self.start_angle = self.start_distance
        self.link_distance = self._link_distance()
        self.closing = self.valid_closing = 0.0
        self.wrong_contact_steps = 0
        self.returns = dict.fromkeys(REWARD_TERMS, 0.0)
        self.object_point_velocity = self._object_point_velocity()
        self.friction_required, self.coupling_sum, self.steps = 0.0, 0.0, 0
        return self._obs(), self._info(self._contact_forces(), penetration=0.0)

    # ---------- task geometry ----------

    def _distance(self):
        return float(np.linalg.norm(self.data.qpos[self.qpos_adr] - self.goal))

    def _link_distance(self):
        fromto = np.zeros(6)
        return max(0.0, min(
            mujoco.mj_geomDistance(self.model, self.data, a, m, 1.0, fromto)
            for a in self.allowed_capsules for m in self.mover_geoms
        ))

    def _contact_forces(self):
        """Normal force on the mover from each arm group, then arm force on everything else."""
        forces = np.zeros(len(CONTACT_GROUPS) + 1)
        wrench = np.zeros(6)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if self.object_geom in (contact.geom1, contact.geom2):
                continue
            mujoco.mj_contactForce(self.model, self.data, i, wrench)
            if contact.geom1 in self.mover_geom_set and self.geom_group[contact.geom2] >= 0:
                forces[self.geom_group[contact.geom2]] += wrench[0]
            elif contact.geom2 in self.mover_geom_set and self.geom_group[contact.geom1] >= 0:
                forces[self.geom_group[contact.geom1]] += wrench[0]
            elif self.geom_group[contact.geom1] >= 0 or self.geom_group[contact.geom2] >= 0:
                forces[-1] += wrench[0]
        return forces

    def descriptor(self):
        slots = np.zeros((SLOTS, SLOT_SIZE))
        for i, joint in enumerate(self.joint_ids):
            hinge = self.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE
            slots[i] = np.concatenate([
                [1.0, float(hinge), float(not hinge)],
                self.data.xaxis[joint], self.data.xanchor[joint],
                [self.data.qpos[self.qpos_adr[i]], self.data.qvel[self.qvel_adr[i]],
                 self.goal[i], self.spec_["range"]],
            ])
        return slots.ravel()

    # ---------- step and reward ----------

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

        distance_before = self._distance()
        forces = np.zeros(len(CONTACT_GROUPS) + 1)
        penetration = 0.0
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
            forces += self._contact_forces()
            if self.data.ncon:
                penetration = max(penetration, -self.data.contact.dist.min())
        forces /= FRAME_SKIP

        terms, terminated = self._reward(action, forces, distance_before - self._distance())
        for name, value in terms.items():
            self.returns[name] += value
        self.last_action = action
        info = self._info(forces, penetration)
        return self._obs(), float(sum(terms.values())), terminated, False, info

    def _reward(self, action, forces, closed):
        """`closed` is this step's decrease of the distance to the goal, in the joint's units."""
        allowed = forces[1:4].sum()
        wrong = forces[0] + forces[4] + forces[5]
        contact_goal = allowed > F_MIN and wrong == 0.0
        gate = contact_goal or not self.gated or self.outcome_reward
        self.closing += max(closed, 0.0)
        self.valid_closing += contact_goal * max(closed, 0.0)
        self.wrong_contact_steps += wrong > 0.0

        distance_before, self.link_distance = self.link_distance, self._link_distance()
        self.friction_required = self._required_friction()
        self.coupling_sum += 1.0
        self.steps += 1
        weight = W_PROGRESS * DOOR_RANGE / self.spec_["range"]
        terms = dict.fromkeys(REWARD_TERMS, 0.0)
        terms["progress"] = weight * (gate * max(closed, 0.0) - max(-closed, 0.0))
        if not self.outcome_reward:
            terms["reach"] = W_REACH * (np.exp(-self.link_distance / REACH_SCALE) - np.exp(-distance_before / REACH_SCALE))
            terms["wrong"] = -W_WRONG * (self.gated and wrong > 0.0)
        if not self.tray_object:
            terms["tilt"] = -W_TILT * max(self._plate_tilt() - TILT_FREE, 0.0)
        terms["force"] = -W_FORCE * min(max(allowed + wrong - F_MAX, 0.0) / F_MAX, 1.0)
        if self.energy_penalty:
            terms["energy"] = -W_ENERGY * np.sum(np.abs(self.data.qfrc_actuator[:7] * self.data.qvel[:7]))
        else:
            terms["smooth"] = -(
                W_ACTION_RATE * np.sum((action - self.last_action) ** 2)
                + W_JOINT_VEL * np.sum(self.data.qvel[:7] ** 2)
                + W_TORQUE * np.sum((self.data.qfrc_actuator[:7] / self.torque_limit) ** 2)
            )
        distance = self._distance()
        if distance < self.spec_["tol"]:
            earned = self.valid_closing if self.gated or self.outcome_reward else self.closing
            terms["success"] = SUCCESS_BONUS * min(earned / max(self.start_distance - distance, 1e-9), 1.0)
            return terms, True
        if self._spilled():
            terms["spill"] = -SPILL_PENALTY
            return terms, True
        return terms, False

    def _obs(self):
        q, qvel = self.data.qpos, self.data.qvel
        yaw = self.fixture_yaw0 + self.cabinet_yaw
        obs = [
            q[:7], qvel[:7], self.q_target - q[:7], self.last_action,
            self.model.body_pos[self.fixture_body][:2], [np.cos(yaw), np.sin(yaw)],
            [self.tilt_max if self.observe_slide_angle else TILT_MAX],
            self.descriptor(),
        ]
        if self.observe_object:
            to_plate = self.data.geom_xmat[self.plate_geom].reshape(3, 3).T
            offset = self.data.geom_xpos[self.object_geom] - self.data.geom_xpos[self.plate_geom]
            obs += [to_plate @ offset, to_plate @ qvel[self.object_qvel : self.object_qvel + 3]]
        return np.concatenate(obs).astype(np.float32)

    def _info(self, forces, penetration):
        info = super()._info(forces, penetration)
        distance = self._distance()
        info["task/door_angle"] = distance  # kept under the door's name so existing scripts read it
        info["task/distance"] = distance
        info["task/distance_norm"] = distance / self.spec_["range"]
        info["task/success"] = float(distance < self.spec_["tol"])
        return info
