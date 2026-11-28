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
    CloseCabinetEnv, add_tray_object, plate_mount,
)

ASSETS = Path(__file__).parents[1] / "assets"
DOOR_RANGE = 1.2  # rad; W_PROGRESS is per rad of door, so a task's progress weight is W_PROGRESS · 1.2 / range
SLOTS = 2
CONTACT_SIZE = 8  # target surface point minus forearm point (3), push direction (3), d/range, ḋ/range
FK_SIZE = 9  # elbow (link 4 origin), forearm (link5_c2 centre), wrist (link 6 origin), world frame
MAX_START_TRIES = 1000
HINGE_FRACTION = 2 / 3  # where along a hinged door the push point sits (0 hinge, 1 free edge)  # arm draws per fixture pose before the pose is redrawn
SLOT_SIZE = 13
LABEL_SLOT = (0, 9, 10, 11, 12)  # valid, s, ṡ, s*, range: the slot without the joint's geometry

# outward: unit vector in the mover's frame; at reset every distal arm geom must lie on that side
# of the plane through the mover's origin, so the arm starts on the side it has to push from.
# goal: ("fixed", values) or ("axis", lo, hi, sign) for a distance along the single joint, or
# ("planar", lo, hi, angle_lo, angle_hi) for a direction in the fixture's xy plane.
# home_yaw (optional): radians added to joint 1 of the home posture the start is drawn around,
# so the arm can start on the far side of the object for negative goals.
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
    "T4xneg": dict(scene="task_box_x.xml", fixture="fixture", mover="mover", joints=("slide_x",),
                   start=(0.0, 0.0), goal=("axis", 0.20, 0.30, -1.0), range=0.25, tol=0.02, outward=None,
                   home_yaw=0.9),
    # Procedural family (TASK_SET.md, 11/25 follow-up): the T4 box on one rail, redrawn every episode
    # in push direction (world azimuth, from `arcs`; each arc has its own home_yaw so the arm starts
    # on the push side), position (cabinet_spread) and box size. The goal is always ahead along the
    # rail. `held_out` removes a band of push azimuths from training.
    "F": dict(scene="task_box_x.xml", fixture="fixture", mover="mover", joints=("slide_x",),
              start=(0.0, 0.0), goal=("axis", 0.20, 0.30, 1.0), range=0.25, tol=0.02, outward=None,
              # Arcs are where a scripted forearm push solves >= 60% of starts (scripts/family_feasibility.py,
              # 2026-11-25: -10..110 deg 80-100%, 210..250 deg 60-65%; -30..-10 and 250..270 fail). The held-out
              # band covers T2's push directions (30 deg ± the 20 deg cabinet turn), so T2 is unseen in both
              # object and direction.
              family=dict(arcs=((np.radians(-10), np.radians(110), 0.0), (np.radians(210), np.radians(250), 0.9)),
                          held_out=(np.radians(10), np.radians(50)), half_xy=(0.04, 0.07), half_z=(0.10, 0.14))),
    "T4y": dict(scene="task_box_y.xml", fixture="fixture", mover="mover", joints=("slide_y",),
                start=(0.0, 0.0), goal=("axis", 0.20, 0.30, -1.0), range=0.25, tol=0.02, outward=None),
    "T4xy": dict(scene="task_box_xy.xml", fixture="fixture", mover="mover", joints=("slide_x", "slide_y"),
                 start=(0.0, 0.0), goal=("planar", 0.25, 0.25, -np.pi / 6, np.pi / 6), range=0.25, tol=0.02,
                 outward=None),
}


class TaskEnv(CloseCabinetEnv):

    def __init__(self, task="T0", render_mode=None, gated=True, start_spread=0.5,
                 cabinet_spread=(0.1, np.radians(20)), tray_object=True, observe_object=False,
                 energy_penalty=True, wrist_leveler=True, outcome_reward=False, labels=None,
                 conditioning="descriptor", family=None):
        spec_ = dict(TASKS[task])
        if family is not None:  # override parts of the family spec (e.g. a held-out band)
            spec_["family"] = {**spec_["family"], **family}
        self.task = task
        self.spec_ = spec_
        spec = mujoco.MjSpec.from_file(str(ASSETS / spec_["scene"]))
        self.plate_mount = plate_mount(spec)
        if tray_object:
            add_tray_object(spec)
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
        # Option (a) of TASK_SET.md: with `labels` (the training tasks, in order), the joint's type,
        # axis and anchor leave the observation and a one-hot over `labels` takes their place. A task
        # not in `labels` gets all zeros, so a held-out task carries no task information at all.
        self.labels = tuple(labels) if labels is not None else None
        # How the task is described (TASK_SET.md): "descriptor" (joint type, axis, anchor, s, ṡ, goal,
        # range) or the label form above; "contact" replaces the fixture pose and descriptor with the
        # contact target (CONTACT_SIZE numbers, see contact_features); "contact_fk" adds Cartesian
        # positions of the elbow, forearm and wrist from forward kinematics (FK_SIZE numbers).
        self.conditioning = "label" if self.labels is not None else conditioning
        if self.conditioning == "descriptor":
            size = 28 + 5 + SLOTS * SLOT_SIZE
        elif self.conditioning == "label":
            size = 28 + 5 + SLOTS * len(LABEL_SLOT) + len(self.labels)
        elif self.conditioning == "contact":
            size = 28 + 1 + CONTACT_SIZE
        elif self.conditioning == "contact_fk":
            size = 28 + 1 + CONTACT_SIZE + FK_SIZE
        else:
            raise ValueError(f"unknown conditioning {conditioning!r}")
        size += 6 * self.observe_object
        self.forearm_geom = self.model.geom("link5_c2").id
        self.fk_bodies = [self.model.body(name).id for name in ("link4", "link6")]
        if "family" in spec_:
            self.box_geom = self.model.geom("box").id
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (size,), np.float32)
        self.q_target = self.home[:7].copy()
        self.last_action = np.zeros(7)
        self.goal = np.zeros(len(self.joint_ids))
        self.outward_local = np.zeros(3)

    # ---------- start state ----------

    def _draw_start(self):
        """Draw a fixture pose and goal, then arm starts until one is valid. Some poses admit none
        (2% of T4xneg's far-side draws), so after MAX_START_TRIES the pose and goal are redrawn.
        The other tasks need at most a handful of tries, so their draws, and T0's match with A20,
        are unchanged."""
        while True:
            start = self._draw_start_for_pose()
            if start is not None:
                return start

    def _draw_start_for_pose(self):
        shift, turn = self.cabinet_spread
        self.cabinet_yaw = self.np_random.uniform(-turn, turn)
        if self.tray_object:
            self.tilt_max = self.np_random.uniform(*SLIDE_ANGLE_RANGE)
            self.model.geom_friction[self.object_geom, 0] = np.tan(self.tilt_max)
        self.model.body_pos[self.fixture_body] = self.cabinet_home + np.append(
            self.np_random.uniform(-shift, shift, 2), 0.0
        )
        home_yaw = self.spec_.get("home_yaw", 0.0)
        if "family" in self.spec_:  # drawn only for the family, so every fixed task's sequence is unchanged
            azimuth, home_yaw = self._draw_family()
            self.cabinet_yaw = azimuth - self.fixture_yaw0
        mujoco.mju_axisAngle2Quat(self.model.body_quat[self.fixture_body], [0.0, 0.0, 1.0],
                                  self.fixture_yaw0 + self.cabinet_yaw)
        self._draw_goal()
        lo, hi = self.spec_["start"]
        home = self.home[:7].copy()
        home[0] += home_yaw
        for _ in range(MAX_START_TRIES):
            for adr in self.qpos_adr:
                self.data.qpos[adr] = self.np_random.uniform(lo, hi) if hi > lo else lo
            spread = self.np_random.uniform(-self.start_spread, self.start_spread, 7)
            self.data.qpos[:7] = np.clip(home + spread, self.ctrl_range[:, 0], self.ctrl_range[:, 1])
            self._level_plate()
            if self.tray_object:
                self._place_object()
            mujoco.mj_forward(self.model, self.data)
            tray_clear = not self.tray_object or self.data.geom_xpos[self.plate_geom][2] > MIN_TRAY_HEIGHT
            if self.data.ncon == 0 and self._plate_tilt() < START_TILT and self._outside_door() and tray_clear:
                return (self.model.body_pos[self.fixture_body].copy(), self.cabinet_yaw, self.tilt_max,
                        self.data.qpos.copy())
        return None

    def _draw_family(self):
        """Push azimuth (outside any held-out band), the arc's home_yaw, and a new box size."""
        fam = self.spec_["family"]
        while True:
            lo, hi, home_yaw = fam["arcs"][self.np_random.integers(len(fam["arcs"]))]
            azimuth = self.np_random.uniform(lo, hi)
            band = fam.get("held_out")
            wrapped = (azimuth + np.pi) % (2 * np.pi) - np.pi
            if band is None or not band[0] <= wrapped <= band[1]:
                break
        half = np.array([*self.np_random.uniform(*fam["half_xy"], 2), self.np_random.uniform(*fam["half_z"])])
        self.model.geom_size[self.box_geom] = half
        self.model.geom_rbound[self.box_geom] = np.linalg.norm(half)
        self.model.geom_aabb[self.box_geom] = np.concatenate([np.zeros(3), half])
        return azimuth, home_yaw

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

    def _hinged(self):
        return self.model.jnt_type[self.joint_ids[0]] == mujoco.mjtJoint.mjJNT_HINGE

    def _hinge_point(self):
        """Hinged mover (one door geom on one hinge): the point HINGE_FRACTION of the way out along the door, on its
        arm-side face, in the world frame (the push point scripts/scripted_baseline.py validated on T0)."""
        door = self.mover_geoms[0]
        half = self.model.geom_size[door]
        local = (self.model.geom_pos[door] + np.array([(2 * HINGE_FRACTION - 1) * half[0], 0.0, 0.0])
                 + half[1] * np.asarray(self.outward_local))
        return self.data.xpos[self.mover_body] + self.data.xmat[self.mover_body].reshape(3, 3) @ local

    def push_direction(self):
        """World direction the push point moves when the mover moves toward the goal. Slide joints: along the
        joint axes. Hinge (2026-11-28; before this it returned the hinge axis, so T0's contact features were
        meaningless): tangent to the push point's arc, toward closing."""
        toward = self.goal - self.data.qpos[self.qpos_adr]
        if self._hinged():
            joint = self.joint_ids[0]
            world = np.cross(self.data.xaxis[joint], self._hinge_point() - self.data.xanchor[joint]) * np.sign(toward[0])
        else:
            world = sum(toward[i] * self.data.xaxis[j] for i, j in enumerate(self.joint_ids))
        norm = np.linalg.norm(world)
        return world / norm if norm > 1e-9 else np.zeros(3)

    def contact_target(self):
        """The surface point to push. Slide joints: where a ray along the push direction, started 0.4 m behind the
        mover's origin, first meets one of the mover's geoms. Hinge: the push point on the door's arm-side face."""
        direction = self.push_direction()
        if self._hinged():
            return self._hinge_point(), direction
        start = self.data.xpos[self.mover_body] - 0.4 * direction
        best = np.inf
        for g in self.mover_geoms:
            dist = mujoco.mju_rayGeom(self.data.geom_xpos[g], self.data.geom_xmat[g], self.model.geom_size[g],
                                      start, direction, self.model.geom_type[g])
            if 0 <= dist < best:
                best = dist
        return start + best * direction if np.isfinite(best) else self.data.xpos[self.mover_body].copy(), direction

    def contact_features(self):
        target, direction = self.contact_target()
        forearm = self.data.geom_xpos[self.forearm_geom]
        speed = -float(np.dot(self.data.qvel[self.qvel_adr], self.goal - self.data.qpos[self.qpos_adr])) / max(self._distance(), 1e-9)
        return np.concatenate([target - forearm, direction,
                               [self._distance() / self.spec_["range"], speed / self.spec_["range"]]])

    def fk_features(self):
        return np.concatenate([self.data.xpos[self.fk_bodies[0]], self.data.geom_xpos[self.forearm_geom],
                               self.data.xpos[self.fk_bodies[1]]])

    def condition(self):
        """What tells the policy which task this is: the descriptor, or the label form of it."""
        slots = self.descriptor()
        if self.labels is None:
            return slots
        one_hot = np.array([float(task == self.task) for task in self.labels])
        return np.concatenate([slots.reshape(SLOTS, SLOT_SIZE)[:, LABEL_SLOT].ravel(), one_hot])

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
        arm = [q[:7], qvel[:7], self.q_target - q[:7], self.last_action]
        tilt = [self.tilt_max if self.observe_slide_angle else TILT_MAX]
        if self.conditioning in ("descriptor", "label"):
            obs = arm + [self.model.body_pos[self.fixture_body][:2], [np.cos(yaw), np.sin(yaw)], tilt, self.condition()]
        else:
            obs = arm + [tilt, self.contact_features()]
            if self.conditioning == "contact_fk":
                obs.append(self.fk_features())
        if self.observe_object:
            to_plate = self._plate_frame().T
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


class MultiTaskEnv(gym.Env):
    """One TaskEnv per task; every reset draws a task uniformly. Spaces are identical across
    tasks, so the constraint descriptor in the observation is the only thing that tells the
    policy which task it faces. `task/index` in info says which one ran."""

    metadata = TaskEnv.metadata

    def __init__(self, tasks=("T0", "T1", "T4x"), task_labels=False, **kwargs):
        self.tasks = tuple(tasks)
        labels = self.tasks if task_labels else None
        self.envs = [TaskEnv(task=task, labels=labels, **kwargs) for task in self.tasks]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.current = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current = int(self.np_random.integers(len(self.envs)))
        sub_seed = int(self.np_random.integers(2**31 - 1))
        obs, info = self.envs[self.current].reset(seed=sub_seed)
        info["task/index"] = float(self.current)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.envs[self.current].step(action)
        info["task/index"] = float(self.current)
        return obs, reward, terminated, truncated, info

    def close(self):
        for env in self.envs:
            env.close()
