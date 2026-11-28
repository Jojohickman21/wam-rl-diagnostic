"""Batched MuJoCo Warp version of TaskEnv's training families with the G3 setup (log.md 2026-11-28).

N worlds step on one device. Physics is MuJoCo Warp (MuJoCo >= 4); everything around it is batched torch on zero-copy
views of the Warp arrays, written to match TaskEnv(task, conditioning="contact_fk") with its defaults: the A20
wrist leveler on joints 5-7, the dense gated reward with the energy term, the tray object and spill, 250-step episodes.

Differences from the C env, all deliberate:
- Starts come from a pool drawn by the C env itself (`start_pool`: TaskEnv.reset, i.e. rejection sampling and the
  250 ms settle), sampled with replacement, instead of a fresh draw per episode.
- The reach term's forearm-to-object distance is approximated: REACH_POINTS points along each allowed capsule
  (links 4-6), distance to the box minus the capsule radius (Josiah's option (a); C uses mj_geomDistance).
- Supported tasks: one box-shaped mover on one joint -- family F (slide) and family H (hinged door). Each episode's
  family geometry (sizes, positions, the hinge axis) is a per-world model field. Policies are evaluated in C MuJoCo.

Training only: evaluation of reported policies stays in C MuJoCo on frozen v2 (sim-backend decision, log.md).
"""
from contextlib import nullcontext as _null
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from mujoco_warp._src import collision_convex

from wam.close_cabinet import (
    ACTION_SCALE, CONTACT_GROUPS, F_MAX, F_MIN, FRAME_SKIP, MAX_LEAD, OBJECT_HALF_HEIGHT, PLATE_HALF_THICKNESS,
    PLATE_RADIUS, REACH_SCALE, SPILL_PENALTY, SUCCESS_BONUS, TILT_MAX, W_ENERGY, W_FORCE, W_PROGRESS, W_REACH, W_WRONG,
)
from wam.tasks import DOOR_RANGE, HINGE_FRACTION, TaskEnv

# Warp's EPA horizon buffer (24 edges) overflows on ~5% of worlds with the cylinder tray contacts, which can corrupt
# contact depth (scripts/warp/bench_physics.py). The kernel bounds by the array's size, so allocate it larger.
collision_convex.MJ_MAX_EPAHORIZON = 64
EPISODE_STEPS = 250
REACH_POINTS = 5
LEVELER_ITERATIONS = 3
BATCHED = ("body_pos", "body_quat", "geom_friction", "geom_size", "geom_pos", "geom_rbound", "geom_aabb", "jnt_axis")


@wp.kernel
def _accumulate_forces(nacon: wp.array(dtype=int), geom: wp.array(dtype=wp.vec2i), worldid: wp.array(dtype=int),
                       force: wp.array(dtype=wp.spatial_vector), group: wp.array(dtype=int), mover: wp.array(dtype=int),
                       obj: int, scene_column: int, out: wp.array2d(dtype=float)):
    """TaskEnv._contact_forces for one substep, all worlds: normal force on the mover per arm group, then arm on
    anything else (scene_column); tray-object contacts skipped."""
    i = wp.tid()
    if i >= nacon[0]:
        return
    g1, g2 = geom[i][0], geom[i][1]
    if g1 == obj or g2 == obj:
        return
    a1, a2 = group[g1], group[g2]
    column = -1
    if mover[g1] == 1 and a2 >= 0:
        column = a2
    elif mover[g2] == 1 and a1 >= 0:
        column = a1
    elif a1 >= 0 or a2 >= 0:
        column = scene_column
    if column >= 0:
        wp.atomic_add(out, worldid[i], column, force[i][0])


POOL_KEYS = ("qpos", "qvel", "q_target", "goal", "fixture_pos", "fixture_quat", "friction", "geom_size", "geom_pos",
             "geom_rbound", "geom_aabb", "jnt_axis")


def _draw_starts(args):
    task, seeds = args
    env = TaskEnv(task=task, conditioning="contact_fk")
    g, j = env.family_geoms, env.joint_ids[0]
    rows = {k: [] for k in POOL_KEYS}
    for s in seeds:
        env.reset(seed=int(s))
        m, d = env.model, env.data
        rows["qpos"].append(d.qpos.copy()), rows["qvel"].append(d.qvel.copy())
        rows["q_target"].append(env.q_target.copy()), rows["goal"].append(env.goal.copy())
        rows["fixture_pos"].append(m.body_pos[env.fixture_body].copy())
        rows["fixture_quat"].append(m.body_quat[env.fixture_body].copy())
        rows["friction"].append(m.geom_friction[env.object_geom].copy())
        rows["geom_size"].append(m.geom_size[g].copy()), rows["geom_pos"].append(m.geom_pos[g].copy())
        rows["geom_rbound"].append(m.geom_rbound[g].copy()), rows["geom_aabb"].append(m.geom_aabb[g].reshape(-1, 2, 3).copy())
        rows["jnt_axis"].append(m.jnt_axis[j].copy())
    return {k: np.array(v) for k, v in rows.items()}


def start_pool(n, seed, path=None, workers=1, task="F"):
    """n starts drawn by the C env (TaskEnv `task`, G3 kwargs), start i with reset seed `seed + i`: the state after
    reset and the per-episode model values. Split over `workers` processes with the same result. Cached at `path`."""
    if path is not None and Path(path).exists():
        pool = dict(np.load(path))
        if "box_size" in pool:  # pre-H format (F only: one family geom, the slide axis of task_box_x.xml)
            pool["geom_size"], pool["geom_rbound"] = pool.pop("box_size")[:, None], pool.pop("box_rbound")[:, None]
            pool["geom_aabb"] = pool.pop("box_aabb")[:, None]
            pool["geom_pos"] = np.zeros_like(pool["geom_size"])
            pool["jnt_axis"] = np.tile([1.0, 0.0, 0.0], (len(pool["qpos"]), 1))
        return pool
    seeds = np.arange(seed, seed + n)
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as p:
            parts = p.map(_draw_starts, [(task, chunk) for chunk in np.array_split(seeds, workers)])
    else:
        parts = [_draw_starts((task, seeds))]
    pool = {k: np.concatenate([part[k] for part in parts]) for k in POOL_KEYS}
    if path is not None:
        np.savez(path, **pool)
    return pool


class WarpFamilyEnv:
    def __init__(self, nworld, pool, device="cuda:0", seed=0, nconmax=48, njmax=256, task="F"):
        self.n, self.device, self.task = nworld, torch.device(device), task
        c = TaskEnv(task=task, conditioning="contact_fk")
        c.reset(seed=0)
        self.c, m = c, c.model
        self.range, self.tol = c.spec_["range"], c.spec_["tol"]
        with wp.ScopedDevice(device):
            self.m = mjw.put_model(m, batch_sizes={f: nworld for f in BATCHED})
            self.d = mjw.put_data(m, c.data, nworld=nworld, nconmax=nconmax, njmax=njmax)
            self.ik = mjw.put_data(m, c.data, nworld=nworld, nconmax=nconmax, njmax=njmax)
            self.contact_ids = wp.array(np.arange(self.d.naconmax, dtype=np.int32), dtype=int)
            self.contact_force = wp.zeros(self.d.naconmax, dtype=wp.spatial_vector)
            self.force_sum = wp.zeros((nworld, len(CONTACT_GROUPS) + 1), dtype=float)
            self.group_wp = wp.array(np.asarray(c.geom_group, dtype=np.int32), dtype=int)
            mover_mask = np.zeros(m.ngeom, dtype=np.int32)
            mover_mask[list(c.mover_geoms)] = 1
            self.mover_wp = wp.array(mover_mask, dtype=int)
        t = wp.to_torch
        d, ik, mw = self.d, self.ik, self.m
        self.qpos, self.qvel, self.ctrl = t(d.qpos), t(d.qvel), t(d.ctrl)
        self.xpos, self.xmat, self.xaxis, self.xanchor = t(d.xpos), t(d.xmat), t(d.xaxis), t(d.xanchor)
        self.geom_xpos, self.geom_xmat, self.qfrc_actuator = t(d.geom_xpos), t(d.geom_xmat), t(d.qfrc_actuator)
        self.qacc_warmstart, self.act = t(d.qacc_warmstart), t(d.act) if d.act.shape[-1] else None
        self.con_geom, self.con_world, self.nacon = t(d.contact.geom), t(d.contact.worldid), t(d.nacon)
        self.con_force = t(self.contact_force)
        self.ik_qpos, self.ik_xmat, self.ik_xaxis = t(ik.qpos), t(ik.xmat), t(ik.xaxis)
        self.force_sum_t = t(self.force_sum)
        self.body_pos, self.body_quat, self.geom_friction = t(mw.body_pos), t(mw.body_quat), t(mw.geom_friction)
        self.geom_size, self.geom_rbound, self.geom_aabb = t(mw.geom_size), t(mw.geom_rbound), t(mw.geom_aabb)
        self.geom_pos, self.jnt_axis = t(mw.geom_pos), t(mw.jnt_axis)

        dev, f32 = self.device, torch.float32
        as_t = lambda x, dtype=f32: torch.as_tensor(np.asarray(x), dtype=dtype, device=dev)
        self.pool = {k: as_t(v) for k, v in pool.items()}
        self.ctrl_lo, self.ctrl_hi = as_t(c.ctrl_range[:, 0]), as_t(c.ctrl_range[:, 1])
        self.plate_mount = as_t(c.plate_mount)
        # Contact bookkeeping: arm group per geom (-1 = not arm), which geoms belong to the mover, the tray object.
        self.geom_group = as_t(c.geom_group, torch.long)
        mover = torch.zeros(m.ngeom, dtype=torch.bool, device=dev)
        mover[list(c.mover_geoms)] = True
        self.mover_geom = mover
        self.ids = dict(link7=c.link7, plate=c.plate_geom, obj=c.object_geom, box=c.mover_geoms[0], fixture=c.fixture_body,
                        mover=c.mover_body, forearm=c.forearm_geom, link4=c.fk_bodies[0], link6=c.fk_bodies[1],
                        task_q=int(c.qpos_adr[0]), task_v=int(c.qvel_adr[0]), task_j=int(c.joint_ids[0]),
                        obj_q=int(c.object_qpos))
        assert [m.geom_type[g] for g in c.mover_geoms] == [mujoco.mjtGeom.mjGEOM_BOX], "the mover must be one box"
        self.family_geoms = as_t(c.family_geoms, torch.long)
        # Warp places geoms of bodies welded to the world once, at put_data, and never again; a family that moves the
        # fixture or resizes its geoms (H's carcass) must place them itself after every per-world change.
        static = [g for g in c.family_geoms if m.geom_bodyid[g] == c.fixture_body]
        assert all(np.allclose(m.geom_quat[g], (1, 0, 0, 0)) for g in static), "static family geoms must be unrotated"
        self.static_geoms = as_t(static, torch.long)
        self.hinged = bool(c._hinged())
        self.outward_local = as_t(c.spec_["outward"] if c.spec_["outward"] is not None else (0.0, 0.0, 0.0))
        caps = list(c.allowed_capsules)
        self.caps = as_t(caps, torch.long)
        self.cap_radius, self.cap_half = as_t(m.geom_size[caps, 0]), as_t(m.geom_size[caps, 1])
        self.cap_t = torch.linspace(-1.0, 1.0, REACH_POINTS, device=dev)
        self.wrist_joints = torch.arange(4, 7, device=dev)
        self.gen = torch.Generator(device=dev).manual_seed(seed)
        # One stream for torch and Warp, so torch's writes (ctrl, resets) are ordered before Warp's launches; then the
        # 20-substep loop (physics, contact forces, accumulation) is captured once as a CUDA graph.
        # CUDA cannot capture a graph on the legacy default stream, so torch moves to a dedicated stream for the rest of
        # the process (the trainer's torch work included) and Warp shares it.
        self.stream = None
        if dev.type == "cuda":
            torch_stream = torch.cuda.current_stream(dev)
            if torch_stream == torch.cuda.default_stream(dev):  # the first env in the process creates it; later ones share
                torch_stream = torch.cuda.Stream(dev)
                torch.cuda.set_stream(torch_stream)
            self.stream = wp.stream_from_torch(torch_stream)
        with wp.ScopedDevice(device), wp.ScopedStream(self.stream) if self.stream else _null():
            with wp.ScopedCapture() as capture:
                self._substeps()
        self.substep_graph = capture.graph

        n = nworld
        z = lambda *s: torch.zeros(n, *s, device=dev)
        self.q_target, self.last_action, self.goal = z(7), z(7), z()
        self.steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.start_distance, self.closing, self.valid_closing, self.link_distance = z(), z(), z(), z()

    # ---------- geometry ----------

    def _plate_frame(self):
        return self.xmat[:, self.ids["link7"]] @ self.plate_mount

    def _plate_up(self):
        axis = self._plate_frame()[..., 2]
        return torch.where(axis[:, 2:3] >= 0, axis, -axis)

    def _plate_tilt(self):
        return torch.arccos(self.xmat[:, self.ids["link7"], 2, 1].abs().clamp(0.0, 1.0))

    def _distance(self):
        return (self.qpos[:, self.ids["task_q"]] - self.goal).abs()

    def _hinge_point(self):
        """TaskEnv._hinge_point: HINGE_FRACTION of the way along the door from the hinge, on its arm-side face."""
        b, mover = self.ids["box"], self.ids["mover"]
        pos, half = self.geom_pos[:, b], self.geom_size[:, b]
        along = torch.where(pos[:, 0] < 0, -1.0, 1.0)
        local = pos.clone()
        local[:, 0] += along * (2 * HINGE_FRACTION - 1) * half[:, 0]
        local += half[:, 1:2] * self.outward_local
        return self.xpos[:, mover] + (self.xmat[:, mover] @ local[..., None])[..., 0]

    def _push_direction(self):
        toward = self.goal - self.qpos[:, self.ids["task_q"]]
        if self.hinged:
            j = self.ids["task_j"]
            world = torch.cross(self.xaxis[:, j], self._hinge_point() - self.xanchor[:, j], dim=1) * torch.sign(toward)[:, None]
            return world / world.norm(dim=1, keepdim=True).clamp(min=1e-9)
        return torch.sign(toward)[:, None] * self.xaxis[:, self.ids["task_j"]]

    def _box(self):
        b = self.ids["box"]
        return self.geom_xpos[:, b], self.geom_xmat[:, b], self.geom_size[:, b]

    def _contact_target(self):
        """Slide: where a ray along the push direction, started 0.4 m behind the mover, first meets the box (slab
        test). Hinge: the door push point (TaskEnv.contact_target)."""
        direction = self._push_direction()
        if self.hinged:
            return self._hinge_point(), direction
        start = self.xpos[:, self.ids["mover"]] - 0.4 * direction
        center, rot, half = self._box()
        o = ((start - center)[:, None, :] @ rot)[:, 0]
        v = (direction[:, None, :] @ rot)[:, 0]
        v = torch.where(v.abs() < 1e-12, torch.full_like(v, 1e-12), v)
        t1, t2 = (-half - o) / v, (half - o) / v
        t_near = torch.minimum(t1, t2).amax(1)
        t_far = torch.maximum(t1, t2).amin(1)
        hit = (t_near <= t_far) & (t_far >= 0)
        t = torch.where(t_near >= 0, t_near, t_far)
        target = torch.where(hit[:, None], start + t[:, None] * direction, self.xpos[:, self.ids["mover"]])
        return target, direction

    def _link_distance(self):
        """Approximate mj_geomDistance(allowed forearm capsules, box), floored at 0 (option (a), log.md)."""
        center, rot, half = self._box()
        cap_c, cap_axis = self.geom_xpos[:, self.caps], self.geom_xmat[:, self.caps, :, 2]
        points = cap_c[:, :, None] + (self.cap_t[None, None, :, None] * self.cap_half[None, :, None, None]) * cap_axis[:, :, None]
        local = ((points - center[:, None, None]) @ rot[:, None]).abs() - half[:, None, None]
        outside = local.clamp(min=0).norm(dim=-1) + local.amax(-1).clamp(max=0)
        return (outside - self.cap_radius[None, :, None]).clamp(min=0).amin((1, 2))

    def _spilled(self):
        up = self._plate_up()
        offset = self.geom_xpos[:, self.ids["obj"]] - self.geom_xpos[:, self.ids["plate"]]
        height = (offset * up).sum(-1)
        radial = (offset - height[:, None] * up).norm(dim=-1)
        return (radial > PLATE_RADIUS) | (height <= 0.0) | (height >= 0.1)

    # ---------- observation ----------

    def _obs(self):
        q, qvel = self.qpos[:, :7], self.qvel[:, :7]
        target, direction = self._contact_target()
        dist = self._distance()
        s, sv = self.qpos[:, self.ids["task_q"]], self.qvel[:, self.ids["task_v"]]
        speed = -(sv * (self.goal - s)) / dist.clamp(min=1e-9)
        contact = torch.cat([target - self.geom_xpos[:, self.ids["forearm"]], direction,
                             (dist / self.range)[:, None], (speed / self.range)[:, None]], 1)
        fk = torch.cat([self.xpos[:, self.ids["link4"]], self.geom_xpos[:, self.ids["forearm"]], self.xpos[:, self.ids["link6"]]], 1)
        tilt = torch.full((self.n, 1), TILT_MAX, device=self.device)
        return torch.cat([q, qvel, self.q_target - q, self.last_action, tilt, contact, fk], 1)

    # ---------- reset ----------

    def _kinematics(self):
        mjw.kinematics(self.m, self.d)
        mjw.com_pos(self.m, self.d)
        if len(self.static_geoms):
            f = self.ids["fixture"]
            q = self.body_quat[:, f]
            w, x, y, z = q.unbind(1)
            rot = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                               2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                               2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).view(-1, 3, 3)
            g = self.static_geoms
            self.geom_xpos[:, g] = self.body_pos[:, f, None] + (rot[:, None] @ self.geom_pos[:, g, :, None])[..., 0]
            self.geom_xmat[:, g] = rot[:, None]

    def reset_worlds(self, mask):
        idx = mask.nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            return
        k = torch.randint(len(self.pool["qpos"]), (len(idx),), generator=self.gen, device=self.device)
        p = {name: v[k] for name, v in self.pool.items()}
        self.qpos[idx], self.qvel[idx] = p["qpos"], p["qvel"]
        self.qacc_warmstart[idx] = 0.0
        if self.act is not None:
            self.act[idx] = 0.0
        self.q_target[idx], self.goal[idx] = p["q_target"], p["goal"][:, 0]
        self.ctrl[idx] = p["q_target"]
        f, o, g = self.ids["fixture"], self.ids["obj"], self.family_geoms
        self.body_pos[idx, f], self.body_quat[idx, f] = p["fixture_pos"], p["fixture_quat"]
        self.geom_friction[idx, o] = p["friction"]
        rows = idx[:, None]
        self.geom_size[rows, g], self.geom_pos[rows, g] = p["geom_size"], p["geom_pos"]
        self.geom_rbound[rows, g], self.geom_aabb[rows, g] = p["geom_rbound"], p["geom_aabb"]
        self.jnt_axis[idx, self.ids["task_j"]] = p["jnt_axis"]
        self.last_action[idx], self.steps[idx] = 0.0, 0
        self.closing[idx], self.valid_closing[idx] = 0.0, 0.0
        self._kinematics()
        self.start_distance[idx] = self._distance()[idx]
        self.link_distance[idx] = self._link_distance()[idx]

    def reset(self):
        self.reset_worlds(torch.ones(self.n, dtype=torch.bool, device=self.device))
        return self._obs()

    # ---------- step ----------

    def _wrist_targets(self, arm, q):
        """A20 leveler: DLS on the plate normal over joints 5-7, from the measured wrist, arm at its new target."""
        self.ik_qpos.copy_(self.qpos)
        self.ik_qpos[:, :4] = arm
        lo, hi = q[:, 4:7] - MAX_LEAD, q[:, 4:7] + MAX_LEAD
        eye = 1e-4 * torch.eye(3, device=self.device)
        for _ in range(LEVELER_ITERATIONS):
            mjw.kinematics(self.m, self.ik)
            normal = self.ik_xmat[:, self.ids["link7"], :, 1]
            up = torch.zeros_like(normal)
            up[:, 2] = torch.where(normal[:, 2] >= 0, 1.0, -1.0)
            error = torch.cross(normal, up, dim=1)
            jac = self.ik_xaxis[:, self.wrist_joints].transpose(1, 2)  # (n, 3, 3): rotational Jacobian of link 7
            step = (jac.transpose(1, 2) @ torch.linalg.solve(jac @ jac.transpose(1, 2) + eye, error[..., None]))[..., 0]
            self.ik_qpos[:, 4:7] = torch.minimum(torch.maximum(self.ik_qpos[:, 4:7] + step, lo), hi)
        return torch.minimum(torch.maximum(self.ik_qpos[:, 4:7], self.ctrl_lo[4:]), self.ctrl_hi[4:])

    def _substeps(self):
        """FRAME_SKIP physics steps with TaskEnv's per-substep contact-force sum (captured as one graph)."""
        self.force_sum.zero_()
        for _ in range(FRAME_SKIP):
            mjw.step(self.m, self.d)
            mjw.contact_force(self.m, self.d, self.contact_ids, False, self.contact_force)
            wp.launch(_accumulate_forces, dim=self.d.naconmax,
                      inputs=[self.d.nacon, self.d.contact.geom, self.d.contact.worldid, self.contact_force, self.group_wp,
                              self.mover_wp, self.ids["obj"], len(CONTACT_GROUPS), self.force_sum])

    def _forces(self):
        """Torch reference for one substep (parity checks): same rule as _accumulate_forces."""
        mjw.contact_force(self.m, self.d, self.contact_ids, False, self.contact_force)
        valid = torch.arange(len(self.con_world), device=self.device) < self.nacon[0]
        g1, g2 = self.con_geom[:, 0].long(), self.con_geom[:, 1].long()
        obj = self.ids["obj"]
        keep = valid & (g1 != obj) & (g2 != obj)
        grp1, grp2 = self.geom_group[g1], self.geom_group[g2]
        on_mover = torch.where(self.mover_geom[g1] & (grp2 >= 0), grp2, torch.where(self.mover_geom[g2] & (grp1 >= 0), grp1, -1))
        scene = (on_mover < 0) & ((grp1 >= 0) | (grp2 >= 0))
        column = torch.where(on_mover >= 0, on_mover, torch.full_like(on_mover, len(CONTACT_GROUPS)))
        weight = self.con_force[:, 0] * (keep & ((on_mover >= 0) | scene))
        out = torch.zeros(self.n, len(CONTACT_GROUPS) + 1, device=self.device)
        out.index_put_((self.con_world.long().clamp(0, self.n - 1), column), weight, accumulate=True)
        return out

    def step(self, action):
        action = action.clamp(-1.0, 1.0)
        q = self.qpos[:, :7].clone()
        arm = torch.minimum(torch.maximum(self.q_target[:, :4] + ACTION_SCALE * action, q[:, :4] - MAX_LEAD), q[:, :4] + MAX_LEAD)
        arm = torch.minimum(torch.maximum(arm, self.ctrl_lo[:4]), self.ctrl_hi[:4])
        wrist = self._wrist_targets(arm, q)
        full_action = torch.cat([action, (wrist - self.q_target[:, 4:]) / ACTION_SCALE], 1)
        self.q_target = torch.minimum(torch.maximum(torch.cat([arm, wrist], 1), self.ctrl_lo), self.ctrl_hi)
        self.ctrl.copy_(self.q_target)

        distance_before = self._distance()
        with wp.ScopedStream(self.stream) if self.stream else _null():
            wp.capture_launch(self.substep_graph)
        forces = self.force_sum_t / FRAME_SKIP
        distance = self._distance()
        closed = distance_before - distance

        allowed, wrong = forces[:, 1:4].sum(1), forces[:, 0] + forces[:, 4] + forces[:, 5]
        contact_goal = (allowed > F_MIN) & (wrong == 0.0)
        self.closing += closed.clamp(min=0)
        self.valid_closing += contact_goal * closed.clamp(min=0)
        link_before, self.link_distance = self.link_distance, self._link_distance()
        weight = W_PROGRESS * DOOR_RANGE / self.range
        reward = (weight * (contact_goal * closed.clamp(min=0) - (-closed).clamp(min=0))
                  + W_REACH * (torch.exp(-self.link_distance / REACH_SCALE) - torch.exp(-link_before / REACH_SCALE))
                  - W_WRONG * (wrong > 0)
                  - W_FORCE * ((allowed + wrong - F_MAX).clamp(min=0) / F_MAX).clamp(max=1.0)
                  - W_ENERGY * (self.qfrc_actuator[:, :7] * self.qvel[:, :7]).abs().sum(1))
        success = distance < self.tol
        earned = (self.valid_closing / (self.start_distance - distance).clamp(min=1e-9)).clamp(max=1.0)
        spill = ~success & self._spilled()
        reward = reward + success * SUCCESS_BONUS * earned - spill * SPILL_PENALTY
        # A world whose physics diverged (non-finite state; seen after long runs with "solver iterations limit" warnings)
        # would poison the trainer's observation normalisation and then the policy, so it ends now with zero reward and
        # restarts from the pool. Counted in info["nonfinite"].
        nonfinite = ~(torch.isfinite(self.qpos).all(1) & torch.isfinite(self.qvel).all(1) & torch.isfinite(reward))
        reward = torch.where(nonfinite, torch.zeros_like(reward), reward)
        success, spill = success & ~nonfinite, spill & ~nonfinite
        self.last_action = full_action
        self.steps += 1
        terminated = success | spill | nonfinite
        truncated = ~terminated & (self.steps >= EPISODE_STEPS)
        valid_fraction = torch.where(self.closing > 1e-6, self.valid_closing / self.closing.clamp(min=1e-9), torch.zeros_like(self.closing))
        info = {"success": success, "spill": spill, "timeout": truncated, "nonfinite": nonfinite, "valid_fraction": valid_fraction,
                "distance_norm": distance / self.range, "forces": forces}
        done = terminated | truncated
        obs = self._obs()  # before any reset: same (start-of-last-substep) kinematics as C's observation
        if done.any():
            self.reset_worlds(done)  # recomputes kinematics in every world, so only the reset rows are replaced
            obs = torch.where(done[:, None], self._obs(), obs)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)  # belt and braces: a reset world is finite again
        return obs, reward, terminated, truncated, info


class MixedEnv:
    """Several WarpFamilyEnv (one per task, one model each) stepped as one batch: worlds [0, n_1) are the first env's,
    and so on. info["task"] gives each world's env index."""

    def __init__(self, envs):
        self.envs, self.sizes = envs, [e.n for e in envs]
        self.n = sum(self.sizes)
        self.task_index = torch.cat([torch.full((e.n,), k, device=e.device) for k, e in enumerate(envs)])

    def reset(self):
        return torch.cat([e.reset() for e in self.envs])

    def step(self, action):
        outs = [e.step(a) for e, a in zip(self.envs, action.split(self.sizes))]
        obs, reward, terminated, truncated = (torch.cat([o[i] for o in outs]) for i in range(4))
        info = {k: torch.cat([o[4][k] for o in outs]) for k in outs[0][4]}
        info["task"] = self.task_index
        return obs, reward, terminated, truncated, info
