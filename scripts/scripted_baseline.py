"""No-RL baseline for the G batch: a scripted push given exactly what G2 is told, on G2's eval starts.

Per step it reads the contact target and push direction (TaskEnv.contact_target, the source of G2's
contact features), drives one forearm point by damped least squares on joints 1-4 to OFFSET behind the
target, then pushes along the push direction. The wrist leveler holds the plate, as in training. One
attempt per start, one forearm point (link5_c2, the point G2's observation is measured from). Unlike
task_feasibility.py there is no best-of-four-points retry. The push phase starts when the point is
within REACHED of its goal or the forearm group touches the object (on hardware: external joint torque).

`cap` scales the action so no joint moves faster than that fraction of full rate. Uncapped, the DLS step
saturates at full rate and throws the tray object within ~8 steps. Tune `cap` on training tasks (F, T4x)
only, then freeze it for T2.

`mode` "staged" (2026-11-27, after Josiah's read of the T2 failure reel) changes only the approach. It
first drives to a standoff STANDOFF in front of the face centre (level with it and aligned with the push
normal), then straight in along the normal to OFFSET, then pushes as "direct" does. Until the push starts,
the driven arm point slides: each step it is whichever of FOREARM_POINTS (elbow end to wrist end) is
nearest the current goal; it is locked when the push starts. STANDOFF was fixed before any run, not tuned.

Env, 250-step limit, yaw spread and reset seeds (50_000 + seed, then the env's own chain) match
eval_policy.py, so seed s here sees the same starts as the policy of seed s. With [outcomes.npy], the
per-episode rows (success, valid fraction, spill, timeout, final distance, reached push) are saved in
episode order, to join with runs/diagnostics/t2_failures/start_heights.csv. A 7th column says whether the
staged approach reached its standoff (always 1 in direct mode).

Hinged movers (T0's door): TaskEnv.contact_target only handles slide joints, so the script pushes
HINGE_FRACTION of the way from the hinge along the door, on its arm-side face, in the direction that point
moves as the door closes (recomputed every step, so the push turns with the door).

`slew` > 0 limits how much the action may change per control step (every joint, from 0 at episode start). Without
it the first action jumps from rest to the cap and throws a settled cylinder (2.8% of staged T2 episodes on env v2).

With [trace.npy], every control step is recorded per episode (stage_audit.py reads it): the controller's stage
(standoff reached, pushing), its distance to the current goal, arm contact force on the mover by group and on anything
else, the task's normalised distance to its goal, the smallest joint-limit margin (joints 2, 4, 6), and the largest
target lead |q_target - q| over joints 1-4.

Mode "staged2" (log.md 2026-11-27, Fixes 1 + 2 from the stage audit) is "staged" plus: (1) if the forearm is more
than `loss` from the mover while pushing, the script falls back to the standoff, at most MAX_REACQUIRE times per
episode (with "steer", the push goal also steers sideways toward the pre-contact point; off by default, it hurt every
training task); (2) Stage 1 also hands off within HANDOFF_NEAR of the standoff once it closes by less than `stall` per
control step. Outcome rows gain an 8th column, the number of re-acquires.

    python scripts/scripted_baseline.py <task> <seed> [episodes] [cap] [direct|staged|staged2] [outcomes.npy] [slew]
        [trace.npy] [loss] [stall] [steer | steer=<gain>]

Ablations: loss 99 never falls back, stall -1 never hands off early, steer (or steer=<gain>) adds the sideways steering.
"""
import sys

import gymnasium as gym
import mujoco
import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import task_feasibility as tf  # noqa: E402
from wam.tasks import TaskEnv  # noqa: E402

task, seed = sys.argv[1], int(sys.argv[2])
episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 200
cap = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
mode = sys.argv[5] if len(sys.argv) > 5 else "direct"
point = "link5_c2"
OFFSET = 0.04 + 0.01  # capsule radius + margin; the contact target is already on the surface
slew = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
STANDOFF, AT_STANDOFF = 0.15, 0.03  # m in front of the face; how close counts as there
HINGE_FRACTION = 2 / 3

u = TaskEnv(task=task, cabinet_spread=(0.1, np.radians(20.0)), conditioning="contact")
env = gym.wrappers.TimeLimit(u, 250)
geom = u.model.geom(point).id
hinge = u.model.jnt_type[u.joint_ids[0]] == mujoco.mjtJoint.mjJNT_HINGE
if hinge:
    door = u.mover_geoms[0]
    half = u.model.geom_size[door]
    local = u.model.geom_pos[door] + np.array([(2 * HINGE_FRACTION - 1) * half[0], 0.0, 0.0]) + half[1] * np.array(u.outward_local)


def target_and_direction():
    """Point to push and the direction it moves toward the goal."""
    if not hinge:
        return u.contact_target()
    rot = u.data.xmat[u.mover_body].reshape(3, 3)
    point_world = u.data.xpos[u.mover_body] + rot @ local
    joint = u.joint_ids[0]
    moving = np.cross(u.data.xaxis[joint], point_world - u.data.xanchor[joint]) * np.sign(u.goal[0] - u.data.qpos[u.qpos_adr[0]])
    return point_world, moving / max(np.linalg.norm(moving), 1e-9)

trace_path = sys.argv[8] if len(sys.argv) > 8 and sys.argv[8] != "-" else None
# Frozen 2026-11-27 on F/T0/T1 only (log.md): loss 0.03, stall 0.0005, no steering. Steering toward the target point
# cost 5-15 points on every training task (it drags a forearm that touched elsewhere along the face), so it is opt-in.
loss = float(sys.argv[9]) if len(sys.argv) > 9 else 0.03
stall = float(sys.argv[10]) if len(sys.argv) > 10 else 0.0005
steer = float(sys.argv[11].split("=")[1]) if len(sys.argv) > 11 and sys.argv[11].startswith("steer=") else \
    1.0 if len(sys.argv) > 11 and sys.argv[11] == "steer" else 0.0  # gain on the sideways correction
HANDOFF_NEAR, MAX_REACQUIRE = 0.06, 3
staged = mode in ("staged", "staged2")
LIMITED = [1, 3, 5]  # joints 2, 4, 6: the only ones with stops inside +-2 pi
traces, steps = [], []
rows = []
env.reset(seed=50_000 + seed)
pushing, standoff_done = False, mode == "direct"
last = np.zeros(env.action_space.shape, np.float32)
previous_error, reacquires = np.inf, 0
while len(rows) < episodes:
    target, direction = target_and_direction()
    if pushing:
        goal = u.data.geom_xpos[geom] + tf.PUSH_STEP / tf.GAIN * direction
        if mode == "staged2" and steer:
            offset = target - OFFSET * direction - u.data.geom_xpos[geom]
            goal += steer * (offset - (offset @ direction) * direction)
    else:
        goal = target - (OFFSET if standoff_done else STANDOFF) * direction
        if staged:
            point = min(tf.FOREARM_POINTS, key=lambda g: np.linalg.norm(u.data.geom_xpos[u.model.geom(g).id] - goal))
            geom = u.model.geom(point).id
    action, error = tf.drive(u, goal, point)
    action *= min(1.0, cap / max(np.abs(action).max(), 1e-9))
    if slew:
        action = (last + np.clip(action - last, -slew, slew)).astype(np.float32)
    last = action
    _, _, terminated, truncated, info = env.step(action)
    forearm = info["force/door_link4"] + info["force/door_link5"] + info["force/door_link6"] if staged else info["force/door_link5"]
    if not standoff_done:
        stalled = mode == "staged2" and error < HANDOFF_NEAR and previous_error - error < stall
        standoff_done = error < AT_STANDOFF or stalled
        previous_error = error
    else:
        pushing |= error < tf.REACHED or forearm > 0
        if mode == "staged2" and pushing and reacquires < MAX_REACQUIRE and u._link_distance() > loss:
            pushing, standoff_done, previous_error = False, False, np.inf
            reacquires += 1
    if trace_path:
        q = u.data.qpos[:7]
        margin = min(min(q[j] - u.ctrl_range[j, 0], u.ctrl_range[j, 1] - q[j]) for j in LIMITED)
        steps.append((standoff_done, pushing, error, info["force/scene"], info["force/door_upper"], info["force/door_link4"],
                      info["force/door_link5"], info["force/door_link6"], info["force/door_ee"], info["task/distance_norm"],
                      margin, np.abs(u.q_target[:4] - q[:4]).max()))
    if terminated or truncated:
        if trace_path:
            traces.append(np.array(steps, np.float32))
            steps = []
        rows.append((info["task/success"], info["task/valid_fraction"], info["task/spill"],
                     float(truncated and not terminated), info["task/distance_norm"], float(pushing), float(standoff_done), reacquires))
        env.reset()
        pushing, standoff_done = False, mode == "direct"
        last = np.zeros(env.action_space.shape, np.float32)
        previous_error, reacquires = np.inf, 0
        if staged:
            point = "link5_c2"
            geom = u.model.geom(point).id
v = np.array(rows)
if len(sys.argv) > 6:
    np.save(sys.argv[6], v)
if trace_path:
    np.save(trace_path, np.array(traces, dtype=object), allow_pickle=True)
won = v[:, 0] > 0
print(f"scripted {mode} (cap {cap}{f', slew {slew}' if slew else ''}{f', loss {loss}, stall {stall}' if mode == 'staged2' else ''}) seed {seed} on {task} ({episodes} episodes): "
      f"success {won.mean():.3f} | forearm-only {np.mean(won & (v[:, 1] > 0.9)):.3f} "
      f"| valid fraction of successes {v[won, 1].mean() if won.any() else float('nan'):.3f} "
      f"| spill {v[:, 2].mean():.3f} | timeout {v[:, 3].mean():.3f} "
      f"| reached push phase {v[:, 5].mean():.3f} | final distance (normalised) median {np.median(v[:, 4]):.2f}"
      + (f" | reached standoff {v[:, 6].mean():.3f} | success from standoff {won[v[:, 6] > 0].mean():.3f}"
         f" | success without {won[v[:, 6] == 0].mean() if (v[:, 6] == 0).any() else float('nan'):.3f}" if staged else "")
      + (f" | re-acquires per episode {v[:, 7].mean():.2f}" if mode == "staged2" else ""),
      flush=True)
