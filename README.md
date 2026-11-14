# wam-rl-diagnostic

A MuJoCo environment and PPO training script for one question: what breaks when model-free RL
is asked to do whole-arm contact manipulation?

The task is CloseCabinet from WACL (Huang, Seita, Bıyık, CoRL 2026, Sec. 4.1): an xArm 7 pushes
a cabinet door closed with links 4 to 6 while its end-effector keeps a plate level.

**The reward is not implemented.** `step()` returns 0.0. The specification is in
[REWARD_DESIGN.md](REWARD_DESIGN.md). Training runs today exercise the pipeline and learn nothing.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is missing
uv sync
```

Python 3.11 or 3.12. Every dependency is pinned in `pyproject.toml` and `uv.lock`.
On Linux the package sets `MUJOCO_GL=egl` for headless rendering. If the node has no EGL,
run with `MUJOCO_GL=osmesa`.

## Verify

```bash
uv run pytest                       # 7 tests, about 3 s
uv run python scripts/verify_env.py # same checks, throughput, artifacts/verify_scene.mp4
```

The checks:

- 10,000 bang-bang actions (every joint at full rate, held 1 to 25 steps): no non-finite
  observation or reward, no MuJoCo solver warning, penetration under 2 mm.
- An open-loop forearm sweep (constant base-yaw rate, nothing else moves) closes the door. Contact
  force appears on link 5 only: zero on the plate, flange, links 1 to 4, link 6, carcass and floor.
  The forearm is in contact on more than 90% of the steps while the door moves.
- Reset starts out of contact. Collision capsules cover the visual meshes of links 4 to 6.

Measured 2026-11-13, Apple M-series laptop, 10 cores, under memory pressure:

| Check | Result |
|---|---|
| Random actions, 3 seeds | max penetration 1.3 to 1.5 mm, 0 warnings, 0 non-finite |
| Forearm sweep | door 1.29 rad to 0.00 rad, link 5 in contact 100% of pushing steps, 10 to 55 N while pushing, 108 N peak against the stop, plate tilt at most 2.4° |
| Env throughput | 5,000 to 5,700 steps/s on one core |
| PPO smoke run, 4 envs | 3,400 steps/s including updates |

At these rates a 2M-step run is a few minutes on 10 or more cores.

## Train

```bash
uv run python scripts/train_ppo.py --num-envs 16 --num-steps 512 --total-timesteps 2000000 --seed 1
uv run tensorboard --logdir runs
```

Add `--track --wandb-project-name wam-rl-diagnostic` for Weights & Biases. Per episode the
script logs every `task/`, `force/` and `reward/` key of the final step's `info`, so
`task/success` and the force on each link group are in TensorBoard.

## Task

| | |
|---|---|
| Robot | xArm 7, no gripper, from `mujoco_menagerie` at `370c439`. Menagerie position actuators (kp 1500/1000/800, torque limits 50/30/20 N m). |
| Primary task | A plate held by its rim on link 7. Level means the link 7 y-axis within 25° of vertical, WACL's AxisCone. |
| Contact set | Links 4 to 6, as in WACL. Link 5 is the forearm tube. |
| Door | 0.22 m wide, 2 kg, hinge damping 2 N m s/rad, friction loss 1 N m, opens across the front of the robot. Starts at 1.0 to 1.4 rad. Success below 0.05 rad. |
| Timing | 1 kHz physics, 50 Hz control (`frame_skip` 20), 250-step episodes (5 s). |
| Action | 7 joint position-target deltas, 0.04 rad per step at full scale (2 rad/s). The target may lead the measured joint by at most 0.1 rad. |
| Observation (30) | joint position 7, joint velocity 7, target minus position 7, door angle and velocity 2, previous action 7. |
| Contact model | Elliptic cone, `impratio` 10, `condim` 3, friction 0.5, `solref` 0.002 1, `solimp` 0.95 0.99 0.001 0.5 2. |

Why these choices:

- **Position deltas, not torque.** Torque control needs the policy to learn gravity compensation
  before it can hold a plate. The PD loop gives compliance for free: contact force is bounded by
  the lead clamp times kp, then by the actuator torque limit.
- **The target is in the observation.** It is integrator state. Without it two identical joint
  states can produce different torques.
- **1 kHz.** At 500 Hz the stiffest legal contact (`solref` time constant 2 × dt) let the plate
  sink 3 to 4 mm into the floor on a full-speed hit. At 1 kHz the worst case is 1.5 mm.
- **Force is averaged over substeps.** Read at the last substep only, a push on an earlier,
  lighter door (damping 0.5) showed contact on 5% of control steps; the constraint switches on
  and off at kHz under light load.
- **Isotropic friction.** MuJoCo only takes directional friction on explicit `<contact><pair>`
  entries, and the two tangent axes follow the contact frame, not the forearm. It would not
  model sliding along the link.
- **Is it Markov?** Contact forces in MuJoCo are a function of positions, velocities and
  controls, so the observation above determines them on a fixed scene. What it leaves out: the
  cabinet pose (fixed in v0, must be added once randomised) and the time left in the episode.

## Layout

```
assets/close_cabinet.xml     scene: cabinet, door, option, cameras, home keyframe
assets/xarm7/xarm7_wam.xml   robot; header lists every difference from menagerie
wam/close_cabinet.py         the Gymnasium env
wam/checks.py                random-action stress, forearm sweep, capsule coverage
scripts/verify_env.py        runs the checks, benchmarks, writes the video
scripts/train_ppo.py         CleanRL PPO at fe8d8a0; header lists every change
tests/test_physics.py        the same checks as assertions
REWARD_DESIGN.md             reward specification and exploit audit
```

## Training stack

CleanRL `ppo_continuous_action.py` on CPU MuJoCo with one process per env. Upstream publishes
benchmark curves for this script on the Gymnasium MuJoCo tasks; the PPO update here is unmodified.

Fallback if a diagnostic needs more than about 100M steps: MuJoCo Playground 1.0
(`mujoco_playground.rl.rsl_rl.train`, MuJoCo Warp on an NVIDIA GPU). It was released 2026-11-06,
is marked alpha, and needs the task rewritten in its `reaf` layers, so it is not the starting point.

## Known limits

- The arm does not collide with itself.
- Fixed cabinet pose and fixed friction. Nothing is randomised except the start state.
- Capsules approximate links 4 to 6 to within 5 mm for about 90% of mesh vertices. Links 1 to 3
  are one capsule each and are rough.
- CleanRL treats the 250-step time limit as termination and does not bootstrap from it.
- CleanRL does not save observation-normalisation statistics, so `--save-model` output cannot
  be replayed without them.

## Provenance

- Robot model and meshes: `google-deepmind/mujoco_menagerie`, `ufactory_xarm7/`, commit
  `370c43965323c0944b433b202aac5058b5f907a9`. UFACTORY's BSD 3-clause licence, `assets/xarm7/LICENSE`.
- PPO: `vwxyzjn/cleanrl`, `cleanrl/ppo_continuous_action.py`, commit
  `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb`. MIT, `scripts/CLEANRL_LICENSE`.
