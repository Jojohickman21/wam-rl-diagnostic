# Results

Hardware for every run: Apple M-series laptop, 10 cores, two runs at a time, CleanRL PPO with 8
async envs, 256 steps per rollout, 2M steps per run. Three seeds per condition. Numbers are means
over each run's final 500 episodes unless stated. Raw event files are in `runs/` (not committed);
`python scripts/summarize_runs.py --last 500` reproduces the table from them.

## 1. Fixed start: gated vs ungated reward (2026-11-16)

Command: `sh scripts/run_gated_vs_ungated.sh WamCloseCabinet-v0 WamCloseCabinetUngated-v0`

| Condition | Seed | Success | Valid fraction | Spill | Episode length | Steps to 90% success | Episodes with any end-effector force |
|---|---|---|---|---|---|---|---|
| Gated | 1 | 1.000 | 0.9995 | 0 | 36.6 | 31.8k | 0 |
| Gated | 2 | 1.000 | 0.9986 | 0 | 38.1 | 28.8k | 0 |
| Gated | 3 | 1.000 | 0.9971 | 0 | 41.2 | 32.4k | 0 |
| Ungated (A0) | 1 | 1.000 | 0.9990 | 0 | 36.3 | 35.6k | 0 |
| Ungated (A0) | 2 | 1.000 | 0.9989 | 0 | 40.1 | 27.6k | 0 |
| Ungated (A0) | 3 | 1.000 | 0.9989 | 0 | 36.8 | 45.5k | 0 |

Valid fraction is the share of all door closing in the episode that happened with links 4 to 6
pushing and nothing else touching. It is computed the same way in both conditions.

**Finding.** The gate made no measurable difference. The ungated policy was free to close the door
with the plate or flange and never did: zero episodes with end-effector force on the door in the
final 500 of every seed. Both conditions reach 90% success in about 30k steps (12 to 15 s of robot
time per episode, 1.2 to 1.8 minutes of wall clock).

**Reading.** The home posture already places the forearm on the door's path with the plate
trailing outside it, so the shortest path to reward is forearm contact. On this start the
whole-arm constraint is not a constraint. The result says RL solves a pre-posed single-contact
push quickly; it says nothing about whether RL finds whole-arm contact when the end-effector is
the easier tool.

**Dips in `task/valid_fraction`.** 1.6% of gated episodes and 3.8% of ungated episodes have a
valid fraction below 1. Almost all are 0.96 to 0.98 and still succeed; 0.4% of them involve any
wrong contact. The mechanism is one or two control steps of door motion with the substep-mean
forearm force below 1 N, consistent with the door separating from the link for a step after a
hard push (its velocity time constant is about one control step). The ungated run has 2.4x more
of them because nothing pays it to keep the link on the door. Roughly 100 episodes per run have
valid fraction 0 and fail; their count is identical across conditions and they are consistent
with early training, which was not checked by step index.

## 2. Randomised start: gated vs ungated (2026-11-16)

Command: `sh scripts/run_gated_vs_ungated.sh WamCloseCabinetRandomStart-v0 WamCloseCabinetRandomStartUngated-v0`

`WamCloseCabinetRandomStart-v0` draws every joint within 0.5 rad of home, levels the plate with
the wrist, and rejects starts that touch anything or sit between the door and the cabinet. The
forearm starts 1 to 40 cm from the door (13 to 21 cm on the fixed start). A scripted
drive-to-home-then-sweep closes the door from 95 of 100 such starts, so the starts are feasible.

| Condition | Seed | Success | Valid fraction | Spill | Episode length | Steps to 90% success | Episodes with any end-effector force |
|---|---|---|---|---|---|---|---|
| Gated | 1 | 0.976 | 0.998 | 0.024 | 44 | 67k | 0 |
| Gated | 2 | 0.976 | 0.993 | 0.024 | 45 | 83k | 0 |
| Gated | 3 | 0.962 | 0.996 | 0.042 | 44 | 66k | 0 |
| Ungated (A0) | 1 | 0.980 | 0.997 | 0.020 | 45 | 83k | 0.008 |
| Ungated (A0) | 2 | 0.968 | 0.999 | 0.032 | 44 | 62k | 0 |
| Ungated (A0) | 3 | 0.974 | 0.994 | 0.026 | 44 | 64k | 0 |

**Finding.** Still no difference between conditions. Success drops from 100% to 96 to 98% and
every failure is a plate spill, which ends the episode. Learning takes about twice as long
(62k to 83k steps to 90%). The ungated policy used the end-effector on the door in 0.8% of
episodes for one seed and never otherwise.

**Reading.** Widening the start by 0.5 rad per joint did not make the plate a competitive tool:
the arm's geometry still puts the forearm on the door's path from almost every start, and the
scripted sweep already solved 95% of these starts. The spills are the new failure mode and they
match the scripted baseline's 5% spill rate, so they come from the transit, not from the push.
The whole-arm constraint has not bound in either experiment. To make it bind, the end-effector
has to be the shorter path: randomise the cabinet pose, or start with the plate between the arm
and the door.

## 3. Randomised cabinet pose: gated vs ungated (2026-11-16)

Command: `sh scripts/run_gated_vs_ungated.sh WamCloseCabinetRandomPose-v0 WamCloseCabinetRandomPoseUngated-v0`

`WamCloseCabinetRandomPose-v0` adds to the randomised start a cabinet shifted by up to 10 cm in
x and y and turned by up to 20 degrees about the hinge. The hinge position and yaw are appended
to the observation. The scripted base-yaw sweep now succeeds on 43 of 100 starts (10 spills, the
rest stop with the door about 0.3 rad open), so the one-joint solution of sections 1 and 2 is
gone. Whether every start is feasible for a full-arm policy was not established.

| Condition | Seed | Success | Valid fraction | Spill | Timeout | Episode length | Steps to 90% success | Wrong-contact steps per episode | Episodes with any end-effector force |
|---|---|---|---|---|---|---|---|---|---|
| Gated | 1 | 0.830 | 0.979 | 0.172 | 0.002 | 60 | 314k | 1.3 | 0.002 |
| Gated | 2 | 0.966 | 0.992 | 0.034 | 0 | 52 | 154k | 0.7 | 0 |
| Gated | 3 | 0.884 | 0.986 | 0.116 | 0 | 50 | 108k | 1.5 | 0.006 |
| Ungated (A0) | 1 | 0.888 | 0.973 | 0.044 | 0.070 | 74 | 342k | 2.6 | 0 |
| Ungated (A0) | 2 | 0.894 | 0.985 | 0.106 | 0 | 54 | 153k | 0.9 | 0 |
| Ungated (A0) | 3 | 0.952 | 0.991 | 0.048 | 0.002 | 53 | 108k | 0.8 | 0 |

**Finding.** Still no gated-vs-ungated gap: seed-to-seed spread (83 to 97% success within the
gated condition) is larger than any difference between conditions. The task got harder in the
ways expected. Success is 83 to 97%, learning takes 4 to 10x longer (108k to 342k steps to 90%),
episodes are 50 to 74 steps instead of 44, and valid fraction drops to 0.97 to 0.99 with 1 to 3
wrong-contact steps per episode. The end-effector still almost never touches the door: at most
0.6% of episodes. Wrong contact is links on the carcass or floor, not the plate on the door.

**Failure modes.** Spills dominate (3 to 17% per run) and now happen mid-push: the median door
angle at failure is 0.4 to 1.0 rad, not the start angle, so the plate tips while the arm adapts
its push to the cabinet's pose. One ungated seed times out on 7% of episodes with the door left
0.66 rad open.

**Reading across sections 1 to 3.** Over 18 runs, PPO with perfect state found forearm contact
on its own every time, and the gate has never changed what the policy learned. The reason is
geometric: on this arm, from these starts, the forearm reaches the door before the plate does,
so the constraint is never the binding one. What did bind as the task widened is the primary
task (plate level), which went from zero cost to the dominant failure. Two conclusions for the
RL route: (1) whole-arm contact is not hard for RL when the scene makes it the shortest path,
and a diagnostic that wants to measure the constraint has to put the end-effector on the
shorter path deliberately; (2) the dual-task coupling is where the difficulty appears first, and
that is the part WACL handles by construction and RL has to learn.

## 4. Physical plate constraint, A7 (2026-11-17)

Command: `sh scripts/run_gated_vs_ungated.sh WamCloseCabinetRandomPoseTray-v0`

`WamCloseCabinetRandomPoseTray-v0` is section 3's env (random start, random cabinet pose, contact
gate on) with the plate constraint made physical. A free cylinder (r 3 cm, h 4 cm, 0.2 kg) stands
on the plate and the episode spills when it leaves. There is no tilt penalty, level gate or tilt
limit. The cylinder's friction is drawn per episode so it slides at 15 to 35 degrees on a still
plate (`scripts/calibrate_tray.py`: holds at 95%, falls at 105%), and the policy observes that
angle and the cylinder's position and velocity on the plate. Starts with the plate below 15 cm are
rejected. Runs went one at a time because of memory pressure, at 2,150 to 2,320 steps/s.

| Seed | Success | Valid fraction | Spill | Timeout | Episode length | Steps to 90% success | Wrong-contact steps per episode | Episodes with any end-effector force |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.944 | 0.993 | 0.056 | 0.002 | 51 | 194k | 0.6 | 0.002 |
| 2 | 0.846 | 0.964 | 0.152 | 0.002 | 61 | 213k | 2.6 | 0 |
| 3 | 0.860 | 0.984 | 0.126 | 0.014 | 65 | 231k | 4.6 | 0.002 |

Spill by the episode's slide angle, last 2,000 episodes of each seed pooled (about 1,500 per bin):

| Slide angle | 15–20° | 20–25° | 25–30° | 30–35° |
|---|---|---|---|---|
| Success | 0.833 | 0.867 | 0.894 | 0.905 |
| Spill | 0.167 | 0.135 | 0.101 | 0.091 |
| Final plate tilt in successful episodes, median | 13.0° | 13.0° | 12.9° | 13.0° |
| Final plate tilt in successful episodes, 90th percentile | 21.2° | 21.8° | 22.7° | 24.7° |

**Finding.** Making the constraint physical did not improve the coupling. Success (85 to 94%)
and spill (6 to 15%) sit inside section 3's range (83 to 97%, 3 to 17%), and learning took about
as long (194k to 231k steps to 90%, against 108k to 314k).

**The policy did not use the observed limit.** Spill falls steadily as the slide angle rises,
from 17% to 9%, and the median plate tilt at the end of a successful episode is 13.0 degrees in
every bin. A policy that read the slide angle would tilt less when the object is slippery and
spill at a flat rate. This one learned a single posture that is safe for a grippy object and
marginal for a slippery one, then let friction absorb the difference.

**Without a tilt penalty the policy tilts.** Median final tilt is 13 degrees, against a level
plate in the penalised runs. That is the expected consequence of rewarding only the object's
survival: tilt is free until it costs the object.

**Reading.** The physical constraint removed the metric to overfit, as intended, but PPO with
2M steps learned to treat the limit as a fixed average rather than an input. Two things would
tell whether that is the learner or the signal: a longer run (if the bins flatten with more
data, it was sample-limited), and a run where the slide angle varies more widely or is the only
thing that changes between episodes. The second gap is that the spill signal arrives late: the
object has to slide 6 cm before the episode ends, so a bad tilt is punished tens of steps after
it happens.

## 5. A18: door reward coupled to required friction, energy penalty (2026-11-17)

Command: `sh scripts/run_gated_vs_ungated.sh WamCloseCabinetTrayFrictionCoupled-v0`

Section 4's tray env with three changes made together: the positive door reward is multiplied by
exp(−10 · [μ_req − tan 12°]₊), where μ_req is the friction a resting object would need given the
plate's tilt and its acceleration over one control step, from arm kinematics only; the slide angle
and the object's position and velocity are removed from the observation; and the smoothness terms
are replaced by the energy penalty −0.004 · Σ|τⱼ q̇ⱼ|. Before training, μ_req matched tan(tilt) on a
still plate and an object at rest started sliding on 0 to 8% of steps with μ_req below its true
friction and 57 to 98% above it. Runs went one at a time, about 2,000 steps/s.

| Seed | Success | Valid fraction | Spill | Timeout | Episode length | Steps to 50% / 90% success | Mean coupling factor |
|---|---|---|---|---|---|---|---|
| 1 | 0.752 | 0.972 | 0.160 | 0.088 | 96 | 229k / never | 0.45 |
| 2 | 0.650 | 0.966 | 0.298 | 0.060 | 111 | 411k / never | 0.35 |
| 3 | 0.864 | 0.980 | 0.118 | 0.018 | 75 | 138k / 808k | 0.54 |

By slide angle, last 2,000 episodes of each seed pooled (about 1,500 per bin):

| Slide angle | 15–20° | 20–25° | 25–30° | 30–35° |
|---|---|---|---|---|
| Success | 0.676 | 0.747 | 0.773 | 0.814 |
| Spill | 0.286 | 0.190 | 0.147 | 0.106 |
| Timeout | 0.040 | 0.063 | 0.081 | 0.080 |
| Final plate tilt in successful episodes, median | 8.5° | 9.0° | 9.1° | 9.4° |

Success by training window (seed 1 / 2 / 3): 0.40 / 0.19 / 0.55 over 0 to 500k steps, 0.74 / 0.64 /
0.88 over 1.5M to 2M. All three seeds had flattened by the last two windows.

**Finding.** A18 is worse than section 4 on every task measure: success 65 to 86% against 85 to 94%,
spill 12 to 30% against 6 to 15%, episodes 75 to 111 steps against 51 to 65, and learning 2 to 6x
slower. It did change the posture: median tilt at success dropped from 13° to about 9°. But spill
still falls with friction, from 29% to 11%, which is steeper than section 4's 17% to 9%.

**Reading.** Three changes were made at once, so the loss cannot be attributed to one of them. The
most likely contributor is the observation: section 4's policy could see the cylinder start to
slide and react; this one is blind to it and must prevent slip open loop, as Heins & Schoellig's
controller does, but without their model. The mean coupling factor of 0.35 to 0.54 says the policy
spends much of each episode in the region the reward discounts, so the door signal it learns from
is roughly halved and noisier, which fits the slower learning. The required-friction signal itself
is sound (it predicted slip before training); what this run shows is that using it as a multiplier,
without letting the policy observe the object or its own required friction, did not teach the
policy to keep margin. Separating the three changes needs three runs: A18 with the object observed,
A18 with μ_req in the observation (encoder-derived, so still hardware-measurable), and section 4
with only the energy penalty.

## 6. Separating A18's three changes (2026-11-18)

Runs: one at a time, 16 to 21 minutes each, seed-major, each
`python scripts/train_ppo.py --env-id <env> --seed <n> --num-envs 8 --num-steps 256 --total-timesteps 2000000 --save-model`.
All nine finished with exit 0 and a full log (976 SPS lines); nothing was rerun.

Tables: `python scripts/analyze_runs.py WamCloseCabinetTrayCoupledSeeRisk-v0 WamCloseCabinetTrayCoupledSeeObject-v0 WamCloseCabinetTrayEnergy-v0 WamCloseCabinetTrayFrictionCoupled-v0 WamCloseCabinetRandomPoseTray-v0`

Each env differs from an earlier one by one change:

| Env | Base | The one change |
|---|---|---|
| `WamCloseCabinetTrayCoupledSeeRisk-v0` | A18 | The policy observes its previous step's μ_req (encoder-derived) |
| `WamCloseCabinetTrayCoupledSeeObject-v0` | A18 | The policy observes the cylinder's position and velocity in the plate frame (not its slide angle) |
| `WamCloseCabinetTrayEnergy-v0` | A7 | The energy penalty replaces the smoothness terms; everything else is A7, including the observed slide angle and object |

"Recovers" means: mean success over the 3 seeds within 5 points of A7's 0.883 (so at least 0.833)
and mean spill at most 0.12.

**SeeRisk (A18 + μ_req observed)**

| Seed | Episodes | Success | Valid fraction | Spill | Timeout | Episode length | Steps to 50% / 90% | Mean coupling | Success 0-0.5M / 1.5-2M |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 25474 | 0.758 | 0.952 | 0.150 | 0.094 | 85 | 174k / never | 0.47 | 0.50 / 0.75 |
| 2 | 29501 | 0.892 | 0.979 | 0.076 | 0.034 | 71 | 128k / 399k | 0.47 | 0.66 / 0.89 |
| 3 | 31215 | 0.924 | 0.969 | 0.070 | 0.006 | 61 | 99k / 250k | 0.61 | 0.73 / 0.90 |

| Slide angle | 15–20° | 20–25° | 25–30° | 30–35° |
|---|---|---|---|---|
| Success | 0.771 | 0.860 | 0.900 | 0.884 |
| Spill | 0.206 | 0.113 | 0.072 | 0.070 |
| Timeout | 0.027 | 0.029 | 0.029 | 0.046 |
| Final plate tilt in successful episodes, median | 8.5° | 8.7° | 8.6° | 9.0° |
| Episodes | 1513 | 1484 | 1533 | 1470 |

**SeeObject (A18 + object position and velocity observed)**

| Seed | Episodes | Success | Valid fraction | Spill | Timeout | Episode length | Steps to 50% / 90% | Mean coupling | Success 0-0.5M / 1.5-2M |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 25006 | 0.900 | 0.987 | 0.048 | 0.052 | 75 | 149k / 841k | 0.50 | 0.59 / 0.88 |
| 2 | 27873 | 0.898 | 0.983 | 0.090 | 0.014 | 68 | 151k / 475k | 0.46 | 0.58 / 0.92 |
| 3 | 28490 | 0.914 | 0.987 | 0.074 | 0.012 | 70 | 68k / 120k | 0.55 | 0.80 / 0.92 |

| Slide angle | 15–20° | 20–25° | 25–30° | 30–35° |
|---|---|---|---|---|
| Success | 0.858 | 0.921 | 0.905 | 0.921 |
| Spill | 0.126 | 0.058 | 0.058 | 0.043 |
| Timeout | 0.016 | 0.021 | 0.037 | 0.037 |
| Final plate tilt in successful episodes, median | 7.0° | 6.9° | 6.7° | 7.0° |
| Episodes | 1480 | 1514 | 1509 | 1497 |

**Energy (A7 with the energy penalty only)**

| Seed | Episodes | Success | Valid fraction | Spill | Timeout | Episode length | Steps to 50% / 90% | Mean coupling | Success 0-0.5M / 1.5-2M |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 32749 | 0.926 | 0.986 | 0.066 | 0.008 | 60 | 59k / 183k | 1.00 | 0.80 / 0.92 |
| 2 | 31370 | 0.842 | 0.979 | 0.156 | 0.004 | 62 | 69k / 285k | 1.00 | 0.75 / 0.89 |
| 3 | 31030 | 0.822 | 0.980 | 0.176 | 0.006 | 62 | 59k / 356k | 1.00 | 0.76 / 0.84 |

| Slide angle | 15–20° | 20–25° | 25–30° | 30–35° |
|---|---|---|---|---|
| Success | 0.812 | 0.876 | 0.916 | 0.896 |
| Spill | 0.194 | 0.120 | 0.082 | 0.096 |
| Timeout | 0.003 | 0.008 | 0.005 | 0.009 |
| Final plate tilt in successful episodes, median | 16.8° | 17.2° | 19.0° | 20.2° |
| Episodes | 1544 | 1447 | 1540 | 1469 |

Coupling is off in this env, so the coupling column reads 1.00 by construction.

The same script reproduces sections 4 and 5 (A7 and A18) exactly, with steps to 50% added: A7
75k / 66k / 57k, A18 229k / 410k / 137k (it prints 410k and 807k for A18 where section 5 says 411k
and 808k; same runs, 1k apart, cause not checked).

Summary, 3-seed means:

| Env | Mean success | Mean spill | Recovers? | Seeds agree? | Spill 15–20° → 30–35° |
|---|---|---|---|---|---|
| A7 (reference) | 0.883 | 0.111 | | no (0.846 to 0.944) | 16.7% → 9.1% |
| A18 | 0.755 | 0.192 | no | no (0.650 to 0.864) | 28.6% → 10.6% |
| SeeRisk | 0.858 | 0.099 | yes, by the definition | no: seed 1 0.758, seeds 2 and 3 0.892 and 0.924 | 20.6% → 7.0% |
| SeeObject | 0.904 | 0.071 | yes | yes: 0.898 to 0.914, spill 0.048 to 0.090 | 12.6% → 4.3% |
| Energy | 0.863 | 0.133 | no: success passes, spill misses by 1.3 points | no: 0.822 to 0.926 | 19.4% → 9.6% |

**Finding.** The observation is what broke A18. Putting the object's position and velocity back
into A18's observation recovers it on every seed: success 0.898 to 0.914 against A18's 0.650 to
0.864, spill 4.8 to 9.0% against 11.8 to 29.8%, with the coupling and the energy penalty still on.
Its mean (0.904 success, 7.1% spill) is slightly better than A7's (0.883, 11.1%) without seeing the
slide angle, but A7's own seeds span 0.846 to 0.944, so "better than A7" is not established.
Observing μ_req instead also meets the recovery definition on the mean (0.858, 9.9%), but only
because two seeds recovered: seed 1 (0.758, 15% spill, never reached 90%) looks like A18's seed 1.
Call it a partial, seed-dependent recovery, not an established one. The energy penalty alone
leaves A7 essentially unchanged: success 0.863 against 0.883 and spill 13.3% against 11.1%, both
inside A7's seed spread. It formally fails the spill threshold by 1.3 points, but that gap is
smaller than the noise between seeds of either env, so the energy term is not what broke A18. It
did change the posture: median tilt at success rose from 13° (A7) to 17 to 20°, and unlike A7 it
rises with slide angle (16.8° → 20.2°), the first sign of any policy grading its tilt to the
object's friction. Spill still falls with slide angle in all three envs. SeeObject flattens the
curve above 20° (5.8%, 5.8%, 4.3%) but still spills twice as often in the slipperiest bin (12.6%).

**Reading.** A18's loss comes from making the policy blind to the object, not from the coupling
or the energy term. With the object observed, the coupled reward does no harm and the policy holds
a flatter plate (7° at success against A7's 13°). μ_req in the observation is a partial substitute
that worked on two of three seeds. That fits what μ_req is: an estimate of risk from arm
kinematics, which cannot see the object start to slide when the true friction is below the
design threshold. No condition has made spill independent of friction. Even with the object
visible, the slipperiest objects spill most; the likely reading (inferred, not measured) is that
the policy reacts to sliding once it starts rather than avoiding it in advance. For the RL-vs-pivot question this says: model-free RL couples the
two tasks only when it gets feedback on the object, and on hardware that feedback means a camera
or tactile estimate of the object's pose on the plate. That is a perception requirement, not a
reward-design one, and it is the same requirement an imitation-learning policy would have.
Whether a controller that owns the plate (A20) removes the need for it is the next question.

## 7. Follow-up suite: A20 wrist leveler, A21 risk observed without coupling (2026-11-18)

Runs: queue `/tmp/pg/section7.sh`, started 07:36, one at a time, seed-major, 15 to 17 minutes each
(A20 was not measurably slower in training despite the wrist IK), same command as section 6. All
six finished with exit 0 and full logs by 09:12; nothing was rerun.

Tables: `python scripts/analyze_runs.py WamCloseCabinetTrayWristLeveler-v0 WamCloseCabinetTraySeeRisk-v0`

| Label | Env | What it is |
|---|---|---|
| A20 | `WamCloseCabinetTrayWristLeveler-v0` | Tray env, object and slide angle unobserved, energy penalty, no coupling, contact gate on. A damped-least-squares wrist controller owns joints 5 to 7 and keeps the plate normal vertical; the policy's action is joints 1 to 4. |
| A21 | `WamCloseCabinetTraySeeRisk-v0` | Tray env, object and slide angle unobserved, energy penalty, no coupling, previous step's μ_req observed. SeeRisk (section 6) without the coupling. |

Both observe only hardware-measurable signals: joint encoders and targets, door angle, cabinet
pose, and for A21 the encoder-derived μ_req.

**A20 (wrist leveler, RL on joints 1 to 4)**

| Seed | Episodes | Success | Valid fraction | Spill | Timeout | Episode length | Steps to 50% / 90% | Mean coupling | Success 0-0.5M / 1.5-2M |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 24167 | 0.924 | 0.982 | 0.032 | 0.044 | 77 | 50k / 205k | 1.00 | 0.84 / 0.89 |
| 2 | 31045 | 0.926 | 0.967 | 0.040 | 0.034 | 65 | 41k / 106k | 1.00 | 0.91 / 0.93 |
| 3 | 32080 | 0.924 | 0.965 | 0.064 | 0.012 | 62 | 43k / 103k | 1.00 | 0.88 / 0.91 |

| Slide angle | 15–20° | 20–25° | 25–30° | 30–35° |
|---|---|---|---|---|
| Success | 0.865 | 0.923 | 0.940 | 0.941 |
| Spill | 0.101 | 0.045 | 0.030 | 0.025 |
| Timeout | 0.035 | 0.033 | 0.029 | 0.034 |
| Final plate tilt in successful episodes, median | 2.0° | 2.0° | 2.0° | 2.0° |
| Episodes | 1470 | 1503 | 1510 | 1517 |

**A21 (μ_req observed, no coupling)**

| Seed | Episodes | Success | Valid fraction | Spill | Timeout | Episode length | Steps to 50% / 90% | Mean coupling | Success 0-0.5M / 1.5-2M |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 33437 | 0.888 | 0.981 | 0.110 | 0.002 | 57 | 74k / 236k | 1.00 | 0.77 / 0.89 |
| 2 | 32796 | 0.838 | 0.974 | 0.146 | 0.016 | 63 | 115k / 1351k | 1.00 | 0.63 / 0.84 |
| 3 | 33505 | 0.904 | 0.987 | 0.092 | 0.006 | 57 | 72k / 343k | 1.00 | 0.74 / 0.90 |

| Slide angle | 15–20° | 20–25° | 25–30° | 30–35° |
|---|---|---|---|---|
| Success | 0.821 | 0.870 | 0.906 | 0.919 |
| Spill | 0.183 | 0.127 | 0.086 | 0.068 |
| Timeout | 0.002 | 0.005 | 0.008 | 0.014 |
| Final plate tilt in successful episodes, median | 13.0° | 12.7° | 12.9° | 13.1° |
| Episodes | 1477 | 1361 | 1563 | 1599 |

Coupling is off in both envs, so the coupling column reads 1.00 by construction.

Summary, 3-seed means, with the earlier tray envs for comparison:

| Env | Observes the object? | Mean success | Mean spill | Recovers? | Seeds agree? | Steps to 90% | Spill 15–20° → 30–35° | Median tilt at success |
|---|---|---|---|---|---|---|---|---|
| A7 | yes, plus slide angle | 0.883 | 0.111 | (reference) | no (0.846 to 0.944) | 194k to 231k | 16.7% → 9.1% | 13° |
| A18 | no | 0.755 | 0.192 | no | no | 807k or never | 28.6% → 10.6% | 9° |
| SeeObject | yes | 0.904 | 0.071 | yes | yes (0.898 to 0.914) | 120k to 841k | 12.6% → 4.3% | 7° |
| A20 | no | 0.925 | 0.045 | yes | yes (0.924 to 0.926, spill 3.2 to 6.4%) | 103k to 205k | 10.1% → 2.5% | 2° |
| A21 | no | 0.877 | 0.116 | yes, by the definition | partly: success 0.838 to 0.904; seed 2's spill (14.6%) is above 0.12 | 236k to 1,351k | 18.3% → 6.8% | 13° |

**Finding.** A20 is the best tray result so far and the only one that is strong on every seed
without observing the object. Success is 0.924 to 0.926 (spread 0.2 points, against 10 points for
A7), spill 3.2 to 6.4%, and it reaches 90% success in 103k to 205k steps, the tightest and fastest range of any tray env
(SeeObject had one seed at 120k, but the other two took 475k and 841k). Its success range does not overlap SeeObject's (0.898 to 0.914), but three seeds and a
2-point gap are not enough to call it better than SeeObject; it is at least as good while needing
no camera on the object. The controller holds the plate at 2.0° in every bin. Spill still falls
with slide angle, from 10.1% to 2.5%. With the plate held near level, those spills come from the
plate's accelerations, not its tilt. Timeouts are higher than in A7 or A21 (1.2 to 4.4% against
under 2%). The cause was not checked; one possibility is that some cabinet poses need a wrist
orientation the leveler does not allow.

A21 meets the recovery definition on the mean (0.877, 11.6% spill), and every seed clears the
success bar, but seed 2 spills 14.6% and took 1.35M steps to reach 90%, so it is borderline and
not cleanly established. It reproduces A7 almost exactly: the same 13° posture in every bin and
the same spill-versus-friction slope, with the slide angle and the object hidden and no coupling.
Taking the coupling out of SeeRisk did not hurt it (0.877 against 0.858) and made it more
consistent: its worst seed went from 0.758 to 0.838.

**Reading.** Three things follow from sections 6 and 7 together.

1. Learning the push and engineering the plate works better than learning both. Giving the plate
   to a 3-iteration IK leveler and leaving RL four joints gave the highest, most consistent and
   fastest result, using only encoder signals. This is Ray's "decide what component to learn"
   tested directly, and it is the strongest evidence in this repo for a structured RL route.
2. The coupling hurts when the policy is blind to the object and is neutral otherwise. A18
   (blind, coupled) is the only bad tray result; A21 (blind, uncoupled) is at A7's level. What
   μ_req in the observation adds on its own is untested: the blind, uncoupled env without μ_req
   was not run, so A21's result cannot be credited to μ_req.
3. No policy has made spill independent of friction, A20 included. The remaining spills are
   inertial and grow on slippery objects, and nothing the policy observes tells it the friction.
   Closing that gap needs either feedback on the object (SeeObject's camera route) or limiting
   the plate's acceleration in the controller. The second keeps everything on encoders: for
   example, the leveler could also cap μ_req, or A20 could observe μ_req.

Not established: A20's advantage over SeeObject (3 seeds, ranges 1 point apart), any effect of
μ_req observation by itself, and the cause of A20's timeouts. All results are final-500-episode
means of 2M-step runs in simulation; nothing here has touched hardware.

### 7.1 Why A20 times out more than A7 (2026-11-18)

A20 timed out on 196 of its last 6,000 training episodes (3.3%), against 26 for A7 (0.4%). Two
sources:

- **Training logs** (exact): each episode's final-step metrics, last 2,000 episodes per seed.
  `python scripts/diagnose_timeouts/profile.py WamCloseCabinetTrayWristLeveler-v0 WamCloseCabinetRandomPoseTray-v0`
  and, for rates by training window, `python scripts/diagnose_timeouts/trend.py <same ids>`.
- **Policy replay** (approximate): the saved A20 policies rerun stochastically for 150k steps per
  seed (5,694 episodes), recording joint, contact and door trajectories.
  `python scripts/diagnose_timeouts/replay.py WamCloseCabinetTrayWristLeveler-v0 <seed> runs/diagnostics/a20_s<seed>.npy 150000`,
  then `analyze.py`, `analyze2.py` and `perseed.py` in the same folder. **Caveat:** training did not
  save the observation-normalisation statistics, so the replay rebuilds them with a 30k-step
  warm-up. Replayed timeout rates are higher than in training (16.7 / 6.7 / 2.1% against 4.4 / 3.4 /
  1.2%), with the same seed order. The replay is used only for mechanism, never for rates, and its
  timeouts match the training-log profile below (level plate, the same split in final door
  angle, wrong contact at the end in 13 to 20%).

**A7's and A20's timeouts are different failures.** From the training logs:

| | A7 timeouts (26) | A20 timeouts (196) |
|---|---|---|
| Median plate tilt at the end | 21.6° | 1.6° |
| Wrong contact on the final step | 92% (arm on carcass or floor) | 20% |
| Wrong-contact steps per episode, median | 150 | 3.5 |
| Final door angle, 25th / 50th / 75th percentile | 0.27 / 0.78 / 1.16 rad | 0.17 / 0.34 / 1.19 rad |

A7's few timeouts are wedges: the arm is jammed against the cabinet with the plate tipped. A20's
are mostly clean. In the replay, over the last 50 steps of a timeout the forearm touches the door on
a median of 0% of steps, the door does not move, and the joints still drift by 0.1 to 0.2 rad each.
The arm is wandering near the door, not pushing. Timeouts split into stalls near closed (42%,
below 0.5 rad), episodes where the door was never closed past 1.0 rad (40%), and the rest.

**The elbow runs into its limits.** In the replay, 44% of A20 timeouts end with joint 4 within
0.05 rad of a hard limit: 28% fully folded (3.93 rad) and 16% fully straight (−0.19 rad). Of 4,948
successful episodes, none comes within 0.05 rad of either limit at any step. Where joint 4 hits the
folded limit, it does so late (median step 189) and usually with the door already near closed
(median 0.36 rad).

**Timeouts concentrate on hard starts** (replay rates, comparative only): 10 to 11% at cabinet
yaw beyond ±10° against 4% within it, 10% in the top quartile of hinge x or y against 6%, and 12.4%
when the elbow starts most folded (top quartile of joint 4's start angle) against 3.5% in the bottom
quartile.

**They are mostly feasible under the leveler.** Seed 3 solves the hardest combination (|yaw| > 10°
and joint 4 starting above 2.75 rad) 81% of the time with 4.3% timeouts. Seed 1 solves 56% with
37% timeouts. So the wrist constraint does not make these starts unreachable as a rule; seeds 1 and 2
learned a worse strategy for them. Only 13% of seed 3's timeouts touch a joint-4 limit, against 47 to
48% for seeds 1 and 2.

**Not fixed by more training.** Timeout rate by 0.5M-step training window, from the logs: seed 1
9.1 / 6.4 / 7.2 / 6.5%, seed 2 3.8 / 2.1 / 2.1 / 2.5%, seed 3 3.4 / 0.9 / 1.4 / 1.6%. All flat
after 0.5M. A7 stays at 0.1 to 1.1% in every window.

**Finding.** A20 times out more because some seeds converge on a strategy that, on hard starts
(large cabinet yaw, far hinge, folded starting elbow), drives the elbow into a joint limit or leaves
the forearm off the door, and nothing ends the episode. Reaching the limit is the clearest marker:
it appears in about half of the timeouts of the two weaker seeds and never in a success. The best
seed shows that most of these starts can be solved with the leveler, so this is mostly a learning
failure, not a feasibility wall.

**Reading (inference, not tested).** Two parts of the design plausibly explain why A7 avoids this:

1. **With the plate level, wandering doesn't end the episode.** In A7, an arm that strays tilts
   the plate and spills: A7's spill is 20% in the first 0.5M steps, and its timeouts are 0.4%. In
   A20 the plate stays level, so the same straying runs to 250 steps. Total failures (spill plus
   timeout) fit this. A20's seeds trade one for the other: seed 1 has 4.1% spill and 6.5%
   timeouts, seed 3 6.9% and 1.6%. Over the last 0.5M steps, A20 fails 7 to 11% of episodes and
   A7 7 to 18%.
2. **The policy is told nothing about the limits.** No reward term or observation flags a joint
   limit, and the reward has no time penalty by design (REWARD_DESIGN.md). A policy at a limit
   receives no signal that would move it off.

Cheap tests, not run:
- a joint-limit margin term in the observation, or a small penalty near the limit;
- replaying seed 1 from seed 3's hard starts with joint 4's range widened, which separates
  "limit blocks the push" from "the policy steers into the limit";
- saving the normalisation statistics in future runs, so replays are exact.

## 7.2 A22: joint-limit margin observed (2026-11-18)

Runs: queue `/tmp/pg/a22.sh`, 11:20 to 12:17, one at a time, same command as section 6, each with
`obs_rms.npz` saved (the first runs to have it). All three finished with exit 0 and full logs.
Seeds 2 and 3 started after the A23 code was added. The A22 env's observation, reward and reset
draws were unchanged by it; the logs gained an always-zero `reward/limit` tag that seed 1's lacks.

`A22 = A20 + distance to the nearer stop of joints 2, 4 and 6, clipped to 0.5 rad` (REWARD_DESIGN.md).

Tables: `python scripts/analyze_runs.py WamCloseCabinetTrayWristLevelerMargin-v0`

| Seed | Episodes | Success | Valid fraction | Spill | Timeout | Episode length | Steps to 50% / 90% | Mean coupling | Success 0-0.5M / 1.5-2M |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 31034 | 0.938 | 0.978 | 0.040 | 0.022 | 61 | 40k / 106k | 1.00 | 0.91 / 0.95 |
| 2 | 30337 | 0.908 | 0.953 | 0.050 | 0.042 | 66 | 48k / 123k | 1.00 | 0.88 / 0.91 |
| 3 | 31437 | 0.936 | 0.971 | 0.046 | 0.018 | 60 | 30k / 50k | 1.00 | 0.93 / 0.94 |

| Slide angle | 15–20° | 20–25° | 25–30° | 30–35° |
|---|---|---|---|---|
| Success | 0.871 | 0.942 | 0.957 | 0.957 |
| Spill | 0.112 | 0.038 | 0.020 | 0.014 |
| Timeout | 0.017 | 0.021 | 0.023 | 0.029 |
| Final plate tilt in successful episodes, median | 1.9° | 2.0° | 2.0° | 2.0° |
| Episodes | 1493 | 1489 | 1555 | 1463 |

Timeout rate by 0.5M-step training window (`scripts/diagnose_timeouts/trend.py`): seed 1 3.9 / 2.1 / 2.0
/ 1.6%, seed 2 3.2 / 2.6 / 2.6 / 2.9%, seed 3 2.6 / 2.4 / 2.4 / 2.3%.

| | Timeout per seed | Pooled timeouts, last 2,000 per seed | Mean success | Mean spill | Steps to 90% |
|---|---|---|---|---|---|
| A20 | 4.4 / 3.4 / 1.2% | 196 of 6,000 (3.3%) | 0.925 | 4.5% | 103k to 205k |
| A22 | 2.2 / 4.2 / 1.8% | 135 of 6,000 (2.3%) | 0.927 | 4.5% | 50k to 123k |

**Finding.** A22 does not reduce timeouts by the bar set before the runs: every seed's rate lies
inside A20's range of 1.2 to 4.4%. The pooled drop from 3.3% to 2.3% comes from A20's worst seed.
Success and spill are unchanged. Learning to 90% was faster on every seed (50k to 123k against 103k
to 205k), but with three seeds per side that is suggestive, not established.

**The margin removed the failure it targeted, and another took its place.** This replay is exact:
it uses the saved normalisation statistics, and its timeout rates match training (1.6 / 3.7 / 2.4%
against 2.2 / 4.2 / 1.8%). Only 4% of A22's 124 replayed timeouts end at the elbow stop, against 44%
for A20 (A20's replay is approximate; see §7.1). A22's timeouts spend 8% of their steps within
0.2 rad of any stop, against 15% for A20. The dominant failure is now the forearm pressing on the
door while the door stays put: 41% of A22 timeouts, against 16% for A20, with a median 350 N over
the last 50 steps and the door still about 1.14 rad open.

**The stuck pushes straddle the door's top edge.** Seed 2 replayed again, 100k steps, with each
forearm-door contact's position, normal and torque about the hinge recorded
(`scripts/diagnose_timeouts/normals.py` and `torque_balance.py`):

| | Contact height above the door's centre, median | Contacts per step | Net torque on the hinge, median | Total of the contact torques' magnitudes, median |
|---|---|---|---|---|
| Timeouts, last 50 steps (59) | +0.200 m (the top edge; half-height 0.2 m) | 2 (≥2 on 70% of steps) | 0.1 N·m; opening on 44% of steps | 19.1 N·m |
| Successes (1,395) | −0.093 m (the face) | 1 | 29.7 N·m, closing | 34.0 N·m |

The forearm rests over the door's top edge and touches both faces, so the contact torques cancel and
the door is pinned. The reward offers no way out. The gate reads valid contact (links 4 to 6 on the
door, nothing else touching), so there is no wrong-contact cost. The force cap charges at most 0.5
per step. Lifting off forfeits reach reward, and nothing pays for it.

**Reading.** Making the joint stops visible did what an observation can do: the policy stopped
parking at them. The rate barely moved because the time went to a different local optimum, which
in this task's geometry is the door's top edge. Two consequences. First, A23, the joint-limit
penalty now running, targets a failure that A22 had already reduced to 4% of timeouts, so I expect
it to match A22 (a prediction made before its results). Second, the edge-straddle is a gap in the
contact gate: "links 4 to 6 touch the door and nothing else does" accepts a contact that cannot
close it. A gate that also required net closing torque, or closing motion, would not pay for the
straddle. That is a contact-geometry condition, which matters for the generality question,
and it needs per-contact geometry the real arm cannot measure.

## 7.3 A23: small joint-limit penalty (2026-11-18)

Runs: queue `/tmp/pg/a23_ee.sh`, 12:20 to 13:15, same command as section 6, exit 0 and full logs
for all three. Launched because A22 missed the timeout bar (§7.2).

`A23 = A22 − 0.05 · Σⱼ ([0.2 − marginⱼ]₊ / 0.2)² per step, joints 2, 4, 6` (REWARD_DESIGN.md).

Tables: `python scripts/analyze_runs.py WamCloseCabinetTrayWristLevelerLimitPenalty-v0`

| Seed | Episodes | Success | Valid fraction | Spill | Timeout | Episode length | Steps to 50% / 90% | Mean coupling | Success 0-0.5M / 1.5-2M |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 28542 | 0.898 | 0.964 | 0.058 | 0.044 | 71 | 38k / 96k | 1.00 | 0.88 / 0.91 |
| 2 | 33055 | 0.888 | 0.949 | 0.092 | 0.020 | 58 | 67k / 134k | 1.00 | 0.86 / 0.88 |
| 3 | 29492 | 0.882 | 0.960 | 0.058 | 0.062 | 68 | 34k / 66k | 1.00 | 0.90 / 0.89 |

| Slide angle | 15–20° | 20–25° | 25–30° | 30–35° |
|---|---|---|---|---|
| Success | 0.830 | 0.890 | 0.913 | 0.929 |
| Spill | 0.142 | 0.067 | 0.036 | 0.033 |
| Timeout | 0.029 | 0.044 | 0.052 | 0.039 |
| Final plate tilt in successful episodes, median | 2.0° | 2.0° | 2.1° | 2.1° |
| Episodes | 1514 | 1440 | 1548 | 1498 |

| | Timeout per seed | Pooled | Success per seed | Mean spill |
|---|---|---|---|---|
| A20 | 4.4 / 3.4 / 1.2% | 3.3% | 0.924 / 0.926 / 0.924 | 4.5% |
| A22 | 2.2 / 4.2 / 1.8% | 2.3% | 0.938 / 0.908 / 0.936 | 4.5% |
| A23 | 4.4 / 2.0 / 6.2% | 4.1% (247 of 6,000) | 0.898 / 0.888 / 0.882 | 6.9% |

**Finding.** A23 did not reduce timeouts. My prediction before the runs, that it would match A22,
was wrong in the unfavourable direction: success is lower on every seed (0.882 to 0.898 against
0.908 to 0.938) and pooled timeouts higher. The penalty itself was close to inert. Over the whole
run, 1.3 to 3.1% of episodes paid any of it, the mean per episode was −0.006 to −0.11 against
returns near 80, and no successful episode in the last 2,000 paid it. In the exact replay (100k
steps per seed, saved statistics), 6% of timeouts end at the elbow stop (A22: 4%) and 30% are
pressing-but-stuck (A22: 41%).

**Reading.** A term this small cannot plausibly cause a 4-point success drop on all three seeds.
The likelier reading is that A23 is effectively a rerun of A22 with the same seeds, and the gap is
run-to-run variance: a tiny change to the reward sends the same seed down a different training
path. That is not proven (a systematic effect this small cannot be excluded with three seeds), but
it sets a working noise floor. **Differences of about 4 points in success, or 2 points in timeout
rate, between 3-seed groups in this setup are not evidence.** That covers A22 against A20, A20
against SeeObject (§7) and Energy against A7 (§6). Neither the margin observation nor the penalty
fixes the timeouts. What does reproduce across A20, A22 and A23 is the edge straddle: a contact the
gate counts as valid, from which nothing in the reward rewards leaving.

## 8. Does the contact gate matter when the end-effector is the nearer tool? And an outcome-only reward (pending)

### 8.0 First build, abandoned (A24v1, 2026-11-18)

Sections 1 to 3 never tested the gate where it would bind: in 400 random-pose starts the forearm
was always nearer the door than the hand, by 11 to 25 cm. A24v1 (`WamCloseCabinetEEFirst-v0`)
rejection-sampled half of all starts so the hand starts at least 3 cm nearer. That needed the start
spread widened from 0.5 to 1.0 rad per joint, on section 3's 7-joint env with no tray.

`python scripts/analyze_ee_first.py WamCloseCabinetEEFirst-v0` (seed 1, the only run; last 1,000 episodes):

| Seed | Start type | Episodes | Success | Valid fraction (successes) | Closed mostly by wrong contact (share of successes) | Spill | Timeout | Episodes with any wrong contact | Steps to 90% success (all starts) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | end-effector-first | 510 | 0.047 | 0.765 | 0.208 | 0.955 | 0.000 | 0.135 | never |
| 1 | normal | 490 | 0.053 | 0.995 | 0.000 | 0.947 | 0.000 | 0.039 | never |

Success stayed at 2 to 7% in every 200k-step window. About 95% of episodes ended in a tilt spill
near step 38, on both start types. Zero actions held the plate on all 40 starts tested, so the
starts are safe; the policy tips the plate when it moves. Wider starts leave the wrist in
configurations where nearly any arm motion tilts the plate, and the 7-joint policy did not learn
level motion from them in 2M steps (section 3's env at 0.5 rad reached 90% in 108k to 342k
steps). With almost no successes there was no contact choice to measure, so the gate question went
unanswered. The queue was stopped by hand at 13:45. A25v1 and A27v1 never ran. One seed, kept as a
record, not a result.

### 8.1 Rebuilt on the wrist leveler (queued)

A24, A25 and A27 rebuilt with the same hand-first starts and rewards, but with A20's controller
holding the plate, plus a tuck (REWARD_DESIGN.md). Pre-training checks: tilt under 5° on 95.7% of
random-action steps (94.8% on the final build); a joint-1 sweep touches the door with the hand on
62% of hand-first starts and 6% of normal ones; forearm-only routes exist on 27% of hand-first
starts (n = 26; none without the tuck) and 38% of normal starts
(`scripts/ee_first/leveler_feasibility.py`).

The first launch (13:50) trained at 277 steps/s and was stopped at 14:05. Two causes, found in
order. Rejection-sampled hand-first resets took up to 325 ms, fixed with a seeded start pool. The
main cause, though: the tuck's 3×3 `np.linalg.inv` made the 8 worker processes contend for CPU
(288 against about 7,000 env steps/s across 8 workers), fixed by applying the same projection
with `np.linalg.solve` (equal to within 3×10⁻¹¹). Its partial log is
`runs/diagnostics/A24-seed1-slow-aborted.log`. Existing envs' resets were checked bit-identical
to the previous commit.

Queue `/tmp/pg/leveler_ee.sh`, restarted 14:21, one run at a time, seed-major (A24, A25, A27, A26
per seed), 2M steps, same command as section 6. Progress: `runs/leveler_ee_progress.log`.

| Label | Env | Question |
|---|---|---|
| A24 | `WamCloseCabinetEEFirstLeveler-v0` | Where the hand is the nearer tool and a controller holds the plate, does the gate keep RL on the forearm, and at what cost? |
| A25 | `WamCloseCabinetEEFirstLevelerUngated-v0` | How often does unconstrained RL use the hand when it is the nearer tool? |
| A27 | `WamCloseCabinetEEFirstLevelerOutcome-v0` | Does an end-of-episode valid-contact check alone keep the forearm in use? |
| A26 | `WamCloseCabinetTrayWristLevelerOutcome-v0` | Does dropping reach shaping and the per-step gate remove the edge straddle in A20's tray env (§7.2)? |

Read A24, A25 and A27 with `scripts/analyze_ee_first.py` (split by start type); read A26 with
`scripts/analyze_runs.py` and `scripts/diagnose_timeouts/profile.py`. Interpreting hand-first
starts: a forearm-only route exists on at least 12% of them by script. A gated success rate far
below that is a learning failure; one near it may be the geometry's limit, not the gate's cost.
Comparison bar, set before results: a difference counts only if seed ranges do not overlap (§7.3).

### 8.2 A24, A25, A27: the gate where the hand is the nearer tool (3 seeds, 2026-11-18)

All nine runs finished with exit 0 and full logs by 17:16.
`python scripts/analyze_ee_first.py WamCloseCabinetEEFirstLeveler-v0 WamCloseCabinetEEFirstLevelerUngated-v0 WamCloseCabinetEEFirstLevelerOutcome-v0`
(last 1,000 episodes per seed, split by `task/ee_first`):

| Label | Start type | Seed 1 success | Seed 2 success | Seed 3 success | Valid fraction (successes) | Closed mostly by wrong contact (successes) | Timeout |
|---|---|---|---|---|---|---|---|
| A24 gated | hand-first | 0.280 | 0.061 | 0.472 | 0.56 mean | 0.41 mean | 0.31–0.63 |
| A24 gated | normal | 0.290 | 0.118 | 0.482 | 0.92 mean | 0.06 mean | 0.34–0.78 |
| A25 ungated | hand-first | 0.567 | 0.558 | 0.588 | 0.40 mean | 0.59 mean | 0.23–0.25 |
| A25 ungated | normal | 0.471 | 0.632 | 0.565 | 0.95 mean | 0.04 mean | 0.29–0.46 |
| A27 outcome | hand-first | 0.532 | 0.440 | 0.502 | 0.41 mean | 0.59 mean | 0.30–0.44 |
| A27 outcome | normal | 0.681 | 0.400 | 0.560 | 0.95 mean | 0.04 mean | 0.27–0.52 |

No condition reached 90% success in any window; success by 0.5M-step training window (seeds
pooled, `0–0.5M / 0.5–1M / 1–1.5M / 1.5–2M`) was A24 `0.18/0.25/0.27/0.28`, A25
`0.43/0.53/0.56/0.56`, A27 `0.42/0.52/0.52/0.51`. All three had flattened by the last two windows
except A24 seed 3, still rising (0.43 → 0.47 over the last two windows). Timeouts (31–78% of
episodes) are far above anything in §4–7 (under 5% except A23's 6.2%), so this task is harder in
this setup than any earlier one, independent of the gate question.

**Finding.** Against the bar (seed ranges must not overlap):

- **Established:** on hand-first starts the gate lowered success. A24 0.061–0.472 against A25
  0.558–0.588. The outcome reward (A27, 0.440–0.532) also scored below ungated there.
- **Not established:** that the gate reduced hand use. Successes closed mostly by the hand average
  41% under A24 against 59% under A25 on hand-first starts, but the seed ranges overlap (0.289–0.478
  against 0.439–0.705). On normal starts hand use is small under every reward (1–13%) and the gate
  does not lower it.
- **Not established:** that the gate's cost is the same on both start types. Mean success under the
  gate is about 30 points below ungated on hand-first starts (0.271 against 0.571) and about 26
  below on normal starts (0.297 against 0.556), but on normal starts the ranges overlap by about a
  point (0.118–0.482 against 0.471–0.632).
- A27 matches A25 in tool choice (59% of hand-first successes closed mostly by the hand under
  both, ranges overlapping): an end-of-episode valid-contact check alone did not keep the forearm
  in use.

A24 seed 2 (hand-first success 0.061) sits below the 27% scripted forearm-only floor from §8.1;
seeds 1 and 3 sit above it. The seeds disagree too much (0.061 to 0.472) to read the floor
comparison as anything but "the gated task is not infeasible".

**Reading.** If the gate's cost were the price of being forced onto the harder tool, it should be
larger on hand-first starts than on normal ones. The means suggest it is about the same on both,
which would make it a general training tax of a per-step contact constraint on a harder task
(wide starts, the leveler, mixed start types), not a tool-specific cost. With overlap on normal
starts, three seeds and no run reaching 90%, that reading is a hypothesis, not a result. What is
solid: on starts where the hand is nearer, dense gated shaping learned the task worse than no
gate, and neither the gate nor the end-of-episode check produced clean forearm-only behaviour.

### 8.3 A26: outcome reward on A20's tray env (3 seeds, 2026-11-18)

`python scripts/analyze_runs.py WamCloseCabinetTrayWristLevelerOutcome-v0`:

| Seed | Episodes | Success | Valid fraction | Spill | Timeout | Episode length | Steps to 50% / 90% | Success 0–0.5M / 1.5–2M |
|---|---|---|---|---|---|---|---|---|
| 1 | 16275 | 0.764 | 0.983 | 0.032 | 0.206 | 117 | 71k / never | 0.62 / 0.74 |
| 2 | 32069 | 0.826 | 0.886 | 0.146 | 0.028 | 58 | 56k / 100k | 0.85 / 0.82 |
| 3 | 28160 | 0.906 | 0.972 | 0.050 | 0.044 | 73 | 45k / 146k | 0.85 / 0.93 |

| Slide angle | 15–20° | 20–25° | 25–30° | 30–35° |
|---|---|---|---|---|
| Success | 0.763 | 0.840 | 0.850 | 0.875 |
| Spill | 0.150 | 0.072 | 0.053 | 0.035 |
| Timeout | 0.088 | 0.089 | 0.096 | 0.090 |
| Final plate tilt in successful episodes, median | 2.2° | 2.3° | 2.2° | 2.2° |

Against A20 (§7): success range [0.764, 0.906] against A20's [0.924, 0.926], no overlap — A26 is
worse. Seed 1's timeout (20.6%) is far outside the other two (2.8%, 4.4%) and outside A20's whole
range (1.2–4.4%); seed 1's timeout rate falls over training (30.2% → 22.3% across 0.5M windows)
but stays high. Spill is not consistently better or worse than A20's 4.5% mean (3.2%, 14.6%, 5.0%).

**Does the edge straddle go away?** Exact replay, 100k steps per seed
(`scripts/diagnose_timeouts/replay.py`, `analyze2.py`, `torque_balance.py`), against A20 and A22
from §7.1–7.2:

| | A20 | A22 | A26 |
|---|---|---|---|
| Timeouts (of replayed episodes) | 7.4% | 2.6% | 7.0% |
| Elbow ends at a hard stop | 44% | 4% | **4%** |
| Pressing-but-stuck (forearm >1N on >80% of last 50 steps, door not moving) | 16% | 41% | 21% |
| Final door angle: <0.5 rad / >1.0 rad | 42% / 40% | 24% / 49% | **64% / 24%** |
| Forearm off the door on the last 50 steps (contact <20% of steps) | 70% | 34% | 50% |

Seed 1 contact torques, last 50 steps of timeouts vs successes (`torque_balance.py`): timeouts had
a median net closing torque of 0.6 N·m against successes' 13.6 N·m, net/total torque ratio 1.00
(A20 and A22's straddles had ratios near 0, i.e. two contacts cancelling), and a median of 1
contact per step (not 2).

**Finding.** The elbow-stop failure A22 fixed with the margin observation (44% → 4%) stayed fixed
in A26 (4%) without that observation — removing the per-step gate and reach shaping also removed
it, by some other route not identified here. But A26's timeouts are not A20/A22's edge straddle:
the forearm's contact force produces real, if weak, closing torque (median 0.6 N·m, not near-zero
from cancelling contacts), on a single contact point, not two. The final door angle in A26's
timeouts sits much closer to success (64% end under 0.5 rad, against A20's 42% and A22's 24%) than
either earlier env. **A26 traded the straddle for a slower, noisier close:** more of its timeouts
are near-misses that ran out of steps, fewer are stuck far from the door.

**Reading.** Dropping reach shaping and the per-step gate changed the shape of A20's failure but
did not reduce the rate (7.0% against 7.4%) or recover its success. The straddle's signature,
cancelling torque from two simultaneous contacts, is absent, which is some evidence the outcome
reward removes the specific trap it was built to remove. But three seeds disagree sharply on
timeout rate (2.8% to 20.6%) and success (0.764 to 0.906), and overall success is below A20's with
no overlap, so the outcome reward is not a clean win: it is less prescriptive, as intended, and
also slower and noisier to train, at least at 2M steps with 3 seeds.

**Taken together, §8.2 and §8.3 narrow what "less prescriptive" bought here.** The outcome reward
removed one named failure (the straddle's torque-cancelling signature) without matching the dense
reward's success rate, and on the harder end-effector-first task it did not clearly separate from
either the gated or ungated dense rewards. Neither result rules out a less prescriptive reward
working better with more steps, more seeds, or a different task; neither establishes that it does.

## 9. The task set on snoopy1: reward across tasks, and the held-out T2 test (2026-11-21)

All 39 runs ran on snoopy1 (SLURM arrays 79719, 79732, 79733; one run per job, 8 async envs, 10 CPUs,
`--save-model`) and finished with exit 0. All 39 logs have the full number of updates (976 for 2M
steps, 4,882 for 10M), and every run has a model and `obs_rms.npz`. Environment: conda `wam`,
MuJoCo 3.3.7, torch 2.4.1, as for every earlier result (`requirements-results.txt`). Cluster runs
used the same trainer and hyperparameters as the laptop runs; they are about half the speed per run.
Task definitions, the feasibility gate and the constraint descriptor are in TASK_SET.md.

### 9.1 One reward template, six tasks, two reward versions (2M steps, 3 seeds each)

`python scripts/analyze_runs.py WamTask<ID>[Outcome]-v0` for each row. Means over each seed's last
500 episodes. Success is the per-seed value for seeds 1 / 2 / 3.

| Task | Reward | Success (seeds 1 / 2 / 3) | Mean success | Mean spill | Mean timeout | Mean valid fraction | Steps to 90% (seeds 1 / 2 / 3) |
|---|---|---|---|---|---|---|---|
| T0 hinged door | dense gated | 0.820 / 0.898 / 0.832 | 0.850 | 0.076 | 0.075 | 0.946 | 137k / 142k / 83k |
| T0 hinged door | outcome | 0.844 / 0.928 / 0.840 | 0.871 | 0.063 | 0.066 | 0.954 | 367k / 149k / 299k |
| T1 sliding door + handle bar | dense gated | 0.932 / 0.976 / 0.972 | 0.960 | 0.033 | 0.007 | 0.963 | 47k / 47k / 35k |
| T1 sliding door + handle bar | outcome | 0.976 / 0.936 / 0.968 | 0.960 | 0.034 | 0.007 | 0.963 | 52k / 63k / 57k |
| T2 drawer push | dense gated | 0.922 / 0.906 / 0.840 | 0.889 | 0.044 | 0.067 | 0.977 | 231k / 176k / 334k |
| T2 drawer push | outcome | 0.676 / 0.886 / 0.798 | 0.787 | 0.089 | 0.125 | 0.933 | never / 323k / never |
| T4x box, x rail | dense gated | 0.894 / 0.914 / 0.864 | 0.891 | 0.073 | 0.037 | 0.950 | 77k / 106k / 47k |
| T4x box, x rail | outcome | 0.848 / 0.904 / 0.804 | 0.852 | 0.081 | 0.068 | 0.943 | 230k / 123k / 43k |
| T4xneg box, x rail, negative goal | dense gated | 0.916 / 0.904 / 0.832 | 0.884 | 0.041 | 0.077 | 0.837 | 225k / 155k / never |
| T4xneg box, x rail, negative goal | outcome | 0.906 / 0.920 / 0.884 | 0.903 | 0.036 | 0.061 | 0.755 | 98k / 128k / 386k |
| T4xy box, planar | dense gated | 0.080 / 0.070 / 0.108 | 0.086 | 0.215 | 0.699 | 0.896 | never / never / never |
| T4xy box, planar | outcome | 0.080 / 0.098 / 0.070 | 0.083 | 0.147 | 0.771 | 0.868 | never / never / never |

"Dense gated" is A20's reward (progress under the per-step contact gate, reach shaping, wrong-contact
penalty); "outcome" is A26's (progress for any closing, no reach shaping or per-step gate, the
success bonus scaled by valid contact at the end). Progress is normalised by each task's range, so a
full close is worth the same on every task (TASK_SET.md). T0 here has A20's dynamics, start distribution and rewards,
and a test confirms identical joint states, rewards and terminations step for step under the same
actions. It is **not** the same learning problem as A20: the observation is the 59-dimensional
descriptor form (A20's is 35-dimensional), so A20's results and these cannot be pooled.

**A20 and T0 differ, and the cause is not isolated.** T0 dense gated is the nearest thing here to
A20 (§7: success 0.924 / 0.926 / 0.924, same seeds 1 to 3). It scores 0.820 / 0.898 / 0.832, with
ranges that do not overlap, so by the bar the task-set version is worse. Two changes happen
together and were not separated: the observation (35 to 59 dimensions, with the descriptor) and the
machine (snoopy1's CPU build of torch against the laptop's). Nothing in these runs says which
mattered. The practical consequence is that absolute numbers on the cluster should not be compared
with laptop numbers from §1 to §8; within-cluster comparisons, which is all this section makes,
are unaffected.

**Finding.** Five of six tasks are learned by both rewards in 2M steps: mean success 0.79 to 0.96,
with no per-task reward tuning beyond the range normalisation. T1 is the easiest (0.96 for both
rewards, 90% success by about 35k to 63k steps). T4xy is not learned by either reward: success 0.07
to 0.11, 70 to 77% timeouts, never near 90% on any of the six seeds.

Against the bar (seed ranges must not overlap), **dense gated and outcome are indistinguishable on
every task**: the ranges overlap on all six pairs. The nearest to separating is T2, where dense gated
is 0.840 to 0.922 and outcome 0.676 to 0.886; the means are 10 points apart (0.889 against 0.787) and
two of the three outcome seeds never reached 90%, but the ranges overlap by 4.6 points, so this is
not established. Outcome's valid fraction on T4xneg (0.755) is below dense gated's (0.837), the one
place the reward changes how the door or box is closed rather than whether; three seeds do not say
whether that gap is real.

**Reading.** This is the clearest answer so far to Ray's reward-generalisation question: the reward
that was tuned on the hinged door (and for which §7.2 found a task-specific trap) transfers to a
sliding door, a drawer and two box-pushing constraints without changes, and the less prescriptive
outcome reward does no better or worse on any of them at this sample size. It is not evidence the
reward would work on tasks unlike pushing (T3 failed the feasibility gate, TASK_SET.md), and T4xy
shows a case, a box free to move in the plane, where neither reward learns in 2M steps. Why T4xy
fails was not investigated: the scripted gate solved only 33% of its starts, so it may be a task that
is hard for four joints and a levelled wrist, or one that needs more steps or exploration help.
Per-task timeout profiles and the check for the §7.2 edge-straddle on the new tasks were not done.

### 9.2 One policy over four tasks, with T2 held out (10M steps, 3 seeds)

`WamMultiTask-v0`: every episode draws one of T0, T1, T4x or T4xneg; the observation has the same
shape on all of them, so the constraint descriptor (joint type, world axis and anchor, position,
velocity, goal, range) is all that says which task is running. Dense gated reward, 10M steps. T2 is
never seen in training. Training-log result, pooled over the four tasks: success 0.644 to 0.876 (mean
0.773), steps to 90% success 300k / 157k / never.

Evaluation: `python scripts/eval_policy.py <train env> <seed> <task> 200 [yaw spread]`. Actions are
sampled from the policy as in training, with the trained observation normalisation frozen (it cannot
drift on an unseen task). 200 episodes per seed, fresh start seeds. Fixture yaw ±20° is the training
range; ±40° is wider than any run saw. These use a different measurement from the training-log
tables above (frozen statistics, new seeds), so the single-task policies were re-evaluated with the
same script for the like-for-like rows.

**Held-out task T2, success (seeds 1 / 2 / 3):**

| Policy | Trained on | Yaw ±20° | Mean | Range | Mean spill | Mean timeout | Yaw ±40° (mean, range) |
|---|---|---|---|---|---|---|---|
| Multi-task | T0, T1, T4x, T4xneg (T2 never seen) | 0.115 / 0.655 / 0.330 | 0.367 | 0.115–0.655 | 0.215 | 0.418 | 0.327, 0.110–0.605 |
| T1 only | T1 | 0.280 / 0.000 / 0.150 | 0.143 | 0.000–0.280 | 0.413 | 0.443 | 0.125, 0.000–0.240 |
| T4x only | T4x | 0.385 / 0.010 / 0.100 | 0.165 | 0.010–0.385 | 0.388 | 0.447 | 0.143, 0.025–0.300 |
| T2 only (ceiling) | T2 | 0.935 / 0.935 / 0.840 | 0.903 | 0.840–0.935 | 0.053 | 0.043 | 0.802, 0.740–0.870 |

The T1-only and T4x-only rows are the control the question needs. They are single-task policies with
the same observation format, trained on one task each, so they show what pushing skill carries to
T2 without having learned to read a descriptor across several tasks.

**Finding (held-out T2).** The multi-task policy reaches 0.367 on average, against 0.143 and 0.165 for
the single-task controls and 0.903 for a policy trained on T2. Against the bar, **the multi-task
policy is not shown to transfer better than the controls**: its range (0.115 to 0.655) overlaps both
controls' (0.000 to 0.280 and 0.010 to 0.385). The means are 20 points apart, so this is a lead, not
a result, and it depends on one seed: seed 2 reaches 0.655 and seed 1 only 0.115. Two things are
solid. The ceiling is far above any zero-shot policy (0.84 to 0.94 against at most 0.655), so T2 is a
task that is learnable and not transferred by any of these policies. And every zero-shot policy
fails differently from the trained one: spill is 21 to 41% and timeouts 42 to 45%, against 5% and
4%, so the held-out failures are mostly the plate tipping and the arm not finishing. Contact
validity of the episodes that do succeed is high but not uniform: the valid fraction of successes
is 0.94 to 1.0 for every policy and seed except one (T4x-only seed 2, 0.492, on 2 successes in 200
episodes), and undefined for T1-only seed 2, which never succeeded.

At yaw ±40° T2 results barely move (multi-task 0.367 to 0.327, controls 0.143 to 0.125, trained
0.903 to 0.802), so fixture orientation is a smaller factor than which task it is.

**In-distribution, like for like (frozen statistics, yaw ±20°, success seeds 1 / 2 / 3):**

| Task | Single-task policy | Multi-task policy | Ranges overlap? |
|---|---|---|---|
| T0 | 0.830 / 0.885 / 0.795 (mean 0.837) | 0.495 / 0.840 / 0.670 (mean 0.668) | yes |
| T1 | 0.940 / 0.990 / 0.970 (mean 0.967) | 0.960 / 0.935 / 0.815 (mean 0.903) | yes |
| T4x | 0.920 / 0.930 / 0.910 (mean 0.920) | 0.865 / 0.890 / 0.510 (mean 0.755) | no, multi-task lower |
| T4xneg | 0.895 / 0.960 / 0.765 (mean 0.873) | 0.730 / 0.750 / 0.755 (mean 0.745) | no, multi-task lower by 1 point at the edge |

The multi-task policy is lower on average on all four tasks (6 to 17 points), and clearly lower
(non-overlapping ranges) on two; the T4xneg separation rests on 0.755 against 0.765. Seed 3 is the
weak seed on T1 and T4x and never reached 90% in training; seed 1 is the weak seed on T0 and
T2. Across seeds, the multi-task policy's T0 and T2 results rank the same way (seed 2 best, then 3,
then 1), which is weak evidence that T2 transfer follows overall policy quality, not task-specific
conditioning. Three seeds cannot test that.

**Reading.** A single policy conditioned on the constraint descriptor learns four tasks in 10M steps
but is worse than separate policies on each, and its zero-shot result on an unseen drawer-push is
better than single-task policies on average but not by a margin the data can separate from
seed-to-seed variation. The honest summary for Ray's question about conditioning: the descriptor
carries enough to learn several tasks with one policy, but this experiment does not show that it
carries enough to generalise to a task it never saw. What would settle it: more seeds (the seed 2
result alone could carry the mean), a harder held-out test (T4xy-like), a held-out task closer to
and further from the training tasks, and a longer run, since seed 3 had not reached 90% by 10M steps.
Also untested: whether option (a), a plain task label, does as well on the training tasks, which
would say whether the descriptor does anything the label does not.

### 9.3 What this does not show

- T3 (pulling) failed the feasibility gate, so nothing here is evidence about pulling.
- T4xneg is a push from the far side, not a pull toward the body (TASK_SET.md).
- Timeout and failure profiles per task, and the §7.2 straddle check, were not run on the new tasks.
- One robot, perfect state, simulation only, 3 seeds, single-task runs at 2M steps and multi-task at 10M.

## 10. More seeds, a task-label baseline, and failure profiles on the new tasks (2026-11-23)

**Runs.** 28 follow-up runs went to snoopy1 on 11/21 (arrays 81790, 81796). SLURM preemption
cancelled 13 of them partway through. They saved no model and were not requeued, yet my launcher
logged each as "done, exit 0": it wrote the progress line regardless of what had happened. Checking
every run's update count and model file caught them. Their partial folders and logs are in
`runs/aborted/` on the server. The 13 were rerun on 11/22 (arrays 82727, 82728) with a fixed
launcher that requeues on preemption, clears unfinished folders on restart, and logs the real exit
status and whether a model was saved. All 13 reruns finished with full update counts and a saved
model, and none was preempted. Every policy below now has 8 seeds.

Evaluations: `scripts/run_followup_evals.sh`, which runs `scripts/eval_policy.py` (frozen
normalisation, sampled actions, 200 episodes, fixture yaw ±20°) for every seed that has a model,
writing to `runs/diagnostics/evals/`. Profiles: `scripts/diagnose_timeouts/run_task_profiles.sh`
(exact replays, 60k steps per seed), summarised by `task_profile.py --summary`, with counts in
`runs/diagnostics/profiles/counts.txt`.

### 10.1 The task-label baseline

`WamMultiTaskLabel-v0` (TASK_SET.md, option a): the same four training tasks as the descriptor
policy (T0, T1, T4x, T4xneg), 10M steps. Each joint slot keeps [valid, position, velocity, goal,
range] but drops the joint's type, axis and anchor, and a one-hot over the four tasks is appended
(observation 47, against 59). On held-out T2 the one-hot is all zeros, so this policy gets **no**
task identity there; the descriptor policy gets T2's joint geometry, which no training task had.

### 10.2 Held-out T2 with 8 seeds

Success on T2 (`python scripts/summarize_followup.py`, from `runs/diagnostics/evals/`):

| Policy | Seeds | Success per seed | Mean | Median | Range | Mean spill | Mean timeout |
|---|---|---|---|---|---|---|---|
| Descriptor, 4 training tasks | 1,2,3,4,5,6,7,8 | 0.115 / 0.655 / 0.330 / 0.050 / 0.290 / 0.575 / 0.410 / 0.370 | 0.349 | 0.350 | 0.050–0.655 | 0.265 | 0.386 |
| Task label, same 4 tasks | 1,2,3,4,5,6,7,8 | 0.220 / 0.520 / 0.500 / 0.560 / 0.485 / 0.340 / 0.350 / 0.405 | 0.422 | 0.445 | 0.220–0.560 | 0.179 | 0.399 |
| T1 only | 1,2,3,4,5,6,7,8 | 0.280 / 0.000 / 0.150 / 0.140 / 0.690 / 0.415 / 0.100 / 0.405 | 0.273 | 0.215 | 0.000–0.690 | 0.416 | 0.311 |
| T4x only | 1,2,3,4,5,6,7,8 | 0.385 / 0.010 / 0.100 / 0.550 / 0.040 / 0.255 / 0.220 / 0.505 | 0.258 | 0.237 | 0.010–0.550 | 0.421 | 0.321 |
| T2 only (ceiling) | 1,2,3,4,5,6,7,8 | 0.935 / 0.935 / 0.840 / 0.800 / 0.915 / 0.740 / 0.685 / 0.850 | 0.838 | 0.845 | 0.685–0.935 | 0.069 | 0.094 |

Comparisons. The bar: a difference counts only if seed ranges do not overlap. p is the exact
two-sided rank-sum test across seeds, a second look, not a replacement for the bar:

| Comparison | Means | Ranges overlap | p |
|---|---|---|---|
| Descriptor vs label | 0.349 vs 0.422 | yes | 0.505 |
| Descriptor vs T1-only | 0.349 vs 0.273 | yes | 0.574 |
| Descriptor vs T4x-only | 0.349 vs 0.258 | yes | 0.328 |
| Descriptor vs both controls pooled | 0.349 vs 0.265 | yes | 0.343 |
| Label vs T1-only | 0.422 vs 0.273 | yes | 0.111 |
| Label vs T4x-only | 0.422 vs 0.258 | yes | 0.137 |
| Label vs both controls pooled | 0.422 vs 0.265 | yes | 0.068 |

**Finding.** **No policy is shown to transfer to the held-out task better than any other.** Every
range overlaps, and no p is below 0.05. The descriptor policy (mean 0.349) is indistinguishable from
the single-task controls (0.273 and 0.258; p 0.33 to 0.57) and from the label policy (0.422; p 0.51).
The label policy is the nearest to a difference, against the pooled controls (0.422 vs 0.265,
p 0.068), so it is the only lead worth a further test. All zero-shot policies are far below the
ceiling: 0.685 to 0.935 for a policy trained on T2, against at most 0.690 for any zero-shot seed.
Zero-shot spill is lower for the label policy (18%) than for the descriptor (27%) and the controls
(42%). Spill is the one consistent difference in failure mode, but it was not tested across seeds.

**How the estimate moved with seeds:**

| Seeds per policy | Descriptor mean | Label mean | Reading at the time |
|---|---|---|---|
| 3 (§9.2) | 0.367 | – | a 20-point lead over the controls, "a lead, not a result" |
| 4 (§10, partial) | 0.288 | 0.516 | the descriptor lead gone; the label policy looked best and most consistent (0.485–0.560) |
| 8 (final) | 0.349 | 0.422 | nothing separable; the label policy's range widened to 0.220–0.560 |

Each partial reading reversed when more seeds came in. Single seeds here range from 0.00 to 0.69
for the same policy, so 3 or 4 seeds cannot rank these policies. That is the main methodological
result of this section.

**In distribution**, the descriptor and label policies on the four training tasks (8 seeds each):

| Task | Descriptor | Label | Ranges overlap | p |
|---|---|---|---|---|
| T0 | 0.495 / 0.840 / 0.670 / 0.420 / 0.740 / 0.790 / 0.825 / 0.635 (mean 0.677) | 0.315 / 0.650 / 0.840 / 0.805 / 0.765 / 0.705 / 0.665 / 0.820 (mean 0.696) | yes | 0.824 |
| T1 | 0.960 / 0.935 / 0.815 / 0.815 / 0.940 / 0.965 / 0.900 / 0.810 (mean 0.892) | 0.930 / 0.955 / 0.895 / 0.950 / 0.955 / 0.940 / 0.940 / 0.935 (mean 0.938) | yes | 0.399 |
| T4x | 0.865 / 0.890 / 0.510 / 0.435 / 0.765 / 0.750 / 0.910 / 0.515 (mean 0.705) | 0.645 / 0.795 / 0.550 / 0.855 / 0.740 / 0.930 / 0.815 / 0.895 (mean 0.778) | yes | 0.505 |
| T4xneg | 0.730 / 0.750 / 0.755 / 0.820 / 0.550 / 0.675 / 0.790 / 0.535 (mean 0.701) | 0.395 / 0.910 / 0.735 / 0.765 / 0.845 / 0.570 / 0.850 / 0.655 (mean 0.716) | yes | 0.574 |

The label policy's mean is higher on all four tasks, and no difference is separable (p 0.40 to 0.82).

**Reading.** At 10M steps and 8 seeds, the joint geometry in the descriptor buys nothing measurable
over a plain task label, on the training tasks or on an unseen one. A policy conditioned on either
learns four tasks, but neither generalises to a fifth well enough to stand out from policies trained
on a single, different task. That answers Ray's conditioning question for this setup in the
negative, with the caveats that matter:
- one held-out task, nearer to some training tasks than others;
- 10M steps per multi-task policy; longer training was not tried;
- only the geometric form of the descriptor tested, with no richer conditioning (contact location, a
  learned encoder, history).

The out-of-distribution explanation from the partial draft (the descriptor's T2 axis and anchor lie
outside anything it trained on) remains untested. With no reliable gap left to explain, it is no
longer needed to account for the data. The test that would still be informative is a held-out task
interpolated between training tasks, for example a slide axis between T1's and T4x's, where the
descriptor has something to interpolate and a label does not.

### 10.3 Failure profiles on the new tasks (seeds 1–3, both rewards)

Exact replays from the saved normalisation statistics. Replay success matches the training logs
within about 2 points (T0 0.838 vs 0.850, T1 0.959 vs 0.960, T2 0.905 vs 0.889, T4xy 0.091 vs 0.086).
Counts are episodes pooled over three seeds. "Stuck" means a timeout with forearm contact on more
than 80% of its last 50 steps and no motion. "Off" means forearm contact on under 20% of those
steps. "At a stop" means a joint within 0.05 rad of a hard limit at the end. An "early spill"
tips the plate before 10% progress, with no forearm contact at all.

| Task, reward | Episodes | Spills | Timeouts | Stuck | Off the object | Joint at a stop | Early spills |
|---|---|---|---|---|---|---|---|
| T0 dense | 2,260 | 192 | 175 | 31 | 109 | 75 | 102 of 192 |
| T0 outcome | 2,249 | 123 | 180 | 18 | 141 | 39 | 44 of 123 |
| T1 dense | 3,056 | 111 | 15 | 4 | 7 | 0 | 83 of 111 |
| T1 outcome | 3,077 | 87 | 16 | 4 | 6 | 2 | 57 of 87 |
| T2 dense | 2,273 | 115 | 103 | 5 | 86 | 33 | 17 of 115 |
| T2 outcome | 2,051 | 169 | 225 | 4 | 202 | 77 | 45 of 169 |
| T4x dense | 2,905 | 223 | 79 | 3 | 74 | 29 | 87 of 223 |
| T4x outcome | 2,408 | 197 | 174 | 6 | 159 | 34 | 91 of 197 |
| T4xneg dense | 2,150 | 72 | 175 | 9 | 155 | 32 | 24 of 72 |
| T4xneg outcome | 2,616 | 96 | 131 | 15 | 98 | 70 | 16 of 96 |
| T4xy dense | 824 | 167 | 582 | 2 | 472 | 9 | 23 of 167 |
| T4xy outcome | 795 | 105 | 622 | 2 | 566 | 53 | 26 of 105 |

**Finding.**

1. **The edge straddle of §7.2 does not recur on the new tasks** (corrected in §10.5 for T1: with 8 seeds a rare straddle-like stall appears there). Stuck pushing is 2 to 15
   episodes per group on T1–T4xy, under 11% of timeouts. Too few to characterise; anecdotally, the
   handful on T1 push perpendicular to the slide (zero force along the axis, as on the panel's face
   rather than the bar), and the 5 on T2 dense have cancelling contacts (net/total 0.10).
2. **The dominant timeout on every new task is the forearm off the object at the end:** 74 to 94%
   of timeouts on T2, T4x, T4xneg and T4xy. The arm has lost or never regained contact. It is not
   pressing on something that won't move.
3. **A joint at a hard stop recurs:** 32 to 53% of timeouts on T2 outcome, T4xneg outcome and T0
   dense, echoing A20 in §7.1 (44%). It is near zero on T1.
4. **T1's remaining failures are early spills.** 83 of 111 dense spills tip the plate before any
   progress and without touching the panel (median step 10). That is the policy's opening motion,
   not the task.
5. **T4xy fails by losing the box, not by being stuck:** 81 to 91% of its timeouts end with the
   forearm off the box, median distance left 0.41 to 0.50 of the range. So the box usually moves
   about halfway before contact is lost. One hypothesis: the box, free in the plane, slides
   sideways off the forearm. That was not measured: the profile records the distance to the goal,
   not the direction of the error.

**Reading.** The new tasks fail differently from the door. The reward's §7.2 loophole, a contact
the gate accepts that cannot move the object, did not reappear when the geometry changed. That
fits it being a property of the door's top edge, not of the reward. What does recur is the arm
losing contact (off-object timeouts) and joints running into stops. Neither the dense reward nor
the outcome reward prevents either one. A contact-maintenance or joint-margin signal is the next
thing to test, and per §7.2–7.3 an observation alone did not fix the stops.

### 10.4 Training curves with 8 seeds (2026-11-23)

**Single-task, dense gated, 8 seeds** (`python scripts/analyze_runs.py WamTask<ID>-v0`; tables in
`runs/diagnostics/tables/`). Success over each seed's last 500 training episodes:

| Task | Success, seeds 1–8 | Mean | Range | Steps to 90% |
|---|---|---|---|---|
| T1 sliding door | 0.932 / 0.976 / 0.972 / 0.940 / 0.984 / 0.912 / 0.956 / 0.970 | 0.955 | 0.912–0.984 | 35k–55k on every seed |
| T2 drawer push | 0.922 / 0.906 / 0.840 / 0.772 / 0.910 / 0.718 / 0.644 / 0.824 | 0.817 | 0.644–0.922 | 176k–1,031k; never on seeds 4, 6, 7 |
| T4x box, x rail | 0.894 / 0.914 / 0.864 / 0.918 / 0.874 / 0.936 / 0.952 / 0.930 | 0.910 | 0.864–0.952 | 47k–111k on every seed |

T1 and T4x hold their 3-seed picture. T2 does not: its mean drops from 0.889 (3 seeds, §9.1) to
0.817, and three of the five new seeds never reach 90%. That weakens §9.1's nearest-to-separating
case further. Dense gated on T2 now spans 0.644 to 0.922 against outcome's 0.676 to 0.886 (still 3
seeds), so the two rewards are even less distinguishable there than §9.1 reported.

**Multi-task policies, per training task** (`python scripts/analyze_multitask.py WamMultiTask-v0
WamMultiTaskLabel-v0`). Each seed's episodes are split by the logged `task/index`. "Final" is the
task's last 500 episodes; "steps to 90%" is the first total training step at which that task's
200-episode rolling success exceeded 0.9:

**Descriptor policy** (`WamMultiTask-v0`)
| Seed | T0 final / steps to 90% | T1 final / steps to 90% | T4x final / steps to 90% | T4xneg final / steps to 90% | Pooled final |
|---|---|---|---|---|---|
| 1 | 0.518 / 545k | 0.962 / 105k | 0.812 / 418k | 0.798 / 361k | 0.771 |
| 2 | 0.854 / 452k | 0.928 / 119k | 0.874 / 514k | 0.796 / 776k | 0.865 |
| 3 | 0.708 / never | 0.796 / 171k | 0.478 / never | 0.706 / 882k | 0.670 |
| 4 | 0.424 / 161k | 0.766 / 132k | 0.480 / 151k | 0.828 / 279k | 0.625 |
| 5 | 0.738 / never | 0.928 / 115k | 0.748 / 208k | 0.558 / never | 0.740 |
| 6 | 0.750 / never | 0.944 / 176k | 0.744 / never | 0.640 / never | 0.765 |
| 7 | 0.782 / never | 0.888 / 159k | 0.920 / 565k | 0.794 / 414k | 0.846 |
| 8 | 0.656 / 1110k | 0.808 / 254k | 0.576 / never | 0.458 / 513k | 0.622 |
| mean | 0.679 (range 0.424–0.854) | 0.878 (range 0.766–0.962) | 0.704 (range 0.478–0.920) | 0.697 (range 0.458–0.828) | |

**Label policy** (`WamMultiTaskLabel-v0`)
| Seed | T0 final / steps to 90% | T1 final / steps to 90% | T4x final / steps to 90% | T4xneg final / steps to 90% | Pooled final |
|---|---|---|---|---|---|
| 1 | 0.318 / never | 0.910 / 100k | 0.674 / 134k | 0.352 / 202k | 0.556 |
| 2 | 0.662 / 399k | 0.938 / 120k | 0.700 / 200k | 0.882 / 247k | 0.796 |
| 3 | 0.822 / 365k | 0.922 / 188k | 0.492 / never | 0.748 / never | 0.744 |
| 4 | 0.784 / 413k | 0.938 / 147k | 0.816 / 182k | 0.784 / never | 0.830 |
| 5 | 0.736 / 180k | 0.934 / 157k | 0.678 / never | 0.788 / 195k | 0.787 |
| 6 | 0.660 / never | 0.906 / 136k | 0.930 / 197k | 0.556 / never | 0.767 |
| 7 | 0.666 / 256k | 0.940 / 133k | 0.826 / 164k | 0.852 / 976k | 0.827 |
| 8 | 0.846 / 146k | 0.934 / 108k | 0.906 / 122k | 0.546 / 165k | 0.806 |
| mean | 0.687 (range 0.318–0.846) | 0.928 (range 0.906–0.940) | 0.753 (range 0.492–0.930) | 0.689 (range 0.352–0.882) | |

**Against the single-task policies** (same training-log measure; T0 and T4xneg have only 3
single-task seeds). The multi-task ranges overlap the single-task ones on T0, T1 and T4x for both
policies. On T4xneg the descriptor policy's range (0.458–0.828) sits entirely below the single-task
range (0.832–0.916), by 0.4 points at the edge. The label policy's does not (0.352–0.882). On T1 the
label policy is tight (0.906–0.940), the descriptor policy wide (0.766–0.962); no difference between
them is separable on any task, as §10.2 found from the evaluations.

**Tasks interfere.** 23 of the descriptor policy's 32 seed-task pairs, and 25 of the label
policy's, reached 90% at some point during training. In each policy, **7 of those ended below 0.75**:
for example, descriptor seed 4 reached 90% on T0 at 161k steps and finished at 0.424, and label seed
1 reached 90% on T4xneg at 202k and finished at 0.352. The single-task runs show nothing like this:
every T1 and T4x seed that reached 90% ended at 0.86 or above. The rate is the same for both
observation forms, so the interference comes from training four tasks in one policy, not from
how the task is described. It is a plausible reason the multi-task policies trail the single-task
ones, and it was not addressed: no per-task loss balancing, no curriculum.

### 10.5 Failure profiles: 8 seeds of T1, T2, T4x, and every policy on held-out T2 (2026-11-23)

`scripts/diagnose_timeouts/run_followup_profiles.sh`: exact replays, 60k steps per single-task seed
on its own task, and 30k steps per seed for every policy on T2. `task_profile.py` can now replay a
policy on a task it did not train on (`[task]` argument), built as the evaluator builds it.
Counts are in `runs/diagnostics/profiles/counts_8seed.txt` and `runs/diagnostics/profiles_t2/`.

**Single-task, 8 seeds, own task** (dense gated; categories as in §10.3):

| Task | Episodes | Spills (early) | Timeouts | Stuck | Off the object | Joint at a stop |
|---|---|---|---|---|---|---|
| T1 | 8,097 | 324 (241) | 67 | 40 | 15 | 0 |
| T2 | 5,668 | 391 (89) | 509 | 27 | 417 | 183 |
| T4x | 7,835 | 472 (196) | 216 | 11 | 196 | 91 |

T2 and T4x keep the §10.3 picture: timeouts end with the forearm off the object, and in 36% (T2)
and 42% (T4x) of them a joint is at a stop. **Correction to §10.3 for T1:** with 8 seeds, 40 of
T1's 67 timeouts are stuck pushes, against 4 of 15 at 3 seeds. They cluster in seeds 4, 7 and 8 (33
of the 40) and take two forms. Some steps carry no force along the slide (the forearm presses
across the slide, as on the panel's face rather than the bar). Others carry large forces that
cancel (net/total 0.00 on 69% of stuck steps), which is straddle-like. So a straddle-like stall
does occur on T1, but rarely: 40 of 8,097 episodes (0.5%). T1's main failure remains early spills,
241 of its 324.

**Every policy on held-out T2, 8 seeds each:**

| Policy | Episodes | Success | Spills (early, before 10% progress and any contact) | Timeouts | Off the object | Joint at a stop |
|---|---|---|---|---|---|---|
| Descriptor | 1,615 | 0.356 | 444 (254) | 596 | 536 | **457** |
| Label | 1,582 | 0.443 | 286 (98) | 596 | 511 | 102 |
| T1 only | 1,757 | 0.282 | 715 (420) | 548 | 474 | 47 |
| T4x only | 1,944 | 0.308 | 843 (515) | 502 | 455 | 275 |
| T2 only (trained) | 2,837 | 0.841 | 200 (42) | 253 | 207 | 85 |

Replay success matches the §10.2 evaluations within about 2 points for four policies, and is 5 points higher for T4x-only (0.308 against 0.258); the replays draw fresh start seeds and are about 200 episodes per seed.

**Finding.**
1. **Zero-shot spills mostly happen before the policy touches the drawer:** 57 to 61% of spills for
   the descriptor, T1-only and T4x-only policies, against 34% for the label policy and 21% for the
   T2-trained one. Untrained policies make an opening move that suits their training tasks and
   tips the plate on this one (inference, from timing and the absence of contact).
2. **The zero-shot policies' timeouts look alike.** 86 to 91% end with the forearm off the drawer
   (82% for the T2-trained policy), and stuck pushing is 3 to 6% for every policy.
3. **The descriptor policy drives a joint into its stop on T2.** This is the clearest difference
   in the whole comparison. 457 of its 596 timeouts end with a joint within 0.05 rad of a hard
   limit, 55% or more on 7 of 8 seeds (per seed: 96, 65, 88, 95, 96, 10, 55, 100%). The label policy
   does this in 102 of 596, at most 52% on any seed. On the per-seed shares, descriptor against
   label gives rank-sum p = 0.002, and against T1-only p = 0.001. One low descriptor seed keeps the
   ranges overlapping, so the bar is not met. T4x-only also hits stops on T2 (275 of 502;
   p = 0.23 against the descriptor). Which joint, and why, was not recorded.

**Reading (inference, not tested).** The two conditioned policies trained on the same four tasks
and differ only in how the task is described, yet on the unseen drawer the descriptor policy
consistently commits to a motion that runs a joint into its stop, and the label policy, which sees
no task identity, does not. T4x-only does it too. That fits the descriptor policy mapping T2's
joint description onto something like its box-pushing behaviour, a motion the drawer's geometry
cannot accommodate, and the label policy falling back to a more average, less committed motion. It
is the first evidence that the descriptor changes *how* a policy acts on an unseen task, even
though it does not change *how often* it succeeds (§10.2). Testing it needs the joint identity at the
stop, and the descriptor policy evaluated on T2 with T4x's joint description swapped in.

## 11. No-RL baseline: a scripted push given what G2 is told (2026-11-25)

**Question.** With G2's contact target and push direction, does a held-out drawer need RL at all?
If a scripted controller given the same information matches G2 on T2, G2 is not an RL result.

**Setup** (`scripts/scripted_baseline.py`). Per step: read `contact_target()` (the source of G2's
contact features); drive the forearm point G2 is measured from (`link5_c2`) by damped least squares
on joints 1–4 to 5 cm behind the target; once within 1.5 cm or touching the object with the forearm,
push along the push direction. The wrist leveler holds the plate. One attempt, one forearm point (the
11/18 feasibility gate took the best of four, so its 80% on T2 was an upper bound). Same env, 250-step
limit, ±20° yaw and reset seeds as `eval_policy.py`, so seed s sees the policy of seed s's starts.
The controller is deterministic, so "seeds" here vary only the starts, not a training run.

Two fixes before the full run, both found on 20 smoke-test episodes:
- Uncapped, the DLS step saturates the action at full rate and throws the cylinder within 5–8 steps.
  Added a speed cap (max |action|). **Tuned on training tasks only** (seed 1, 60 episodes):

  | cap | F (G training family) | T4x |
  |---|---|---|
  | 1.0 | 0.567 | 0.800 |
  | 0.5 | **0.633** | 0.850 |
  | 0.3 | 0.600 | **0.883** |
  | 0.2 | 0.517 | 0.650 |
  | 0.1 | 0.183 | 0.200 |

  Frozen at **0.5** (best on F, the distribution G1–G3 train on; T4x within 3 points) before any T2 run.
- The push phase never started: the 4 cm capsule radius stops the point ~3 cm short of a 1.5 cm
  threshold. Added the forearm-contact trigger (on hardware: external joint torque).

**Result** (8 start sets × 200 episodes, cap 0.5; raw lines in `artifacts/scripted_baseline/`):

| Task | Success, mean (range) | Forearm-only | Spill | Timeout | Reached push phase |
|---|---|---|---|---|---|
| **T2 (held out)** | **0.621** (0.585–0.645) | 0.616 | 0.108 | 0.271 | 0.65 |
| T1 | 0.846 (0.810–0.880) | 0.799 | 0.054 | 0.100 | 0.84 |
| T4x | 0.848 (0.830–0.875) | 0.837 | 0.067 | 0.086 | 0.90 |
| T4xneg | 0.728 (0.700–0.800) | 0.333 | 0.070 | 0.202 | 0.54 |
| T2, cap 0.3 (sensitivity, not the frozen choice) | 0.662 (0.610–0.695) | 0.651 | 0.113 | 0.226 | 0.55 |

**Finding.**
1. With no training, the script gets **0.62 on T2**, above every zero-shot policy so far (§10: label
   0.422, descriptor 0.349) and below the T2-trained RL policy (0.838). So in-distribution RL beats this
   script on T2 by ~22 points, and no cross-task RL policy has yet matched it.
2. Its main loss is timeouts (27%): in ~35% of episodes the forearm never reaches the pre-contact point.
   The straight-line approach has no collision or posture reasoning, so this is a lower bound on what a
   model-based controller with the same information gets, not a ceiling.
3. On T4xneg it mostly closes with the hand (forearm-only 0.33 of 0.73), as the gate found on 11/18.

**What this sets for G1–G3** (added before any G result is in): G2's T2 number is read against 0.62.
G2 "beats scripted control" if its mean is ≥ 0.77 (15 points above) and its seed range does not overlap
0.585–0.645. A G2 between 0.62 and 0.77 is better than a naive script but not by enough to carry an
RL claim, since a better approach planner would close part of that gap.

**Threshold replaced 2026-11-27 (Josiah's decision, made after the G results; it does not change any G
verdict, which fails both versions).** A better approach planner did close part of the gap. The staged
script (`scripted_baseline.py ... staged`: 0.15 m standoff in front of the face, in along the normal,
arm point sliding over the four forearm points, nothing tuned on T2) scores **0.755 (0.705–0.790)** on
the same 8 × 200 starts (log.md, 2026-11-27). The same rule applied to it: a policy "beats scripted
control" on T2 if its mean is **≥ 0.905** (15 points above 0.755) **and** its seed range does not overlap
**0.705–0.790**. That is above the T2-*trained* RL ceiling (0.838), so as configured no policy in this
repo, held-out or trained on T2, would clear it. The 0.77 bar above is superseded.

## 12. Env v2 and the frozen scripted baseline, staged2 (2026-11-27)

### 12.1 Env v2: two tray-contact artifacts removed (commit 3d97fc4)
A 1 kHz spill audit of the staged script on T2 (`scripts/diagnose_timeouts/spill_audit.py`) found that
64% of its spills were simulator artifacts:
- **Contact "pops", 34%.** MuJoCo's native CCD sometimes reported a fixed 48 mm plate–cylinder
  penetration for one substep, and the solver ejected the cylinder.
- **Unsettled spawn, 30%.** The cylinder was placed 1 mm above the plate and the arm moved before it
  landed.

v2 sets `ccd_tolerance` 1e-9 on the tray contact: 0 pops in 800 episodes, against 35, at the same speed.
It also holds the arm still for 250 ms at reset so the cylinder lands and the arm's gravity sag decays.
64- and 32-sided convex prisms also removed the pops, but made 5 of 10 seeds creep at 95% of the slide
angle, against 0–1 of 10 for primitive cylinders, so v2 keeps primitives. Starts are drawn before the
settle, so the start sequence is unchanged.

On v2, staged-T2 spill fell from 12.0% to 7.4%, and success moved only +0.6 to +2 points per task:
episodes that used to spill now time out. G3 on v2 T2 averages 0.563 (v1: 0.561), so its failures
are its own. Its checkpoints were trained on v1, so pop damage during training is not undone.

Two follow-up checks found no reason for a v3 (log.md):
- A slew-rate ramp on the scripted action is a weak negative result: −2 to +4 points.
- A plate-impulse spike check shows the cylinder chatters on the plate under any arm motion. That
  happens at nearly the same rate in successes and spills (70 vs 79–94 spikes per 1,000 substeps),
  and is only weakly tied to the 50 Hz control updates.

These remain sim-to-real caveats.

### 12.2 Where the staged script's timeouts come from
`scripts/diagnose_timeouts/stage_audit.py` reads a per-step trace (`scripted_baseline.py` arg 8)
and puts every timeout into one bucket (rules in log.md, fixed before running).

**No task was clock-limited.** At standoff arrival 206–227 of 250 steps remained, and successes
needed 37–51.

**Stage 1:**
- Handoff misses: 1–4.5%. The arm hovered about 3.5 cm from the standoff against a 3 cm trigger.
- Obstacle blocks: T0 9.2%, T4xneg 6.0%.

**Stage 2:**
- T2 lost contact: 10.9%. Once pushing, the script never re-aimed or re-acquired.
- T0 wedged near closed: 3.9%.

### 12.3 Fixes and calibration (thresholds chosen on F, T0 and T1 only)
`staged2` = staged plus:
- **Stall handoff:** within 6 cm of the standoff and closing by less than 0.5 mm per step. This is in
  addition to the 3 cm radius.
- **Contact-loss fallback:** if the forearm–mover distance exceeds 3 cm while pushing, return to the
  standoff, at most 3 times per episode.

A sideways re-aim toward the contact target during the push was also tried, on F, T0 and T1, 2 × 200
episodes each:
- **Full gain:** −4.7 / −14.5 / −13.5 points. It drags a forearm that made contact elsewhere along the
  face.
- **Gains 0.1, 0.25 and 0.5:** F/T0/T1 means 0.768, 0.763 and 0.731, against 0.771 without steering.

Steering is therefore off, and available only as an ablation flag.

Each piece alone on F / T0 / T1:
- Stall handoff: +5.0 / +1.2 / +6.2.
- Fallback: about 0.

### 12.4 Frozen baseline: staged2 on v2, 8 seeds × 200 starts
Same starts as every eval since §11. The script is deterministic. Ranges are over the 8 start sets.

| Task | staged | **staged2** (range) | Reached standoff | Push from standoff | Spill | Timeout |
|---|---|---|---|---|---|---|
| F (training family) | 0.703 | **0.761** (0.730–0.780) | 0.76 | 0.890 | 11.4% | 12.6% |
| T0 hinged door | 0.700 | **0.710** (0.655–0.740) | 0.86 | 0.829 | 11.9% | 17.1% |
| T1 sliding door | 0.831 | **0.879** (0.855–0.910) | 0.95 | 0.924 | 9.3% | 2.8% |
| T2 drawer (held out for RL) | 0.761 | **0.840** (0.805–0.885) | 0.87 | 0.910 | 7.4% | 8.6% |
| T4x box | 0.889 | **0.907** (0.875–0.955) | 0.97 | 0.931 | 8.7% | 0.6% |
| T4xneg box | 0.827 | **0.873** (0.835–0.905) | 0.77 | 0.955 | 4.0% | 8.9% |

Timeout buckets for staged2, as a share of all episodes:

| Bucket | F | T0 | T1 | T2 | T4x | T4xneg |
|---|---|---|---|---|---|---|
| S1 handoff miss | 0.1% | 0.1% | 0 | 0.1% | 0 | 0 |
| S1 obstacle block | 6.2% | **9.4%** | 0.8% | 3.1% | 0 | 5.5% |
| S1 out of time | 0.6% | 0.3% | 0 | 0.3% | 0 | 0.6% |
| S1 IK / joint-limit stall | 1.9% | 0.7% | 0.9% | 0.5% | 0.1% | 1.0% |
| S2 out of time | 1.9% | 1.9% | 0.1% | 0.9% | 0.1% | 0.7% |
| S2 wedged | 1.4% | **4.3%** | 0.3% | 1.2% | 0.1% | 0.9% |
| S2 lost contact | 0.4% | 0.3% | 0.6% | 2.2% | 0.3% | 0.1% |
| S2 other | 0.1% | 0.1% | 0 | 0.3% | 0.1% | 0.1% |

**What this shows**
- With no learning, staged2 gets 0.71–0.91 on every task. On held-out T2 (0.840) its seed range does
  not overlap staged's.
- On F, G3's own training family, it scores 0.761. The G3 seeds checked there scored 0.31–0.74.
- **What is left:**
  - Stage-1 obstacle blocks: T0 9.4%, F 6.2%, T4xneg 5.5%, T2 3.1%.
  - T0's near-closed wedge: 4.3%.
  - Spill: 4–12%, the largest single failure on T1, T4x and F.

**What this does not show**
- **The bar.** The §11 "beats scripted control" bar (0.905, from staged's 0.755) is now outdated.
  Applying the same rule to staged2 gives 0.99, which is not meaningful. G3's best T2 seed (0.865 on
  v2) falls within staged2's range. How the RL comparison is restated is open (log.md).
- **The steering result is narrow.** It applies to this one form of steering, toward the face-centre
  point. Steering that keeps the forearm inside the face's extent was not tried.

## 13. Pre-registered evaluation for F_geom RL (written 2026-11-27, before any F_geom env or training exists)

The gates are Josiah's, recorded verbatim below. `scripts/eval_preregistered.py` enforces them and prints
pass/fail per gate. Everything under "Operational definitions" is my translation of the wording into
measurements. It was fixed at the same time as the gates, and Josiah should confirm it before the first run.
F_geom, its housing-force term and the parameters d_recess, d_overhang and w_h_force do not exist in the repo
yet. The evaluator takes the F_geom env ids as arguments.

### 13.1 Protocol
8 seeds × 200 episodes = 1,600 episodes per task on frozen env v2 (`7853b61` physics; current code `e5cf7eb`),
identical reset seeds across methods, deterministic policy mean (`det=True`). Report 8-seed mean, worst-seed
score, IQR, and 95% bootstrap CIs.

### 13.2 Gates (verbatim)
**Gate 0 (must pass before any OOD evaluation):**
1. Final policy action std ≥ 0.065 on 8/8 seeds.
2. F_geom held-out validation (1,000 unseen procedural scenes): mean ≥ 0.850, worst seed ≥ 0.780.
3. Sustained housing collision rate (F_house > 10 N for ≥ 5 steps) ≤ 5.0%.

**Phase 1 OOD bars (T0, T1, T2, T4x, T4xneg vs. the fixed staged script S_script):**
1. Macro-mean: S_PPO ≥ S_script + 0.030 with 95% bootstrap CI strictly > 0.
2. T0: S_PPO ≥ S_script + 0.080; Stage-1 housing collisions < 4.0%; Stage-2 wedges < 2.0%.
3. T4xneg: S_PPO ≥ S_script + 0.050; Stage-1 housing collisions < 3.0%.
4. T2 (held-out): S_PPO ≥ S_script − 0.020 AND ≥ 0.840 overall.
5. Reliability: ≥ 7/8 seeds macro-mean ≥ 0.800; worst seed macro-mean ≥ 0.750.
6. Tray safety: mean cylinder spill ≤ 5.0% across the 5 tasks, max single-task spill ≤ 7.0%.

**Phase 2 OOD bars (one-look point-cloud PPO):**
1. Visual-to-state retention: S_visual ≥ 0.92 × S_state across T0–T4xneg.
2. Beats the direct oracle baseline without oracle targets: S_visual ≥ 0.800.

**Post-run decision branches:**
- **A** (Gate 0 + all of Phase 1 pass): freeze the state-input policy and weights, move to Phase 2 point-cloud input.
- **B** (T0/T4xneg pass, T2/T1 trails S_script by > 0.03 due to contact drift): switch the PPO action space to
  contact-anchor selection + clearance waypoint + push residual over the 4+3 leveling kinematics.
- **C** (Gate 0 passes, T0/T4xneg obstacle blocks > 5%): widen d_recess/d_overhang and raise w_h_force from 0.25 to 0.40.

### 13.3 Operational definitions (mine, fixed with the gates)
- **S_script** = `scripted_baseline.py` mode staged2 as frozen in e5cf7eb (§12.4), run by the evaluator on the
  same starts with the same instrumentation. Its v2 values: T0 0.710, T1 0.879, T2 0.840, T4x 0.907, T4xneg 0.873
  (macro 0.842).
- **Starts:** task t, seed s uses eval_policy.py's chain (reset seed 50,000 + s, then the env's own sequence),
  the same for PPO, S_script and every earlier eval. Policy seed s is evaluated on start set s, so episodes are
  paired with S_script's.
- **Deterministic:** the actor's mean action, clipped to [−1, 1]. Observation normalisation frozen at the run's
  saved obs_rms.npz (pooled), as in eval_policy.py.
- **Action std (Gate 0.1):** exp(actor_logstd) of the saved checkpoint. Every action dimension must be ≥ 0.065 (the minimum over dimensions).
- **Held-out validation scenes (Gate 0.2):** episode j of 1,000 uses reset seed 900,000 + j, the same 1,000
  scenes for every policy seed, in the F_geom validation env given to the evaluator. Mean and worst seed are
  over the 8 policy seeds' success rates.
- **Housing:** the fixture body's own geoms (cabinet carcass or chest), not the moving part, not the floor.
  F_house is the total normal force between arm geoms (links, plate) and housing geoms, averaged over each
  control step's 20 physics substeps, as the env averages its own contact forces.
- **Sustained housing collision (Gate 0.3, measured on the validation scenes):** an episode with ≥ 5
  consecutive control steps of F_house > 10 N. The rate is the share of episodes, pooled over seeds.
- **Stages for a policy** (it has no explicit stages): Stage 1 = control steps before the first forearm
  contact with the moving part (links 4–6, force > 0); Stage 2 = from that step on. Stage-1 housing collision =
  a sustained housing collision that starts in Stage 1. The script's stages (standoff-based, §12) are not used
  here, so PPO and S_script are measured the same way.
- **Stage-2 wedge:** a timeout that reached Stage 2, with forearm–moving part contact on ≥ 80% of the last 25
  control steps and < 0.02 normalised progress over them (the stage_audit.py rule).
- **Obstacle block (Branch C):** a timeout with sustained housing contact (F_house > 0) on ≥ 30% of the last 25
  steps, no forearm–moving part contact in them, and < 0.02 normalised progress over them.
- **Contact drift (Branch B):** on the trailing task, lost contact (a timeout that reached Stage 2 with forearm
  contact on ≤ 20% of the last 25 steps) is the largest failure category among timeouts and spills.
- **Macro-mean:** the unweighted mean over T0, T1, T2, T4x, T4xneg. Per seed, the mean of that seed's five
  task rates.
- **Bootstrap (bar 1.1):** 10,000 hierarchical resamples. Policy seeds are drawn with replacement, then
  episodes within each seed with replacement, paired with S_script on the same start. The statistic is the
  macro-mean of the per-episode differences (PPO − S_script). The CI is the 2.5–97.5 percentiles, and the bar needs both the point
  estimate ≥ +0.030 and the lower bound > 0. Per-task 95% CIs on S_PPO use the same resampling unpaired.
- **IQR:** 75th minus 25th percentile of the 8 per-seed rates.
- **Tray safety (bar 1.6):** spill = the env's spill termination. "Mean across the 5 tasks" = the mean of the
  five 8-seed mean spill rates.
- **Phase 2:** S_state is the Phase-1 policy's macro-mean from the saved Phase-1 summary, and S_visual the
  visual policy's macro-mean on the same starts. Bar 2.1: S_visual ≥ 0.92 × S_state. Bar 2.2: S_visual ≥ 0.800.
  For reference, the direct oracle script's v2 macro is 0.784 (§12, direct: T0 0.815, T1 0.864, T2 0.633,
  T4x 0.863, T4xneg 0.743).

### 13.4 Points to confirm (raised when writing, not resolved)
1. "Housing" is read as the cabinet or chest carcass. If it meant the arm's elbow motor housing (log.md
   2026-11-13), Gate 0.3 measures something else.
2. Bar 1.4: S_script on T2 is 0.840, so "≥ S_script − 0.020" (≥ 0.820) is always implied by "≥ 0.840". The binding condition is ≥ 0.840.
3. The branches don't cover every outcome. Gate 0 failing, or Phase 1 failing in ways that match neither B nor
   C, has no branch. Several branches can also hold at once. The evaluator prints every branch whose
   condition holds, or "none".
4. Gate 0.3 is measured on the F_geom validation scenes. Bars 1.2–1.3 measure Stage-1 housing collisions on
   T0 and T4xneg in the OOD set.
5. Pairing assumes the F_geom policies' observations can be built for T0–T4xneg by TaskEnv with the
   training env's kwargs, as eval_policy.py does now. If F_geom adds inputs TaskEnv cannot produce for those
   tasks, the evaluator stops with an error rather than guessing.

## 14. W1: GPU PPO in MuJoCo Warp, scored in C MuJoCo (2026-11-28; seeds 1–2 pending)

**Setup** (pre-registered in log.md, "Warp run W1", before launch). G3's task, observation (46), reward and
64-64 tanh policy, trained on family F in the batched MuJoCo Warp env (`wam/warp_env.py`, §12 physics on MuJoCo
4.0.1, reach distance approximated). Training settings:
- 4,096 worlds × 32 steps per update, 4 minibatches, 5 epochs;
- entropy coefficient 0.01, learning rate 3e-4 annealed;
- 200M steps per seed, seeds 1–3, about 33,000 steps/s on a shared RTX A6000.

Every reported number is evaluated in C MuJoCo 3.3.7 on frozen v2 (`eval_policy.py --exp w1`, mean action,
200 episodes per task, the usual starts).

**What differs from G3** (CPU, 20M steps): about 10× the samples, 4,096 parallel worlds, the exploration
settings, and the simulator used for training. The network, observation, reward and the 4+3 controller split
are unchanged.

**Results so far**

| Task | seed 3, final (200M) | seed 1 (95M, interim) | seed 3 (147M, interim) | staged2 (§12) | G3 (§10–12) |
|---|---|---|---|---|---|
| F (training family) | **0.950** | 0.935 | 0.955 | 0.761 | 0.31–0.74 (4 seeds) |
| T1 sliding door | **0.985** | 0.980 | 0.995 | 0.879 | — |
| T2 drawer, held out | **0.950** | 0.900 | 0.950 | 0.840 | 0.563 |
| T4x box | **0.995** | 0.985 | 1.000 | 0.907 | — |
| T4xneg box | **0.990** | 0.925 | 0.995 | 0.873 | — |
| T0 hinged door | 0.050 | 0.110 | 0.050 | 0.710 | — |

- **Seed 3 final:**
  - Spill ≤ 5% and timeout ≤ 1.5% on every sliding task.
  - T0: 0.485 spill, 0.465 timeout.
  - Warp training success over its last ~1M steps was 0.970, against 0.950 for F in C. The train/evaluate gap is 0.02.
- **Hinge:** family F has only slide joints, so T0 is outside the training distribution in kind. This is a
  scope limit, not a regression.
- **Action std:** it grew to 5.9 / 4.75 / 8.7 / 10.1 by the end of training. The entropy bonus inflates the
  noise around a clipped action at no cost to the policy. Evaluation uses the mean action, so the scores stand,
  but prediction W1-P1 (std ≥ 0.065) is met trivially and no longer measures healthy exploration. Later runs
  should use a squashed or capped policy.
- **Two crashes on the way:**
  - Seeds 1–2 hit NaN at 95M and 48M steps, from a diverged world's NaN observation. Non-finite worlds are now
    reset (e9de79d).
  - Their first resume failed on a GPU-only RNG-state bug (2cd4628).
  - They continue from clean checkpoints at iterations 725 and 350.

**Pending:** seeds 1–2 final, then the four pre-registered predictions scored across seeds (W1-P1 std,
W1-P2 F ≥ 0.80, W1-P3 T2 ≥ 0.70, W1-P4 Warp-vs-C within 0.10).

**What this shows so far (one final seed plus two interim checkpoints):**
- With scale and exploration fixed, the same RL setup that reached 0.563 on held-out T2 now reaches 0.90–0.95.
- It beats the training-free geometric controller staged2 on all five sliding tasks, held-out T2 included.

**What it does not show:**
- Hinge transfer.
- Behaviour on hardware.
- Whether the gain comes from scale, from exploration, or from both. Separating those needs the CPU exploration
  test (log.md), or a Warp run with G3's original settings.
