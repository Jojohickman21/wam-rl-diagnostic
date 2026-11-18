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
