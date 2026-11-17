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
