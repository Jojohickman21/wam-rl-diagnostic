# Task set for testing generalisation (scope, 2026-11-18)

Status: built 2026-11-18 (`wam/tasks.py`, `assets/task_*.xml`, envs `WamTask<ID>-v0` and
`WamTask<ID>Outcome-v0`). Decisions (Josiah, 11/18): T1 gets a vertical handle bar; attempt T3, and
if it fails the feasibility gate, add pull goals (negative positions) to T4; condition with the
constraint descriptor (option b); per-link contact forces are acceptable in the training reward;
run everything on the lab server (snoopy1).

Why (Ray, 11/18; notes.md): the reward and approach so far were tuned on one task, and §7.2's edge
straddle shows how a reward can fit one task's geometry. A set of tasks tests whether the reward
fits the task or the method. The second goal is inference-time generalisation: one policy
conditioned on the articulation or constraint it faces. Ray named sliding doors, push and pull,
and objects with different x, y constraints. He ruled out automated reward search for now.

## What carries over from RESULTS.md §1–8

- **Structure:** a controller owns the primary end-effector constraint and RL owns the push (A20,
  §7). It was the most consistent result, and it is what the rest of the set should test.
- **Observations:** hardware-measurable only (joint encoders, object articulation state, object
  pose). Per-link contact forces and friction stay out of observations.
- **Reward:** two candidates, because §8 did not settle which generalises. Dense gated (A20's)
  learned better on the original task (§8.3). Outcome-only (A26) removed the straddle signature
  but learned worse. The task set is the right place to choose between them.
- **Statistics:** 3 seeds minimum, and a difference counts only if seed ranges do not overlap
  (noise floor about 4 points of success, §7.3).
- **Engineering:** speed-test under the trainer's 8 async workers before any launch (§8.1), and
  save normalisation statistics (done).
- **Harder starts are a separate axis:** §8's ±1.0 rad starts raised timeouts to 31–78% on every
  reward. Add hard initialisation per task only after its easy version learns, or the two effects
  are confounded.

## The tasks

Every task keeps the plate held by the A20 leveler. Each has one coordinate s that measures the
object's progress, a goal s*, and a range used to normalise reward across tasks.

| ID | Task | Articulation (MuJoCo) | s, s*, range | Contact that moves it | Main risk |
|---|---|---|---|---|---|
| T0 | Close hinged door (existing) | hinge, vertical axis | angle, 0, 1.0–1.4 rad | forearm on the door face | none; this is the reference |
| T1 | Close sliding door | slide, horizontal, in the door plane | lateral offset, 0, 0.25–0.40 m | forearm on the panel's leading edge; a push on the face only gives friction along the slide | the edge is a thin target; may need a vertical handle bar to push on |
| T2 | Push drawer shut | slide, horizontal, into the cabinet | extension, 0, 0.15–0.30 m | forearm on the drawer front, normal aligned with motion | probably easy; the low-difficulty member |
| T3 | Pull drawer open | same as T2 | extension, 0.25 m, start 0 | forearm or elbow hooked behind a handle; plain contact cannot pull | highest. Whole-arm pulling needs a hook geometry; may be infeasible without it |
| T4x | Push box along an x rail | slide x, friction | position, goal 0.20–0.30 m away | forearm on the box side | low |
| T4y | Push box along a y rail | slide y | as T4x | as T4x | low |
| T4xy | Push box in the plane | slide x + slide y (no rotation) | distance to goal | as T4x; contact point decides the direction | the push direction has to be steered, which the others do not need |

T4x, T4y and T4xy are the "objects with different constraints". The same box under different
joint constraints isolates the constraint from the geometry.

**Feasibility gate before any training**, as for A20 (§7): a scripted forearm motion must solve at
least 30% of starts, and a scripted hand motion must show whether the hand is the easier tool.
A task that fails is dropped or redesigned before it costs compute. T3 is the likely casualty.

## One reward template for every task

    r_prog = w · (|sₜ₋₁ − s*| − |sₜ − s*|) / range

Reopening is charged at the same rate, as now. Success when |sₜ − s*| < tolerance. The success
bonus is scaled by the valid-contact share. Spill, energy and force cap are unchanged. Normalising
by range gives every task the same progress budget, so no task dominates a shared policy's return.

Two variants, run across the whole set:
- **Dense gated:** progress pays only under valid contact, plus reach shaping to the moving body.
- **Outcome:** progress pays any closing; valid contact is judged at the end (A26's design).

The gate uses per-link contact forces, which a real xArm cannot measure. That is acceptable in a
training reward only if the deployed policy never needs them; decide this before building (open
question 3).

## Conditioning: how one policy knows which task it faces

| Option | Observation added | Generalises to | Cost |
|---|---|---|---|
| a. Task one-hot | a vector over T0–T4 | nothing unseen | trivial |
| b. Constraint descriptor | joint type (hinge or slide), axis (world unit vector), anchor point, s, s*, range | unseen axes and placements of known joint types | a perception step on hardware, like the cabinet pose now |
| c. Inferred from history | none; recent joint and object motion (RMA-style adaptation) | unknown articulations | harder to train; a later step |

Recommended: **(b)**, with a held-out test. Train on T0–T2 and T4x/T4y at some axes; test on
unseen axis orientations and on T4xy. That turns "generalisation at inference time" into one
number. Ray's teacher-student point fits here: a teacher with full object state, distilled into a
student that sees only the descriptor and encoders.

The primary-task constraint can be conditioned the same way, using WACL's vocabulary: an AxisCone
(plate or cup upright) or a PositionRegion (TCP inside a box), given to the controller as
parameters. That is the second axis of "different constraints", and it maps onto the base paper's
representation (WACL Table 3, App. A.3: AxisCone, PositionRegion).

Contact conditioning ("different whole-arm contacts") would make the allowed link set an input,
for example a mask over links 4, 5 and 6. It depends on the per-link gate, so it is tied to open
question 3.

## Staging and cost

Laptop: about 2,000 steps/s, so 2M steps take about 17 minutes. One run at a time.

| Stage | Work | Runs | Laptop time |
|---|---|---|---|
| 0 | Refactor the env from "door" to "articulated object with coordinate s" (29 door-specific references). T0 must reset and step bit-identically to now; check it the way §8.1 did. | 0 | build only |
| 1 | Build T1, T2, T4x, T4y, T4xy (and T3 if attempted); feasibility gate per task | 0 | build only |
| 2 | Single-task baselines, A20 structure, dense gated, 3 seeds each | 15–18 | 4.5–5 h |
| 3 | Outcome reward on the same tasks | 15–18 | 4.5–5 h |
| 4 | One conditioned multi-task policy (option b), held-out test | 3+ | likely 10M+ steps each, about 1.5 h per run; needs lab compute |

Stages 0–2 fit before the Erdem meeting (Wed 11/25). Stage 4 waits on the USC compute access.

## Open questions for Josiah

1. **T3 (pulling):** attempt it, accepting it may fail the feasibility gate, or leave pulling for
   later? It is the task most unlike pushing and the most informative if it works.
2. **Conditioning:** option (b) first, or start with (a) to get a multi-task baseline cheaply?
3. **The gate on hardware:** keep per-link contact in the training reward (sim-only, never in the
   policy's input), or only use the outcome reward from here on? This decides whether contact
   conditioning is possible.
4. **Order:** the full Stage 2 before the Erdem meeting, or T1 and T4x only, to bring a
   two-task result sooner?

## Build notes and feasibility gate (2026-11-18)

- **T0 reproduces A20's dynamics and rewards exactly** (not its observation: 59 dimensions with the
  descriptor against A20's 35; see RESULTS.md §9.1). Same seed and actions give identical joint states, rewards and
  terminations over 2,400 random steps (test `test_task_t0_reproduces_a20_step_for_step`).
- **Movers must not coast.** The first build gave the slide joints light damping (velocity time
  constant about 0.4 s), so a single hit glided the object most of the way and that motion counted
  as invalid closing. Damping is now about mass / 0.02 s, as for the hinged door (REWARD_DESIGN.md
  exploit 3): 150 N·s/m for the panel and drawer, 100 for the box.
- **Gate** (`scripts/task_feasibility.py <task> 30`): a scripted forearm push, tried with four
  forearm points (elbow end to wrist end), the wrist leveler holding the plate. A start counts as
  solved if any point closes it with valid fraction > 0.9.

| Task | Solved forearm-only | Gate |
|---|---|---|
| T1 sliding door (handle bar) | 93% | pass |
| T2 drawer push | 80% | pass |
| T4x box, x rail | 100% | pass |
| T3 drawer pull | 0% (never got the forearm into the handle gap) | fail |
| T4y box, y rail (motion at azimuth −30°) | 23% | fail |
| T4xy box, planar | 7% | fail |

The scripted controller moves one forearm point and cannot set the forearm's orientation, so a
failed gate is a lower bound, not proof of infeasibility.

**T3 (pull) failed three scripted designs**, so pulling was dropped per the decisions above:
1. Position-only, coming down into the handle gap from above: 0%. The forearm never entered the gap.
2. With an added term turning the forearm parallel to the handle (`scripts/t3_hook_feasibility.py`):
   0%. The alignment reached 4–19°, but the forearm point stayed 10–31 cm from the hook pose
   with joints 2 and 4 far from their stops. The pose is out of reach: laying the forearm across a
   handle 0.6 m from the base needs more reach than joints 1–4 have while the wrist is levelling.
3. The chest moved so the handle sits about 0.4 m out: aligned above the gap on up to 25% of starts,
   into the gap on none.
Reading: with this arm, a levelled wrist and 4 learned joints, the forearm pushes objects along
its sweep, but getting behind an object and drawing it toward the body is not shown to be
feasible. Reaching behind the T4 box from above to draw it back failed the same way (0%).

**T4 redesign.** The pull fallback is implemented as negative goals on the x rail with the arm
starting on the box's far side (`home_yaw` 0.9 rad added to joint 1): T4xneg. To be explicit: that
is the forearm pushing from the other side, not drawing toward the body. It puts goals of both
signs on one constraint, which is what the conditioning test needs. T4y (a radial rail) failed in
every orientation tried (at most 23%, or no valid start at all). T4xy now draws its goal within ±30°
of the rail's x axis instead of across 90°: the box is free in the plane, but the goal stays
pushable.

| Task | Solved forearm-only | Gate |
|---|---|---|
| T4xneg (x rail, negative goal, far-side start) | 67% | pass |
| T4xy (planar, goal within ±30° of x) | 33% | pass, marginal |
| T4y | ≤23% | fail, not trained |

**Start sampling could hang.** For 2% of T4xneg's fixture poses no valid far-side start exists, and
the rejection loop never ended (found by a multi-task test that hung). After 1,000 arm draws the
pose and goal are now redrawn. Every other task needs at most a few draws, so their starts, and
T0's match with A20, are unchanged (tested).

## What runs on snoopy1 (2026-11-18)

| Stage | Envs | Seeds | Steps | Notes |
|---|---|---|---|---|
| 2 + 3 | T0, T1, T2, T4x; dense gated and outcome | 3 | 2M | array 79719, started 18:24 |
| 2 + 3 | T4xneg, T4xy; dense gated and outcome | 3 | 2M | starts when 79719 finishes |
| 4 | `WamMultiTask-v0`: one policy over T0, T1, T4x, T4xneg, dense gated | 3 | 10M | T2 held out; evaluate with `scripts/eval_policy.py`, including fixture yaw ±40° |

At most 6 runs at once (60 CPUs), under the lab's half-server rule. About 30 minutes per 2M-step
run on snoopy1 (about 1,000–1,200 steps/s per run, slower per run than the laptop's 2,000).

## Follow-up runs (2026-11-21): more seeds and the task-label baseline

RESULTS.md §9.2 could not separate the descriptor-conditioned policy from single-task controls on
held-out T2 with 3 seeds. Follow-up, on snoopy1, same trainer and hyperparameters:

| Env | What it is | New seeds | Steps |
|---|---|---|---|
| `WamMultiTask-v0` | descriptor conditioning (option b), T0/T1/T4x/T4xneg | 4–8 (8 total) | 10M |
| `WamMultiTaskLabel-v0` | **task-label baseline (option a)**: same tasks, each slot keeps [valid, s, ṡ, s*, range] but drops joint type, axis and anchor; a one-hot over the 4 training tasks is appended (observation 47). On held-out T2 the one-hot is all zeros. | 1–8 | 10M |
| `WamTaskT1-v0`, `WamTaskT4x-v0` | single-task controls for the held-out test | 4–8 (8 total) | 2M |
| `WamTaskT2-v0` | T2 ceiling | 4–8 (8 total) | 2M |

The label baseline isolates what the joint geometry adds over a task ID: both policies see the
object's position, velocity, goal and range. On T2, the descriptor policy sees a new joint
geometry, while the label policy sees no task identity at all. Comparison bar unchanged: a
difference counts only if seed ranges do not overlap. With 8 seeds that bar is stricter than with
3, so a rank-sum test across seeds is reported alongside it, as a second look, not a replacement.

**Outcome (2026-11-23, RESULTS.md §10):** all 8 seeds of every policy completed after preempted
runs were rerun. On held-out T2 no policy separates (descriptor 0.349, label 0.422, single-task
controls 0.27 and 0.26, every range overlapping, smallest p 0.068). In distribution the descriptor
and label policies are indistinguishable on all four tasks. The joint geometry adds nothing
measurable over a task label in this setup. Next informative test: a held-out task interpolated
between training tasks.

## Generalisation batch G1–G3 (2026-11-25, after the Erdem meeting)

Why: §10 showed that conditioning on four discrete tasks does not transfer to a held-out task. Likely
reasons: too little variety, an input that makes the policy do its own kinematics, and interference
between discrete tasks. Erdem endorsed three changes (notes.md, 11/25): contact targets as
observations, procedurally randomised articulations, and forward-kinematics features.

**Training family "F"** (`wam/tasks.py`): the T4 box on one slide rail, redrawn every episode.
- **Push direction:** azimuth drawn from −10°…110°, or 210°…250° with the far-side start.
- **Held-out band:** 10°…50° is never drawn in training.
- **Also redrawn:** position (±10 cm), box half-sizes (4–7 cm wide, 10–14 cm tall), goal 0.20–0.30 m ahead.

The arcs are where a scripted forearm push solves at least 60% of starts
(`scripts/family_feasibility.py`, 20 starts per 20° bin):

| Push direction | Solved forearm-only |
|---|---|
| −30° to −10° | 25% (excluded) |
| −10° to 110° | 80–100% |
| 210° to 250° (far-side start) | 60–65% |
| 250° to 270° | 20% (excluded) |

**Variants** (same family, same reward, A20's controller + RL structure; 8 seeds each, 20M steps):

| Label | Env | The policy is told |
|---|---|---|
| G1 | `WamFamilyDesc-v0` | joint descriptor (as in §9–10): type, world axis and anchor, s, ṡ, goal, range, plus fixture pose (obs 59) |
| G2 | `WamFamilyContact-v0` | contact target: (target surface point − forearm point), push direction, distance to goal / range, its rate (obs 37). No fixture pose or joint parameters |
| G3 | `WamFamilyContactFK-v0` | G2 plus Cartesian positions of the elbow (link 4), forearm (link5_c2 centre) and wrist (link 6), from forward kinematics of the encoders (obs 46) |

The target surface point is where a ray along the push direction first meets the object. It is the
same rule for every task: the handle bar on T1, the drawer front on T2, the box face on T4. The FK
positions are *added* to the joint angles, not substituted for them, since the joint angles are
still needed for control and joint limits. Erdem suggested replacing them; that is a possible
follow-up.

**Test tasks (zero-shot, `scripts/eval_policy.py`, 200 episodes per seed):**

| Task | Role |
|---|---|
| T4x, T4xneg | the family's own box: an in-distribution check |
| T1 | new object (panel and handle bar), mostly trained directions |
| **T2** | **new object (drawer and chest) and held-out push directions: the main test** |

**Bar, fixed before running:**
1. **Validity:** G2 and G3 reach at least 0.80 mean on T4x and T4xneg. If not, the family didn't train, and T2 numbers are not interpretable.
2. **Main claim:** G2 or G3 reaches at least **0.70 mean on held-out T2**, with a seed range that does not overlap the §10 label policy's (0.22–0.56). The T2-trained ceiling is 0.838.
3. **Between variants:** G2 vs G1 and G3 vs G2 on T2 count only if seed ranges do not overlap. An exact rank-sum p is reported as a second look.
