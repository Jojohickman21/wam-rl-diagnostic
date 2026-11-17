# Reward design for WamCloseCabinet-v0

Status: specification only. `CloseCabinetEnv.step` returns `0.0`; nothing here is implemented.

Every quantity below is already computed by the env and reported in `info`, except the
link-to-door distance d, which is one `mujoco.mj_geomDistance` call per contact capsule.

## Symbols

| Symbol | Meaning | Source | Range |
|---|---|---|---|
| θₜ | door angle, 0 = closed | `task/door_angle` | 0 to 1.75 rad; reset draws 1.0 to 1.4 |
| Δₜ = θₜ₋₁ - θₜ | closing progress in one control step | | sweep reaches 0.02 rad/step |
| F_ℓ | normal force on the door from link group ℓ, averaged over the 20 physics substeps | `force/door_*` | N |
| F_A = F₄ + F₅ + F₆ | force from the allowed set A = links 4 to 6 (WACL Sec. 4.1) | | scripted sweep: 10 to 55 N while pushing |
| F_W = F_ee + F_upper + F_scene | force from every wrong contact: plate or flange on the door, links 1 to 3 on the door, anything on carcass or floor | | N |
| dₜ | smallest distance from the collision capsules `link4_c*`, `link5_c*`, `link6_c*` to the door geom | `mj_geomDistance` | 0.13 to 0.21 m at reset (50 seeds) |
| αₜ | plate tilt from level | `task/plate_tilt` | rad |
| aₜ | action | | [−1, 1]⁷ |
| q̇ₜ, τₜ | joint velocity, actuator torque (`qfrc_actuator`) | | τ_lim = 50, 50, 30, 30, 30, 20, 20 N m |

Constants: F_min = 1 N, F_max = 60 N, σ = 0.1 m, α_free = 10°,
α_max = 25° (WACL Table 3 AxisCone), θ_s = 0.05 rad.

## What the two reference papers do

**Pedipulate** (Arm, Mittal, Kolvenbach, Hutter; arXiv:2402.10837v1; read from `papers/`, Sec. III-B).
Total reward R = R_e + R_n.
Task: R_e = w₁ exp(-‖r_f − r_f*‖ / σ) with w₁ = 15, and σ printed as "0.8 m⁻¹".
Penalties: R_n = w₂ ‖q̇‖² + w₃ ‖q̈‖² + w₄ ‖τ‖² + w₅ ‖aₜ - aₜ₋₁‖² + w₆ n_c + w₇ k
with w₂ = −5·10⁻², w₃ = −5·10⁻⁶, w₄ = −2·10⁻⁵, w₅ = −10⁻², w₆ = −2 per colliding link pair, w₇ = −80 on termination.
The task is position tracking with no contact term at all; contact happens because the user steers the foot target into the object.

**WoCoCo** (Zhang, Xiao, He, Shi; arXiv:2406.06005v2, Sec. 3.1 and Appendices E to G).
These equations were extracted from the arXiv HTML by a fetch tool, not read from the PDF by hand. Spot-check before citing.
r = r_WoCoCo + r_reg + r_task, with r_WoCoCo = w_con r_con + w_stage r_stage + w_curi r_curi.
Contact: r_con = n_corr - n_con · n_wrong · 𝟙(n_stage > 0) + 2 n_con² F_con F_task. The wrong-contact penalty is masked in the first stage to keep exploration alive.
Stage count: r_stage = n_stage F_task. The F_task factor stops the policy from collecting stage reward while leaving the task goal unmet.
Curiosity: count-based, r_curi = 1/√visits over a hashed observation, weights 5000 to 40000.
Regularisers include torque Σ(τ/τ_lim)² at −0.5, action rate at −250, contact force above a cap Σmax(|F| - 550, 0) at −0.005, termination at −200.

**MuJoCo Playground, RSS 2025 Panda `open_cabinet.py`** (read in the repo at `rss2025/`).
Reach term 1 - tanh(5‖·‖) at scale 4.0, a task term at scale 8.0 multiplied by a latched `reached_box` flag, a no-collision term at 0.25 and a posture term at 0.3. The latched flag is precedent for gating task reward on a contact condition.

## What carries over, and what does not

| Source idea | Decision | Reason |
|---|---|---|
| Pedipulate exponential distance kernel | Keep the kernel, pay its **difference** | Pedipulate's goal is to hold the foot at the target, so paying the kernel every step is correct there. Here it would pay 250 steps of reward for hovering at the door. |
| WoCoCo n_corr paid every step | Drop | A per-step tick for correct contact is the "rest on the door" exploit. Correct contact enters only as a gate on progress. |
| WoCoCo F_con F_task conjunction | Keep, as the gate gₜ | Progress pays only while the contact goal holds. |
| WoCoCo wrong-contact penalty | Keep, unmasked | One stage only, and a wrong contact already earns nothing through the gate. Masking it during approach is ablation A3 if exploration stalls. |
| WoCoCo r_stage | Keep, as the success bonus scaled by the validly earned fraction | Same purpose: the bonus cannot be collected by a route that skipped the contact goal. |
| WoCoCo curiosity | Not in v0 | One contact stage and a start 0.13 to 0.21 m from the door. Add it only if runs never reach first contact. |
| Pedipulate and WoCoCo regularisers | Keep action rate, joint velocity, normalised torque, force above a cap | Same roles. Weights rescaled to this action space; see the budget below. |
| Termination penalty | Only for the primary-task failure (plate spill) | See exploit 5. |

## Reward terms

Reward per control step (0.02 s), rₜ = Σ_i r_i.

**Gate.** The contact goal holds when an allowed link carries the push and nothing else touches:

    gₜ = 𝟙[F_A > F_min] · 𝟙[F_W = 0]

**1. Gated progress.**

    r_prog = w_prog(gₜ [Δₜ]₊ - [−Δₜ]₊),  w_prog = 50 rad⁻¹

Closing pays only under the gate. Reopening always costs the same rate, so open-then-close cycles sum to zero or less. Episode total is at most 50 θ₀, so 50 to 70.

**2. Reach shaping.**

    r_reach = w_reach(exp(−dₜ/σ) - exp(−dₜ₋₁/σ)),  w_reach = 5, σ = 0.1 m

The sum telescopes to w_reach(exp(−d_T/σ) - exp(−d₀/σ)), so it is bounded by 5 for the episode however long the arm stays near the door. With d₀ = 0.13 to 0.21 m the available total is 3.6 to 4.4.

**3. Wrong contact.**

    r_wrong = −w_wrong 𝟙[F_W > 0],  w_wrong = 0.5 per step

**4. Plate level (the primary task).**

    r_tilt = −w_tilt [αₜ - α_free]₊,  w_tilt = 2 rad⁻¹ per step

At the 25° limit this is −0.52 per step.

**5. Force cap.**

    r_force = −w_force min([F_A + F_W - F_max]₊ / F_max, 1),  w_force = 0.5 per step

F_max = 60 N sits above the 10 to 55 N the scripted sweep needs and below the 102 N it reaches when it keeps driving into the closed door.

**6. Regularisers.**

    r_reg = −0.01 ‖aₜ - aₜ₋₁‖²  -  0.001 ‖q̇ₜ‖²  -  0.01 Σ_j(τ_j/τ_lim,j)²

Worst cases per step: 0.28 (every joint flipping sign), 0.028 (every joint at 2 rad/s), 0.07 (every joint saturated).

**7. Success.** On the first step with θₜ < θ_s:

    r_succ = B ρ,  ρ = (Σ(k≤t) gₖ [Δₖ]₊) / (θ₀ - θₜ) ∈ [0, 1],  B = 20

ρ is the share of the closing that happened under the gate. The env must carry the running sum Σ gₖ[Δₖ]₊; it resets with the episode.

**Terminations.**

| Condition | Type | Reward |
|---|---|---|
| θₜ < θ_s | terminated (success) | +Bρ |
| αₜ > α_max | terminated (spill) | −15 |
| 250 steps | truncated | none |

No alive bonus and no time penalty. Discounting (γ = 0.99) already favours closing sooner, and a negative per-step constant would make the spill termination a way out.

**Budget.** A clean episode earns about 4 + 60 + 20 = 84. Regularisers on the scripted sweep cost 2.2 (measured). CleanRL's `NormalizeReward` wrapper divides by the running return standard deviation, so only the ratios between terms matter.

## Exploit audit

| # | Exploit | Why a policy finds it | What blocks it |
|---|---|---|---|
| 1 | Push with the plate, flange or wrist end instead of the forearm | The plate is the largest, most distal surface and reaches the door first under random exploration | gₜ = 0 whenever F_W > 0, so that closing earns nothing. r_wrong costs 0.5 per step of such contact. ρ removes the matching share of the success bonus. A door closed entirely by the plate scores 0 - 0.5 n + 0. |
| 2 | Pin or rest on the door without moving it | Any per-step contact reward pays 250 times per episode | There is no per-step contact reward. Contact appears only inside gₜ, multiplied by [Δₜ]₊, which is zero for a stationary door. Holding still costs torque and, above 60 N, r_force. |
| 3 | Vibrate against the door, or bat it and let it swing shut | Oscillation farms any reward tied to contact onset; a strike moves the door with one cheap action | Reopening is charged at the same 50 per rad as closing is paid, so an oscillation sums to at most zero. Progress on a step where the arm is not in contact has gₜ = 0. Physically the door's velocity time constant is I/b = 0.042/2 = 21 ms, about one control step, so it cannot coast. Action rate penalises the oscillation itself. |
| 4 | Slam the door, or keep pressing after it hits its stop | Faster closing is earlier reward; the open-loop sweep itself ends at 102 N against the stop | r_force charges up to 0.5 per step above 60 N. Success terminates the episode on the first step below θ_s, so there is no reward left to press for. |
| 5 | Tilt the plate past 25° on purpose to end a bad episode | Termination stops every per-step penalty | No alive or time penalty exists, and every per-step penalty is zero out of contact with a level plate, so retreating always beats the −15. Residual risk: a policy wedged against the carcass, where r_wrong and r_force together reach 1.0 per step and outweigh the −15 after 15 steps. Watch the spill rate in the logs. |
| 6 | Hover at the door to collect reach reward | Dense distance kernels pay for proximity | r_reach is a difference of a potential, so its episode total is fixed by the end points and bounded by 5. |

Not blocked by the reward: links 5 and 6 are wrist joints on the xArm 7 and are in WACL's allowed set. A policy that closes the door with the link 6 hub is valid under this spec. Ablation A1 narrows the set.

## Ablations that make the run diagnostic

| ID | Change | Question it answers |
|---|---|---|
| A0 | Ungated progress (gₜ ≡ 1), no r_wrong | How does unconstrained RL close the door? Expect the plate. This is the measured size of the cheating problem. |
| A1 | Allowed set = link 4 and link 5 only | Does true forearm contact emerge, or only wrist contact? |
| A2 | Append the six substep-averaged contact forces to the observation | Does explicit contact sensing change sample efficiency? The state is already Markov without them (see README), so a gain here is about learnability, not observability. |
| A3 | Mask r_wrong until first valid contact, as WoCoCo does | Is the wrong-contact penalty suppressing exploration? |
| A4 | Gate progress on the plate being level: gₜ also requires αₜ < α_free | Does coupling the two tasks in the reward cut spills without stalling? |
| A5 | Quadratic tilt barrier: r_tilt = −1 · ([αₜ − α_free]₊ / (α_max − α_free))² per step, in place of the linear term | Does a gradient that steepens toward the limit warn the value function in time? |
| A6 | A4 + A5 | The two together. Env `WamCloseCabinetRandomPoseCoupled-v0`; A5 alone is `...RandomPoseBarrier-v0`. |
| A7 | A6, with the spill limit α_max drawn from 15° to 35° each episode and appended to the observation; the barrier scales to it | Does the policy learn to keep margin from whatever limit it is given, instead of fitting one threshold? Read success and spill binned by α_max. Env `WamCloseCabinetRandomPoseCoupledLimit-v0`. |

## Reading the result

| Observation in the logs | Reading |
|---|---|
| `task/success` rises, `force/door_ee` stays 0 | RL works on the single-contact case. Move to randomised cabinet pose. |
| Return rises, `task/success` flat, `force/door_link*` zero at episode end | Reach shaping learned, contact never converted: exploration, or the gate is too strict. Run A3. |
| A0 succeeds with `force/door_ee` > 0, gated version fails | The whole-arm constraint is the hard part, not door closing. This is the result that argues for learning the contact trajectory and not the joint motion. |
| Success needs more than a few million steps on one fixed cabinet | Sample efficiency will not survive randomisation. Evidence for a prior (demonstrations or a pretrained policy). |
| Spill terminations dominate | The dual-task constraint is the bottleneck. Compare with w_tilt = 0. |

One run per condition says nothing about variance. Use at least 3 seeds per row before drawing a conclusion.
