# Reward design for WamCloseCabinet-v0

Status: specification only. `CloseCabinetEnv.step` returns `0.0`; nothing here is implemented.

Every quantity below is already computed by the env and reported in `info`, except the
link-to-door distance $d$, which is one `mujoco.mj_geomDistance` call per contact capsule.

## Symbols

| Symbol | Meaning | Source | Range |
|---|---|---|---|
| $\theta_t$ | door angle, 0 = closed | `task/door_angle` | 0 to 1.75 rad; reset draws 1.0 to 1.4 |
| $\Delta_t = \theta_{t-1} - \theta_t$ | closing progress in one control step | | sweep reaches 0.02 rad/step |
| $F_\ell$ | normal force on the door from link group $\ell$, averaged over the 20 physics substeps | `force/door_*` | N |
| $F_A = F_4 + F_5 + F_6$ | force from the allowed set $A$ = links 4 to 6 (WACL Sec. 4.1) | | scripted sweep: 10 to 55 N while pushing |
| $F_W = F_{ee} + F_{upper} + F_{scene}$ | force from every wrong contact: plate or flange on the door, links 1 to 3 on the door, anything on carcass or floor | | N |
| $d_t$ | smallest distance from the collision capsules `link4_c*`, `link5_c*`, `link6_c*` to the door geom | `mj_geomDistance` | 0.13 to 0.21 m at reset (50 seeds) |
| $\alpha_t$ | plate tilt from level | `task/plate_tilt` | rad |
| $a_t$ | action | | $[-1, 1]^7$ |
| $\dot q_t$, $\tau_t$ | joint velocity, actuator torque (`qfrc_actuator`) | | $\tau_{lim}$ = 50, 50, 30, 30, 30, 20, 20 N m |

Constants: $F_{min} = 1$ N, $F_{max} = 60$ N, $\sigma = 0.1$ m, $\alpha_{free} = 10°$,
$\alpha_{max} = 25°$ (WACL Table 3 AxisCone), $\theta_s = 0.05$ rad.

## What the two reference papers do

**Pedipulate** (Arm, Mittal, Kolvenbach, Hutter; arXiv:2402.10837v1; read from `papers/`, Sec. III-B).
Total reward $R = R_e + R_n$.
Task: $R_e = w_1 \exp(-\lVert r_f - r_f^* \rVert / \sigma)$ with $w_1 = 15$, and $\sigma$ printed as "0.8 m⁻¹".
Penalties: $R_n = w_2 \lVert\dot q\rVert^2 + w_3 \lVert\ddot q\rVert^2 + w_4 \lVert\tau\rVert^2 + w_5 \lVert a_t - a_{t-1}\rVert^2 + w_6 n_c + w_7 k$
with $w_2 = -5\cdot10^{-2}$, $w_3 = -5\cdot10^{-6}$, $w_4 = -2\cdot10^{-5}$, $w_5 = -10^{-2}$, $w_6 = -2$ per colliding link pair, $w_7 = -80$ on termination.
The task is position tracking with no contact term at all; contact happens because the user steers the foot target into the object.

**WoCoCo** (Zhang, Xiao, He, Shi; arXiv:2406.06005v2, Sec. 3.1 and Appendices E to G).
These equations were extracted from the arXiv HTML by a fetch tool, not read from the PDF by hand. Spot-check before citing.
$r = r_{WoCoCo} + r_{reg} + r_{task}$, with $r_{WoCoCo} = w_{con} r_{con} + w_{stage} r_{stage} + w_{curi} r_{curi}$.
Contact: $r_{con} = n_{corr} - n_{con}\, n_{wrong}\, \mathbb{1}(n_{stage} > 0) + 2 n_{con}^2 F_{con} F_{task}$. The wrong-contact penalty is masked in the first stage to keep exploration alive.
Stage count: $r_{stage} = n_{stage} F_{task}$. The $F_{task}$ factor stops the policy from collecting stage reward while leaving the task goal unmet.
Curiosity: count-based, $r_{curi} = 1/\sqrt{\text{visits}}$ over a hashed observation, weights 5000 to 40000.
Regularisers include torque $\sum(\tau/\tau_{lim})^2$ at $-0.5$, action rate at $-250$, contact force above a cap $\sum\max(\lvert F\rvert - 550, 0)$ at $-0.005$, termination at $-200$.

**MuJoCo Playground, RSS 2025 Panda `open_cabinet.py`** (read in the repo at `rss2025/`).
Reach term $1 - \tanh(5\lVert\cdot\rVert)$ at scale 4.0, a task term at scale 8.0 multiplied by a latched `reached_box` flag, a no-collision term at 0.25 and a posture term at 0.3. The latched flag is precedent for gating task reward on a contact condition.

## What carries over, and what does not

| Source idea | Decision | Reason |
|---|---|---|
| Pedipulate exponential distance kernel | Keep the kernel, pay its **difference** | Pedipulate's goal is to hold the foot at the target, so paying the kernel every step is correct there. Here it would pay 250 steps of reward for hovering at the door. |
| WoCoCo $n_{corr}$ paid every step | Drop | A per-step tick for correct contact is the "rest on the door" exploit. Correct contact enters only as a gate on progress. |
| WoCoCo $F_{con} F_{task}$ conjunction | Keep, as the gate $g_t$ | Progress pays only while the contact goal holds. |
| WoCoCo wrong-contact penalty | Keep, unmasked | One stage only, and a wrong contact already earns nothing through the gate. Masking it during approach is ablation A3 if exploration stalls. |
| WoCoCo $r_{stage}$ | Keep, as the success bonus scaled by the validly earned fraction | Same purpose: the bonus cannot be collected by a route that skipped the contact goal. |
| WoCoCo curiosity | Not in v0 | One contact stage and a start 0.13 to 0.21 m from the door. Add it only if runs never reach first contact. |
| Pedipulate and WoCoCo regularisers | Keep action rate, joint velocity, normalised torque, force above a cap | Same roles. Weights rescaled to this action space; see the budget below. |
| Termination penalty | Only for the primary-task failure (plate spill) | See exploit 5. |

## Reward terms

Reward per control step (0.02 s), $r_t = \sum_i r_i$.

**Gate.** The contact goal holds when an allowed link carries the push and nothing else touches:

$$g_t = \mathbb{1}[F_A > F_{min}] \cdot \mathbb{1}[F_W = 0]$$

**1. Gated progress.**

$$r_{prog} = w_{prog}\left(g_t\,[\Delta_t]_+ - [-\Delta_t]_+\right), \qquad w_{prog} = 50\ \text{rad}^{-1}$$

Closing pays only under the gate. Reopening always costs the same rate, so open-then-close cycles sum to zero or less. Episode total is at most $50\,\theta_0$, so 50 to 70.

**2. Reach shaping.**

$$r_{reach} = w_{reach}\left(e^{-d_t/\sigma} - e^{-d_{t-1}/\sigma}\right), \qquad w_{reach} = 5,\ \sigma = 0.1\ \text{m}$$

The sum telescopes to $w_{reach}(e^{-d_T/\sigma} - e^{-d_0/\sigma})$, so it is bounded by 5 for the episode however long the arm stays near the door. With $d_0$ = 0.13 to 0.21 m the available total is 3.6 to 4.4.

**3. Wrong contact.**

$$r_{wrong} = -w_{wrong}\,\mathbb{1}[F_W > 0], \qquad w_{wrong} = 0.5\ \text{per step}$$

**4. Plate level (the primary task).**

$$r_{tilt} = -w_{tilt}\,[\alpha_t - \alpha_{free}]_+, \qquad w_{tilt} = 2\ \text{rad}^{-1}\ \text{per step}$$

At the 25° limit this is $-0.52$ per step.

**5. Force cap.**

$$r_{force} = -w_{force}\,\min\!\left(\frac{[F_A + F_W - F_{max}]_+}{F_{max}},\ 1\right), \qquad w_{force} = 0.5\ \text{per step}$$

$F_{max} = 60$ N sits above the 10 to 55 N the scripted sweep needs and below the 102 N it reaches when it keeps driving into the closed door.

**6. Regularisers.**

$$r_{reg} = -0.01\,\lVert a_t - a_{t-1}\rVert^2 \;-\; 0.001\,\lVert\dot q_t\rVert^2 \;-\; 0.01\sum_j\left(\tau_j/\tau_{lim,j}\right)^2$$

Worst cases per step: 0.28 (every joint flipping sign), 0.028 (every joint at 2 rad/s), 0.07 (every joint saturated).

**7. Success.** On the first step with $\theta_t < \theta_s$:

$$r_{succ} = B\,\rho, \qquad \rho = \frac{\sum_{k \le t} g_k [\Delta_k]_+}{\theta_0 - \theta_t} \in [0, 1], \qquad B = 20$$

$\rho$ is the share of the closing that happened under the gate. The env must carry the running sum $\sum g_k[\Delta_k]_+$; it resets with the episode.

**Terminations.**

| Condition | Type | Reward |
|---|---|---|
| $\theta_t < \theta_s$ | terminated (success) | $+B\rho$ |
| $\alpha_t > \alpha_{max}$ | terminated (spill) | $-15$ |
| 250 steps | truncated | none |

No alive bonus and no time penalty. Discounting ($\gamma = 0.99$) already favours closing sooner, and a negative per-step constant would make the spill termination a way out.

**Budget.** A clean episode earns about $4 + 60 + 20 = 84$. Regularisers on the scripted sweep cost 2.2 (measured). CleanRL's `NormalizeReward` wrapper divides by the running return standard deviation, so only the ratios between terms matter.

## Exploit audit

| # | Exploit | Why a policy finds it | What blocks it |
|---|---|---|---|
| 1 | Push with the plate, flange or wrist end instead of the forearm | The plate is the largest, most distal surface and reaches the door first under random exploration | $g_t = 0$ whenever $F_W > 0$, so that closing earns nothing. $r_{wrong}$ costs 0.5 per step of such contact. $\rho$ removes the matching share of the success bonus. A door closed entirely by the plate scores $0 - 0.5 n + 0$. |
| 2 | Pin or rest on the door without moving it | Any per-step contact reward pays 250 times per episode | There is no per-step contact reward. Contact appears only inside $g_t$, multiplied by $[\Delta_t]_+$, which is zero for a stationary door. Holding still costs torque and, above 60 N, $r_{force}$. |
| 3 | Vibrate against the door, or bat it and let it swing shut | Oscillation farms any reward tied to contact onset; a strike moves the door with one cheap action | Reopening is charged at the same 50 per rad as closing is paid, so an oscillation sums to at most zero. Progress on a step where the arm is not in contact has $g_t = 0$. Physically the door's velocity time constant is $I/b = 0.042/2 = 21$ ms, about one control step, so it cannot coast. Action rate penalises the oscillation itself. |
| 4 | Slam the door, or keep pressing after it hits its stop | Faster closing is earlier reward; the open-loop sweep itself ends at 102 N against the stop | $r_{force}$ charges up to 0.5 per step above 60 N. Success terminates the episode on the first step below $\theta_s$, so there is no reward left to press for. |
| 5 | Tilt the plate past 25° on purpose to end a bad episode | Termination stops every per-step penalty | No alive or time penalty exists, and every per-step penalty is zero out of contact with a level plate, so retreating always beats the $-15$. Residual risk: a policy wedged against the carcass, where $r_{wrong}$ and $r_{force}$ together reach 1.0 per step and outweigh the $-15$ after 15 steps. Watch the spill rate in the logs. |
| 6 | Hover at the door to collect reach reward | Dense distance kernels pay for proximity | $r_{reach}$ is a difference of a potential, so its episode total is fixed by the end points and bounded by 5. |

Not blocked by the reward: links 5 and 6 are wrist joints on the xArm 7 and are in WACL's allowed set. A policy that closes the door with the link 6 hub is valid under this spec. Ablation A1 narrows the set.

## Ablations that make the run diagnostic

| ID | Change | Question it answers |
|---|---|---|
| A0 | Ungated progress ($g_t \equiv 1$), no $r_{wrong}$ | How does unconstrained RL close the door? Expect the plate. This is the measured size of the cheating problem. |
| A1 | Allowed set = link 4 and link 5 only | Does true forearm contact emerge, or only wrist contact? |
| A2 | Append the six substep-averaged contact forces to the observation | Does explicit contact sensing change sample efficiency? The state is already Markov without them (see README), so a gain here is about learnability, not observability. |
| A3 | Mask $r_{wrong}$ until first valid contact, as WoCoCo does | Is the wrong-contact penalty suppressing exploration? |

## Reading the result

| Observation in the logs | Reading |
|---|---|
| `task/success` rises, `force/door_ee` stays 0 | RL works on the single-contact case. Move to randomised cabinet pose. |
| Return rises, `task/success` flat, `force/door_link*` zero at episode end | Reach shaping learned, contact never converted: exploration, or the gate is too strict. Run A3. |
| A0 succeeds with `force/door_ee` > 0, gated version fails | The whole-arm constraint is the hard part, not door closing. This is the result that argues for learning the contact trajectory and not the joint motion. |
| Success needs more than a few million steps on one fixed cabinet | Sample efficiency will not survive randomisation. Evidence for a prior (demonstrations or a pretrained policy). |
| Spill terminations dominate | The dual-task constraint is the bottleneck. Compare with $w_{tilt} = 0$. |

One run per condition says nothing about variance. Use at least 3 seeds per row before drawing a conclusion.
