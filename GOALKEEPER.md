# Goalkeeper policy (Phase 1) — distillation → repair-oracle → RL polish → mixture-of-experts

Our Phase-1 goalkeeper improves on the reference Humanoid-Goalkeeper checkpoint
by a multi-stage pipeline — distillation, a per-scenario repair oracle, RL polish,
and finally a region mixture-of-experts — with **no eval-seed gaming, no
unrealistic robot, and no deploying the reference weights as-is**.

## Benchmark (the single source of truth)

**`diagnose_gk.py --seed 2810 --num-envs 256 --batches 16`** (= 4096 trials, fixed
seed 2810). This is the ONLY number we compare. Reproducible to ±0.1% on a given
GPU; aligns to ±0.3% across GPUs (3090↔4090, FP-only). The grader's 50-trial eval
has ±6% binomial noise — never compare on 50 trials.

| version | policy | block rate (seed 2810, 4096) |
|---|---|---|
| — | zero-agent baseline | ~33% |
| — | reference `goalkeeper.pt` (teacher) | 72.5% |
| — | distilled student `goalkeeper_distilled_v3.pt` | 65% |
| v1 | `goalkeeper_polished.pt` (RL polish) | 77.6% |
| v2 | `goalkeeper_polished_v2.pt` (best single MLP) | 80.5% |
| v3 | `goalkeeper_moe_v3.pt` (mixture-of-experts) | 83.1% |
| v5 | `goalkeeper_moe_v5.pt` (MoE) | 83.6% |
| v6 | `goalkeeper_moe_v6.pt` (MoE + sharp-contact Mid) | 84.0% |
| v9 | `goalkeeper_moe_v9.pt` (3-way MoE) | 84.4% |
| moe6 | `goalkeeper_moe6.pt` (6-way MoE + descent gate, latch 3.5, no mirror) | 86.0% |
| **moe6+** | **`goalkeeper_moe6.pt` (+ early latch 5.0 + L/R mirror of the Up expert)** | **91.0%** |

Eval the best: `python scripts/eval_moe6.py --moe6 src/assets/soccer/weight/goalkeeper_moe6.pt --seed 2810 --num-envs 256 --batches 16` (the bundle now carries `latch_hi`, `mirror_map`, and the z/vz gate config, so no flags are needed). Generalises off-benchmark: seeds 4567/99 → 90.5% / 89.7% (mean ≈ 90.5%).
**6-way MoE** = one specialist per region (`train_polish.py --train-regions <r>`; Up specialists use the crossing-instant reward `--w-cross`), routed by a ballistic gate on crossing height + side + descent rate (low-landing balls descend steeply → routed Low even when they cross higher).

> **Where the weights live.** The deployable policies are committed under
> `src/assets/soccer/weight/` (`goalkeeper_moe6.pt` = this 91% deliverable;
> `goalkeeper_polished_v2.pt`, `goalkeeper_distilled_v3.pt`, reference `goalkeeper.pt`).
> Any `logs/…pt` paths in script *defaults* (`train_polish.py --init`, `eval_moe.py`, …)
> are training-box scratch — `logs/` is git-ignored and is **not** needed to evaluate
> or deploy. To run anything, point at the `src/assets/soccer/weight/` paths above.

**The 86 → 91 jump came from two eval-time fixes, no retraining.** A per-region diagnostic (gate accuracy vs true region; block rate of correctly-routed vs misrouted balls) showed the 86→89.8 gap was *not* routing accuracy (a flat 77.7% — most misroutes are same-side-adjacent and still saved) but two other things:

1. **Routing *timing*, not accuracy (+4%).** The gate only latched once the ball was close (`bx < 3.5`); the first frames of *every* ball were handled by the default expert 0 (Right-Mid, a *right*-diver), which commits left balls the wrong way before the left expert ever engages — fatal for the left side (Left-Low: 97.4% with the right expert from frame 0 vs 86.4% when the right expert handled the first frames). Latching as soon as the ball is clearly approaching (`bx < latch_hi = 5.0`, a broad plateau 5.0–7.0) lets the correct specialist drive the whole dive. This *recovers and slightly exceeds* the perfect-gate ceiling (the crossing-height heuristic even routes some hard balls to better-suited experts than their launch region would): **86.0% → 90.0%**, and it generalises across seeds (+3.3 to +4.1%).

2. **Bilateral symmetry on the weak side (+1%).** The teacher has a consistent left-bias, so the Right-Up specialist (79.8%) badly trailed Left-Up (87.9%) even under a perfect gate. The task is exactly L/R symmetric (`modules/symmetry.py`, FK-validated to 0.01 mm), so we route Right-Up balls through the *mirrored Left-Up* expert — `π_RUp(o) = mirror_action(sr3d(mirror_obs(o)))` (`mirror_map="2:3"`). Right-Up correctly-routed 79.2% → 85.3%, overall **90.0% → 91.0%**. Mirroring the already-strong Mid/Low right experts does *not* help (they are trained on right balls; a small sim asymmetry makes the mirror a wash), so only the Up expert is mirrored.

**Remaining ceiling.** With both fixes the floor is the genuinely hard high balls: Right-Up ≈ 85% (mirrored), the Mid regions ≈ 88%. These are *timing*-limited near-misses (a reactive net can't reproduce the per-scenario oracle dive). Lifting the shared Up specialist further would raise both Up regions at once.

**On the oracle's 90% (per-scenario) vs a deployable policy.** The repair oracle reaches ~90% *per ball* by CEM-optimising an open-loop residual for that exact scenario. We tried to deploy those dives directly (`eval_residlib.py`: predict the ballistic crossing, look up the nearest oracle dive, execute open-loop ± phase-sync): it scored 69–74%, *below* the base, disrupting every region. The residual is co-optimised with the base's exact closed-loop trajectory for one scenario, so on any other ball it fights the reactive base — it does not transfer. So the oracle's 90% is a per-scenario optimum, not deployable headroom. A *single* reactive MLP caps ~80.5%; the **6-way region MoE** reaches **91.0%** once it is routed early and the weak side is mirrored — +18.5% over the published reference (72.5%).

The policy is a `960 → 1024 → 512 → 256 → 29` MLP. Input = 10 stacked frames of
(ball position in robot frame, base angular velocity, projected gravity, 29 joint
positions, 29 joint velocities, last action). Output = 29 joint-position targets
(PD, 50 Hz). Evaluation uses the deterministic mean.

---

## Why this is hard (and why we don't just clone the teacher)

A per-region diagnosis (`diagnose_gk.py`) shows the **reference teacher itself is
only 72.5%**, bottlenecked by **high balls**: low balls ~95%, but Right-Up 61% /
**Left-Up 38%**. Pure imitation can never beat the teacher, so to do better we
must *generate* high-ball saving behaviour the teacher doesn't have — that is the
job of the repair oracle (Stage B). RL (Stage C) then optimises the **exact**
block objective and is what pushes past the imitation ceiling.

---

## Stage A — Teacher distillation  (→ 65%)

`scripts/distill_goalkeeper.py`. The reference `GoalkeeperActorCritic` is
*inference-only* under rsl_rl 5.0.1 (legacy API), so we DAgger-clone its action
mapping into a trainable native MLP. This gives a competent, reactive **diving**
base policy (`goalkeeper_distilled_v3.pt`).

## Stage B — Repair oracle + DAgger  (→ 69.6%)

`scripts/repair_oracle.py`, `scripts/distill_repairs.py`.

The oracle is a **per-scenario trajectory optimiser**. For a frozen base policy
`π`, it searches an **open-loop residual action sequence** `r_t` so that
`a_t = clip(π(o_t) + r_t)` blocks a ball `π` would have conceded:

- **Layout:** `N = G scenarios × P population`. All `P` envs of a scenario share
  one forced ball trajectory; each gets a different residual = the CEM population.
- **Search:** iCEM over 12 spline knots concentrated in the save window
  (steps 0–80), elite-mean as the point estimate, elite carry-over.
- **Dense cost:** `w_goal·conceded + w_dist·min(blocking-link → ballistic
  crossing point) + w_res·‖r‖² + w_upright·…`, so the search has gradient even
  before the first contact. The crossing point is the ball's `(y,z)` where its
  ballistic trajectory crosses the keeper plane — computable from the observed
  ball position+velocity to ±0.05 m by step ~10 (legitimate, not privileged).

This **proves high balls are saveable**: Left-Up 40% → 87% in-sample, all-region
67% → 90%. We then **DAgger-distil** the repaired `(obs, action)` pairs back into
the student (oracle base = the *current student*, so the ~65% it already blocks
keep their own action and only its failures get corrected → easy balls protected).

Two correctness details that each previously caused a silent collapse:
1. **Eval-matched demonstrations.** Forcing the ball *after* `env.reset()` leaves
   10 stale ball-history frames → train/eval mismatch. Fixed by forcing the
   scenario *through* reset (`env._gk_forced` read by
   `reset_ball_with_parabolic_trajectory`).
2. **Base = current student, not the teacher** — keeps the demonstrations on the
   policy's own distribution (true DAgger), so corrections are a gentle nudge.

## Stage C — RL polish  (→ 78.4%)  ← the key mechanism

`scripts/train_polish.py` + `src/tasks/soccer/modules/bc_anchor_ppo.py`.

Imitation caps ~70% (a single reactive net can't reproduce the per-scenario
oracle exactly, and learning hard-ball corrections interferes with easy balls).
To break that ceiling we **reinforcement-learn directly against the true block
metric**, starting from the diving student. Prior RL here always collapsed
(forgetting / a non-directional "flop" / cold critic); `BCAnchorPPO` adds three
safeguards that make it stable:

1. **Reward = the exact objective + safe shaping.**
   `goal_conceded` (−15, fires exactly when the ball crosses the goal — identical
   to the eval criterion) + `intercept_point` (+2, pulls any limb to the ballistic
   crossing point — *directional*) + `posture` (+1, anti-flop) + small
   `action_rate`. **No coverage / whole-body-to-current-ball reward** — those are
   "flop attractors" that caused every earlier collapse.
2. **`BCAnchorPPO` = PPO + two add-ons:**
   - **Tiny action-std clamp (0.07–0.08)** so the *deterministic mean* (what eval
     uses) is what improves, not high-variance flailing.
   - **A behaviour-cloning anchor**: every update also takes one gradient step
     pulling the actor toward the oracle's repair actions, which **prevents the
     forgetting/drift** that collapsed naive fine-tuning.
3. **Critic warm-up, then a train → eval → rollback loop.** Warm the value
   function first (actor frozen). Then repeatedly: 5 PPO iters → evaluate the
   deterministic policy on 1024+ trials → keep it if it's a new best, **roll back
   to the best checkpoint if block rate drops >2%**. So training can only keep or
   improve, and it must be run **long (120+ blocks)** — it keeps climbing.

Result (seed 2810, 4096 trials): 77.3% → **80.5%** (`goalkeeper_polished_v2.pt`,
the best single MLP), improving every region.

The single-MLP RL polish then plateaus at ~80.5% across ~15 reward/curriculum/
DAgger/symmetry configs. The tell: its **weakest region shifts every run** (fix
Right-Up and Left-Up drops). A single feed-forward MLP cannot hold all six
region-dives at high fidelity at once — a capacity/architecture limit, not a
tuning one. That motivates Stage D.

## Stage D — Region mixture-of-experts  (→ 83.1% → 86% → **91.0%**)  ← current best

`scripts/train_polish.py --train-regions ...` (specialists) + `scripts/eval_moe.py`
(3-way) → `scripts/eval_moe6.py` (6-way, the deployed best). The headline result and
the two eval-time fixes that took it 86 → 91% are documented at the top of this file
("The 86 → 91 jump came from two eval-time fixes"). Below is how the MoE itself was built.

We train **height-group specialists** off the v2 policy, each RL-polished on only
its ball regions (`--train-regions`): a **Low** expert (regions 4,5), a **Mid**
expert (0,1), and an **Up** expert (2,3). Freed of the trade-off, each beats the
single net on its own region (e.g. Up's Left-Up 68→76).

At inference a fixed **ballistic-crossing gate** routes each ball to the right
expert: from the *observed* ball position+velocity it extrapolates the ball's
height `z` where it crosses the keeper plane (`x_rel=0`), latches the height-group
once the ball is clearly approaching, and holds it for the episode. `z < 0.80 →
Low, 0.80–1.30 → Mid, > 1.30 → Up`. The gate is legitimate (a real keeper predicts
where the ball is going) and adds no learned parameters.

Result (seed 2810, 4096): **83.1%** — per-region Mid 77/82, Up 74/76, Low 96/92.
Bundled as the single deployable file `goalkeeper_moe_v3.pt` (3 expert state-dicts
+ gate thresholds); eval with `eval_moe.py --moe ...`.

**From 83.1% (3-way) to 91.0% (6-way).** Splitting into **one specialist per region**
(`eval_moe6.py`, six experts) and then fixing **routing timing** (latch the gate as soon
as the ball is approaching, `latch_hi=5.0`) and **the left-bias** (route Right-Up through
the mirrored Left-Up expert, `mirror_map="2:3"`) took the MoE to **91.0%** (seed 2810,
4096 trials; ~90.5% mean across seeds). Both fixes are eval-time only — see the top of
this file for the full diagnosis. The deployed bundle `goalkeeper_moe6.pt` is
self-contained (6 actor state-dicts + gate config); eval needs no flags.

---

## Files

Pipeline (this PR):
- `scripts/repair_oracle.py` — CEM/iCEM per-scenario repair oracle (`--mode
  prove` to measure base→repaired, `--mode collect` to dump a dataset).
- `scripts/distill_repairs.py` — BC-distil repaired `(obs, action)` pairs into the
  native MLP student.
- `scripts/train_polish.py` — RL polish loop (BCAnchorPPO, eval + rollback).
- `src/tasks/soccer/modules/bc_anchor_ppo.py` — PPO + std-clamp + BC anchor.
- `scripts/diagnose_gk.py` — vectorised N-trial eval with per-region breakdown +
  near-miss histogram (native, reference, custom-net, or residual checkpoints).
- `src/tasks/soccer/mdp/goalkeeper_ball_reset.py` — adds the forced-scenario reset
  path (`env._gk_forced`) used by the oracle.

Mixture-of-experts (the deployed best):
- `scripts/eval_moe6.py` — 6-way MoE evaluator + ballistic gate (reads `latch_hi`,
  `mirror_map`, z/vz thresholds from the bundle), with a per-region routing diagnostic.
- `scripts/eval_moe.py` — earlier 3-way (Low/Mid/Up) MoE evaluator.
- `src/tasks/soccer/modules/symmetry.py` — FK-validated L/R mirror (`mirror_obs`,
  `mirror_action`) used by the Up-expert mirror and `--sym-coef`.
- `scripts/viz_moe6.py` — per-episode mp4 renderer (run on WSL; see its docstring).
- `src/assets/soccer/weight/goalkeeper_moe6.pt` — **the final 91.0% policy** (39 MB,
  actor-only: 6 region experts + gate config). `goalkeeper_polished_v2.pt` is the best
  single MLP (80.5%); `goalkeeper_polished.pt` was the earlier 78.4% milestone.

Stage-A / eval (pre-existing): `scripts/distill_goalkeeper.py`,
`scripts/eval_naive_goalkeeper.py`, `src/tasks/soccer/config/g1/gk_train_cfg.py`,
`src/assets/soccer/weight/goalkeeper_distilled_v3.pt`.

## Reproduce

```bash
pip install -e . --no-build-isolation        # editable install
export MUJOCO_GL=egl WANDB_MODE=disabled

# A. teacher distillation  (→ ~65%)
python scripts/distill_goalkeeper.py --num-envs 512 --dagger-iters 24 \
    --out logs/rsl_rl/g1_goalkeeper/distilled/model.pt

# B. repair oracle: prove high balls are saveable, then collect a dataset + distil
python scripts/repair_oracle.py --mode prove   --checkpoint <student> --regions 0 1 2 3
python scripts/repair_oracle.py --mode collect --checkpoint <student> \
    --regions 0 1 2 3 4 5 --G 16 --P 64 --iters 6 --clip 1.0 --batches 64 --out logs/repairs/r1.pt
python scripts/distill_repairs.py --data logs/repairs/r1.pt --resume <student> --out logs/repairs/r1_student.pt

# C. RL polish  (→ 78.4%) — train long
python scripts/train_polish.py --init logs/repairs/r1_student.pt --bc-data logs/repairs/r1.pt \
    --warmup 50 --block-iters 5 --blocks 120 --lr 6e-5 --std 0.08 --bc-coef 1.5 \
    --w-conceded 15 --w-intercept 2 --w-body 0 --w-stop 0 --w-posture 1.0 \
    --out logs/repairs/polished.pt

# D. region specialists (per region) then eval the 6-way MoE  (→ 91.0%)
python scripts/train_polish.py --init <v2> --bc-data <oracle> --train-regions 3 --w-cross 6 --out logs/repairs/sr3d.pt
python scripts/eval_moe6.py --moe6 src/assets/soccer/weight/goalkeeper_moe6.pt --seed 2810 --num-envs 256 --batches 16

# grader-style eval (our modified loader auto-detects the MoE6 bundle)
python scripts/eval_naive_goalkeeper.py --headless --num-trials 50 --checkpoint src/assets/soccer/weight/goalkeeper_moe6.pt

# visualization — render on WSL, NOT lab_1 (lab_1 EGL glitches on dives; see viz_moe6.py)
python scripts/viz_moe6.py --episodes 12 --seed 2810   # -> viz_moe6_wsl/ep*.mp4
```

## Phase 2 (tournament REST API) — status

The deployed policy plugs into the Phase-1 block-rate eval (`eval_naive_goalkeeper.py`,
which the rules allow us to modify). The Phase-2 standardized API (`scripts/api_server.py`
served to `scripts/compete.py`) needs a **goalkeeper-specific customisation** before the
MoE6 can run there — the reference server stacks history *frame-major* while the policy
expects mjlab's *term-major* layout, and the MoE gate must read the ball pos/vel from the
`raw_state` (both are provided by the protocol; `/reset` already calls `policy.reset()`).
This is the per-team `compute_obs()` customisation the assignment expects; not yet wired.
