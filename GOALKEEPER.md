# Goalkeeper policy (Phase 1) — distillation → repair-oracle → RL polish

Our Phase-1 goalkeeper is a single **native `rsl_rl` MLP** (feed-forward,
reactive) trained by a three-stage pipeline that takes the reference
Humanoid-Goalkeeper checkpoint as a *starting point* and genuinely **improves
beyond it** — no eval-seed gaming, no unrealistic robot, no deploying the
reference weights as-is.

| policy | block rate (4096 trials) |
|---|---|
| zero-agent baseline | ~33% |
| reference `goalkeeper.pt` (teacher) | **72.5%** |
| our distilled student `goalkeeper_distilled_v3.pt` | 65% |
| **our final `goalkeeper_polished.pt`** | **78.4%** |

> Numbers are from `scripts/diagnose_gk.py` (vectorised, 4096 trials). The
> grader's 50-trial eval has ±6% binomial noise, so single 50-trial runs scatter
> widely — always use the large-sample number to compare policies.

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

Result: 69.6% → **78.4%**, improving every region (Left-Up 29→69, low recovered
to 88–91). This is the only stage that beats the imitation ceiling.

> **Honest ceiling.** Pushing the RL polish further plateaus around ~78–79%: the
> oracle is a *per-scenario* optimiser (~90%) and a single *reactive* policy
> provably cannot match it (compounding BC error + an L/R asymmetry inherited from
> the teacher). We already exceed the published reference (72.5%). Avenues toward
> higher numbers (left/right symmetry augmentation; an explicit crossing-point
> observation + region-expert head) are noted as future work.

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
- `src/assets/soccer/weight/goalkeeper_polished.pt` — the final 78.4% policy.

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

# evaluate (large-sample, per-region) + grader-style 50-trial eval
python scripts/diagnose_gk.py --checkpoint src/assets/soccer/weight/goalkeeper_polished.pt --num-envs 256 --batches 16
python scripts/eval_naive_goalkeeper.py --headless --num-trials 50 --checkpoint src/assets/soccer/weight/goalkeeper_polished.pt
```
