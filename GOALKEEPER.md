# Goalkeeper policy (Phase 1)

Our Phase-1 goalkeeper is a native `rsl_rl` MLP policy obtained by **DAgger
distillation** of the reference Humanoid-Goalkeeper checkpoint
(`src/assets/soccer/weight/goalkeeper.pt`) into the framework's standard,
trainable MLP architecture.

**Why distillation?** The reference `GoalkeeperActorCritic` implements the legacy
rsl_rl ActorCritic API and is *inference-only* under rsl_rl 5.0.1 — it cannot be
trained by the new modular PPO. So we distill its behavior into a native MLP
(which the framework can train and serve) and load that natively for evaluation.

## Files

- `scripts/distill_goalkeeper.py` — DAgger distillation of the reference teacher
  into the MLP student (collect teacher actions on the `Eval-Goalkeeper` env,
  fit the student, aggregate, repeat).
- `src/tasks/soccer/config/g1/gk_train_cfg.py` — the MLP actor/critic config
  (`goalkeeper_train_runner_cfg`) used to build/load the student.
- `scripts/eval_naive_goalkeeper.py` — evaluation; `_load_policy` now also loads
  our native MLP checkpoint (in addition to the reference HIMPPO format).
- `scripts/render_cases.py` — render N random episodes to mp4 (save / goal).
- `src/assets/soccer/weight/goalkeeper_distilled_v3.pt` — the trained policy.
- `pyproject.toml` — adds a PEP 660 build backend so the repo installs editable.

## Result

Block rate ≈ **74%** over 100 eval trials (`eval_naive_goalkeeper.py`,
`Eval-Goalkeeper`, 6-region parabolic ball, fell_over disabled) — on par with the
reference checkpoint's own ~75–80% on this benchmark.

## Reproduce

```bash
# install (editable)
pip install -e . --no-build-isolation

# distill the policy (GPU)
MUJOCO_GL=egl WANDB_MODE=disabled python scripts/distill_goalkeeper.py \
    --num-envs 512 --dagger-iters 24 \
    --out logs/rsl_rl/g1_goalkeeper/distilled/model.pt

# evaluate (50 trials)
MUJOCO_GL=egl python scripts/eval_naive_goalkeeper.py --headless --num-trials 50 \
    --checkpoint src/assets/soccer/weight/goalkeeper_distilled_v3.pt

# visualize random cases
MUJOCO_GL=egl python scripts/render_cases.py \
    --ckpt src/assets/soccer/weight/goalkeeper_distilled_v3.pt --n 20 --out-dir viz
```
