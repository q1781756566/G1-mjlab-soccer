"""Runner / network config for the distilled goalkeeper policy.

Our Phase-1 goalkeeper is a native rsl_rl MLP policy distilled (DAgger) from the
reference Humanoid-Goalkeeper checkpoint. This module defines the actor-critic
network used both to build the student during distillation
(``scripts/distill_goalkeeper.py``) and to load it at evaluation
(``scripts/eval_naive_goalkeeper.py`` / ``scripts/render_cases.py``).

obs_normalization is OFF: the goalkeeper observations are already manually scaled
(ang_vel*0.25, dof_vel*0.05, ...) to match the reference's input distribution, so
an extra running normalizer would double-normalize.
"""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

# MLP hidden dims of the distilled goalkeeper actor/critic. Must match the saved
# checkpoint (src/assets/soccer/weight/goalkeeper_distilled_v3.pt).
_GK_MLP_HIDDEN = (1024, 512, 256)


def goalkeeper_train_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """rsl_rl runner config for the native MLP goalkeeper policy."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=_GK_MLP_HIDDEN,
      activation="elu",
      obs_normalization=False,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(
      hidden_dims=_GK_MLP_HIDDEN,
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
      entropy_coef=0.005, num_learning_epochs=5, num_mini_batches=4,
      learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95,
      desired_kl=0.01, max_grad_norm=1.0,
    ),
    experiment_name="g1_goalkeeper",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )
