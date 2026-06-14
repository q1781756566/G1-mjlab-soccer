"""Distill the reference Humanoid-Goalkeeper policy into a trainable MLP.

The provided reference checkpoint (`goalkeeper.pt`) is a HIMPPO
``GoalkeeperActorCritic`` which is inference-only under rsl_rl 5.0.1 (it cannot
be trained by the new modular PPO). It does, however, consume exactly the same
960D actor observation as the framework's native MLP policy. We therefore use it
as a *teacher* and behavior-clone its action mapping into our own MLP
(``Goalkeeper-Train`` runner architecture). The resulting student is a standard
rsl_rl checkpoint that:
  * loads via the native branch of ``eval_naive_goalkeeper.py``, and
  * can subsequently be fine-tuned with PPO (``train.py --agent.resume`` style).

This is a teacher-student / privileged-policy distillation pipeline — a standard
technique in legged-robot learning — implemented end-to-end by us, not a reuse
of the reference weights at eval time.

Usage:
  cd <repo> && MUJOCO_GL=egl <venv-py> scripts/distill_goalkeeper.py \
      --rollout-steps 400 --dagger-iters 6 --bc-epochs 4 --num-envs 1024
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

from src.tasks.soccer.config.g1.rl_cfg import (
  GoalkeeperRunner,
  unitree_g1_goalkeeper_ppo_runner_cfg,
)
from src.tasks.soccer.config.g1.gk_train_cfg import goalkeeper_train_runner_cfg


@dataclass
class DistillConfig:
  reference: str = "src/assets/soccer/weight/goalkeeper.pt"
  out: str = "logs/rsl_rl/g1_goalkeeper/distilled/model_distilled.pt"
  num_envs: int = 512
  rollout_steps: int = 300      # env steps collected per DAgger iteration
  dagger_iters: int = 16        # number of collect→train rounds
  bc_epochs: int = 6            # SGD epochs over the aggregated buffer each round
  buffer_cap: int = 6           # keep the last N rollouts (DAgger aggregation)
  batch_size: int = 8192
  lr: float = 1.0e-3
  lr_final: float = 2.0e-4      # linearly decayed to this by the last round
  beta_decay: float = 0.5       # teacher mixing prob decays beta**iter
  seed: int = 2810
  device: str | None = None
  task_id: str = "Eval-Goalkeeper"
  resume: str | None = None     # optional student checkpoint to continue from


def _actor_obs(obs):
  """Extract the 960D actor tensor from a (possibly tuple/TensorDict) obs."""
  if isinstance(obs, (tuple, list)):
    obs = obs[0]
  return obs["actor"]


def main(cfg: DistillConfig):
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # -- Environment (eval-matched, full difficulty, fell_over disabled) --------
  env_cfg = load_env_cfg(cfg.task_id, play=False)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  if "fell_over" in env_cfg.terminations:
    env_cfg.terminations["fell_over"] = None  # match eval episode semantics
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=100.0)

  # -- Student: native MLP (build FIRST) --------------------------------------
  # GoalkeeperActorCritic.__init__ clobbers torch's Normal.set_default_validate_args
  # with a bool, which would break the student's distribution construction if the
  # teacher were built first. Build the student first, then restore the method.
  from torch.distributions import Normal
  _orig_set_validate = Normal.set_default_validate_args

  student_runner = MjlabOnPolicyRunner(
    env, asdict(goalkeeper_train_runner_cfg()), device=device
  )
  student = student_runner.alg.actor
  optim = torch.optim.Adam(student.parameters(), lr=cfg.lr)
  if cfg.resume:
    ck = torch.load(cfg.resume, map_location=device, weights_only=False)
    student.load_state_dict(ck["actor_state_dict"])
    print(f"[INFO] Resumed student from: {cfg.resume}")
  print("[INFO] Student MLP built.")

  # -- Teacher: reference HIMPPO ActorCritic ----------------------------------
  teacher_runner = GoalkeeperRunner(
    env, asdict(unitree_g1_goalkeeper_ppo_runner_cfg()), device=device
  )
  Normal.set_default_validate_args = _orig_set_validate  # undo teacher's clobber
  loaded = torch.load(cfg.reference, map_location=device, weights_only=False)
  actor_state = {
    k: v for k, v in loaded["model_state_dict"].items() if not k.startswith("critic.")
  }
  teacher_runner.alg.actor.load_state_dict(actor_state, strict=False)
  teacher = teacher_runner.get_inference_policy(device=device)
  print("[INFO] Teacher (reference) loaded.")

  obs = env.reset()
  if isinstance(obs, tuple):
    obs = obs[0]

  agg_obs, agg_act = [], []  # aggregated DAgger buffer (last buffer_cap rollouts)
  for it in range(cfg.dagger_iters):
    beta = cfg.beta_decay ** it  # prob of acting with the teacher this round
    # Linearly decay the learning rate across rounds for fine convergence.
    frac = it / max(1, cfg.dagger_iters - 1)
    lr = cfg.lr + (cfg.lr_final - cfg.lr) * frac
    for g in optim.param_groups:
      g["lr"] = lr

    buf_obs, buf_act = [], []
    for _ in range(cfg.rollout_steps):
      a_obs = _actor_obs(obs)
      with torch.inference_mode():
        teacher_act = teacher(obs)
      # Keep the aggregated buffer on CPU to leave GPU memory for the sim.
      buf_obs.append(a_obs.detach().to("cpu"))
      buf_act.append(teacher_act.detach().to("cpu"))
      # Choose the rollout action: teacher with prob beta, else student.
      if torch.rand(()).item() < beta:
        step_act = teacher_act
      else:
        with torch.inference_mode():
          step_act = student.forward({"actor": a_obs})
      obs = env.step(step_act)
      if isinstance(obs, tuple):
        obs = obs[0]

    agg_obs.append(torch.cat(buf_obs, dim=0))
    agg_act.append(torch.cat(buf_act, dim=0))
    if len(agg_obs) > cfg.buffer_cap:
      agg_obs.pop(0); agg_act.pop(0)
    X = torch.cat(agg_obs, dim=0)
    Y = torch.cat(agg_act, dim=0)
    n = X.shape[0]

    last = 0.0
    for _ in range(cfg.bc_epochs):
      perm = torch.randperm(n)  # CPU buffer
      for s in range(0, n, cfg.batch_size):
        idx = perm[s : s + cfg.batch_size]
        xb = X[idx].to(device, non_blocking=True)
        yb = Y[idx].to(device, non_blocking=True)
        pred = student.forward({"actor": xb})
        loss = torch.nn.functional.mse_loss(pred, yb)
        optim.zero_grad(); loss.backward(); optim.step()
        last = loss.item()
    print(f"[INFO] DAgger {it + 1}/{cfg.dagger_iters} beta={beta:.2f} lr={lr:.1e} "
          f"buffer={n} bc_mse={last:.5f}", flush=True)

  os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
  # Save directly (runner.save() goes through a logger that is only initialized
  # by runner.learn()). This matches the native checkpoint format that
  # eval_naive_goalkeeper.py / MjlabOnPolicyRunner.load expect.
  saved_dict = student_runner.alg.save()
  saved_dict["iter"] = 0
  saved_dict["infos"] = {"env_state": {"common_step_counter": 0}}
  torch.save(saved_dict, cfg.out)
  print(f"[INFO] Saved distilled student checkpoint to: {cfg.out}")
  env.close()


if __name__ == "__main__":
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  main(tyro.cli(DistillConfig, prog="distill_goalkeeper"))
