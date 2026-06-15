"""Distill the repair-oracle's (obs, repaired_action) pairs into the native MLP
student. Trains the student to reproduce the closed-loop repaired behavior
(base + CEM residual) that blocks balls the base policy missed.

Init from an existing student (or scratch). Only BLOCKED repaired frames are used
as targets. Saves a native rsl_rl checkpoint loadable by eval_naive_goalkeeper.py.
"""
from __future__ import annotations
import os
from dataclasses import asdict, dataclass
import torch, tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from src.tasks.soccer.config.g1.gk_train_cfg import goalkeeper_train_runner_cfg


@dataclass
class Cfg:
  data: tuple[str, ...] = ("logs/repairs/repairs.pt",)
  out: str = "logs/rsl_rl/g1_goalkeeper/distilled/model_repaired.pt"
  resume: str | None = "src/assets/soccer/weight/goalkeeper_distilled_v3.pt"
  epochs: int = 30
  batch_size: int = 16384
  lr: float = 5.0e-4
  lr_final: float = 5.0e-5
  blocked_only: bool = True
  hidden: tuple[int, ...] = ()     # override actor/critic hidden dims (e.g. 2048 1024 512 256)
  seed: int = 2810
  device: str = "cuda:0"


def main(cfg: Cfg):
  configure_torch_backends()
  dev = cfg.device
  torch.manual_seed(cfg.seed)

  X, Y = [], []
  for p in cfg.data:
    d = torch.load(p, map_location="cpu", weights_only=False)
    o, a, bk = d["obs"], d["act"], d["blocked"].bool()
    if cfg.blocked_only:
      o, a = o[bk], a[bk]
    X.append(o); Y.append(a)
    print(f"  loaded {p}: {o.shape[0]} frames (blocked_only={cfg.blocked_only})", flush=True)
  X = torch.cat(X); Y = torch.cat(Y)
  n = X.shape[0]
  print(f"[INFO] total {n} training pairs, obs {X.shape[1]}, act {Y.shape[1]}", flush=True)

  # Build the student (env needed only to construct the runner/model shapes).
  env_cfg = load_env_cfg("Eval-Goalkeeper", play=False)
  env_cfg.scene.num_envs = 2
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=dev), clip_actions=100.0)
  agent = goalkeeper_train_runner_cfg()
  if cfg.hidden:
    agent.actor.hidden_dims = tuple(cfg.hidden)
    agent.critic.hidden_dims = tuple(cfg.hidden)
    print(f"[INFO] custom net hidden_dims={tuple(cfg.hidden)}", flush=True)
  runner = MjlabOnPolicyRunner(env, asdict(agent), device=dev)
  student = runner.alg.actor
  if cfg.resume:
    ck = torch.load(cfg.resume, map_location=dev, weights_only=False)
    student.load_state_dict(ck["actor_state_dict"])
    print(f"[INFO] resumed student from {cfg.resume}", flush=True)
  optim = torch.optim.Adam(student.parameters(), lr=cfg.lr)

  for ep in range(cfg.epochs):
    frac = ep / max(1, cfg.epochs - 1)
    lr = cfg.lr + (cfg.lr_final - cfg.lr) * frac
    for g in optim.param_groups: g["lr"] = lr
    perm = torch.randperm(n)
    last = 0.0
    for s in range(0, n, cfg.batch_size):
      idx = perm[s:s + cfg.batch_size]
      xb = X[idx].to(dev, non_blocking=True); yb = Y[idx].to(dev, non_blocking=True)
      pred = student.forward({"actor": xb})
      loss = torch.nn.functional.smooth_l1_loss(pred, yb)
      optim.zero_grad(); loss.backward(); optim.step()
      last = loss.item()
    if (ep + 1) % 5 == 0 or ep == 0:
      print(f"[INFO] epoch {ep+1}/{cfg.epochs} lr={lr:.1e} huber={last:.5f}", flush=True)

  os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
  saved = runner.alg.save(); saved["iter"] = 0
  saved["infos"] = {"env_state": {"common_step_counter": 0}}
  torch.save(saved, cfg.out)
  print(f"[INFO] saved repaired student to {cfg.out}", flush=True)
  env.close()


if __name__ == "__main__":
  import mjlab.tasks, src.tasks  # noqa
  main(tyro.cli(Cfg, prog="distill_repairs"))
