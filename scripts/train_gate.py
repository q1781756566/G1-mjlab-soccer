"""Train a learned gate for the 6-way MoE: ball (pos+vel) at the prediction step
-> region (0..5). The ballistic threshold gate misroutes ~3.5% (low-landing balls
cross the keeper plane at overlapping heights); the perfect gate would give 89%.
The ball state fully determines its region (ballistic), so a tiny MLP recovers it.

Collects (ball_pos[3], ball_vel[3]) at the first step the ball is clearly
approaching (bx<obs_x), labelled by the true region, then fits a classifier.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch, tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends


@dataclass
class Cfg:
  num_envs: int = 512
  batches: int = 40           # resets; samples ~= num_envs * batches
  obs_x: float = 3.0          # record ball state when bx first drops below this
  epochs: int = 60
  out: str = "logs/repairs/gate.pt"
  seed: int = 1234
  device: str = "cuda:0"


def main(cfg: Cfg):
  configure_torch_backends()
  dev = cfg.device; torch.manual_seed(cfg.seed)
  env_cfg = load_env_cfg("Eval-Goalkeeper", play=False)
  env_cfg.scene.num_envs = cfg.num_envs; env_cfg.seed = cfg.seed
  if "fell_over" in env_cfg.terminations: env_cfg.terminations["fell_over"] = None
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=dev), clip_actions=100.0)
  ball = env.unwrapped.scene["ball"]; org = env.unwrapped.scene.env_origins
  N = cfg.num_envs; A = env.num_actions
  X, Y = [], []
  for b in range(cfg.batches):
    env.reset()
    reg = env.unwrapped._gk_region.clone().long()
    rec = torch.zeros(N, dtype=torch.bool, device=dev)
    with torch.no_grad():
      for _ in range(40):
        bp = ball.data.root_link_pos_w - org; bv = ball.data.root_link_lin_vel_w
        bx = bp[:, 0]; vx = bv[:, 0]
        take = (vx < -1.0) & (bx < cfg.obs_x) & (bx > 0.2) & (~rec)
        if take.any():
          ti = take.nonzero(as_tuple=False).squeeze(-1)
          X.append(torch.cat([bp[ti], bv[ti]], -1).cpu()); Y.append(reg[ti].cpu())
          rec[ti] = True
        env.step(torch.zeros(N, A, device=dev))
    print(f"  batch {b+1}/{cfg.batches}: {sum(x.shape[0] for x in X)} samples", flush=True)
  X = torch.cat(X).to(dev); Y = torch.cat(Y).to(dev)
  mean = X.mean(0); std = X.std(0).clamp(min=1e-3)
  Xn = (X - mean) / std
  net = torch.nn.Sequential(torch.nn.Linear(6, 128), torch.nn.ReLU(),
                            torch.nn.Linear(128, 128), torch.nn.ReLU(),
                            torch.nn.Linear(128, 6)).to(dev)
  opt = torch.optim.Adam(net.parameters(), 1e-3)
  n = X.shape[0]
  for ep in range(cfg.epochs):
    perm = torch.randperm(n, device=dev); last = 0.0
    for s in range(0, n, 8192):
      idx = perm[s:s + 8192]
      loss = torch.nn.functional.cross_entropy(net(Xn[idx]), Y[idx])
      opt.zero_grad(); loss.backward(); opt.step(); last = loss.item()
    if (ep + 1) % 10 == 0:
      acc = (net(Xn).argmax(1) == Y).float().mean().item()
      print(f"  epoch {ep+1} loss={last:.4f} train_acc={acc:.4f}", flush=True)
  acc = (net(Xn).argmax(1) == Y).float().mean().item()
  torch.save({"state": net.state_dict(), "mean": mean.cpu(), "std": std.cpu(),
              "obs_x": cfg.obs_x, "acc": acc}, cfg.out)
  print(f"[INFO] gate train_acc={acc:.4f} saved to {cfg.out}", flush=True)
  env.close()


if __name__ == "__main__":
  import mjlab.tasks, src.tasks  # noqa
  main(tyro.cli(Cfg, prog="train_gate"))
