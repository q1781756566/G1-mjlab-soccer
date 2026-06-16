"""Deploy the repair-oracle's OPEN-LOOP dives via a residual library.

A reactive net caps high balls at ~80% because it can't reproduce the oracle's
precise dives closed-loop. This controller sidesteps that: it predicts the ball's
ballistic crossing (y,z) + flight-time from the first observed frames (legitimate
— a keeper predicts where the ball is going), looks up the oracle's pre-optimised
dive (residual spline) for the nearest matching scenario, and executes it
OPEN-LOOP on top of a reactive base — `a_t = clip(base(o_t) + dive(t))`.

Both the oracle rollout and this deployment start the episode at ball launch
(step 0 = reset), so the dive — indexed by absolute step — stays time-aligned as
long as the lookup matches flight-time (it does: t_flight is in the library key).
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import torch, tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper, MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from src.tasks.soccer.config.g1.gk_train_cfg import goalkeeper_train_runner_cfg

_REGION = ["Right-Mid", "Left-Mid", "Right-Up", "Left-Up", "Right-Low", "Left-Low"]


@dataclass
class Cfg:
  base: str = "src/assets/soccer/weight/goalkeeper_polished_v2.pt"
  lib: str = "logs/repairs/residlib_T2.pt"
  num_envs: int = 256
  batches: int = 16
  seed: int = 2810
  clip: float = 1.0          # residual clamp at deploy
  predict_step: int = 8      # start predicting/diving once the ball is observed this long
  device: str = "cuda:0"


def main(cfg: Cfg):
  configure_torch_backends()
  dev = cfg.device
  torch.manual_seed(cfg.seed)
  env_cfg = load_env_cfg("Eval-Goalkeeper", play=False)
  env_cfg.scene.num_envs = cfg.num_envs; env_cfg.seed = cfg.seed
  if "fell_over" in env_cfg.terminations:
    env_cfg.terminations["fell_over"] = None
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=dev), clip_actions=100.0)

  runner = MjlabOnPolicyRunner(env, asdict(goalkeeper_train_runner_cfg()), device=dev)
  runner.load(cfg.base, load_cfg={"actor": True})
  base = runner.get_inference_policy(device=dev)

  d = torch.load(cfg.lib, map_location=dev, weights_only=False)
  bk = d["blocked"].bool()
  feat = d["feat"][bk].to(dev)           # (M,3) crossing_y, crossing_z, t_flight
  mu = d["mu"][bk].to(dev)               # (M,K,J)
  scross = d["scross"][bk].to(dev)       # (M,) ball crossing step in the library scenario
  K, J = mu.shape[1], mu.shape[2]
  span = int(d.get("knot_span", 80)); T = 150
  fmean = feat.mean(0); fstd = feat.std(0).clamp(min=1e-3)
  featn = (feat - fmean) / fstd
  kt = torch.linspace(0, span, K, device=dev)
  print(f"[INFO] library {feat.shape[0]} dives, K={K} span={span}", flush=True)

  def interp_shift(my_mu, s):  # per-env spline value at (clamped) step s -> (N,J)
    s = s.clamp(0, span)
    ix = torch.clamp(torch.bucketize(s, kt) - 1, 0, K - 2)
    a0 = kt[ix]; a1 = kt[ix + 1]; fr = ((s - a0) / (a1 - a0).clamp(min=1e-6)).unsqueeze(-1)
    ar = torch.arange(my_mu.shape[0], device=dev)
    return (1 - fr) * my_mu[ar, ix] + fr * my_mu[ar, ix + 1]

  ball = env.unwrapped.scene["ball"]; org = env.unwrapped.scene.env_origins
  N = cfg.num_envs; g = 9.81
  blocked = 0; total = 0; rec = []
  for b in range(cfg.batches):
    obs = env.reset()
    if isinstance(obs, tuple): obs = obs[0]
    with torch.no_grad():
      my_mu = torch.zeros(N, K, J, device=dev); latched = torch.zeros(N, dtype=torch.bool, device=dev)
      shift = torch.zeros(N, device=dev)   # deploy_step - lib_step alignment (phase-sync)
      entered = torch.zeros(N, dtype=torch.bool, device=dev)
      region = env.unwrapped._gk_region.clone().long() if hasattr(env.unwrapped, "_gk_region") else None
      for t in range(T):
        bp = ball.data.root_link_pos_w; bv = ball.data.root_link_lin_vel_w
        bx = bp[:, 0] - org[:, 0]; vx = bv[:, 0]
        valid = (vx < -1.0) & (bx > 0.2) & (bx < 4.0) & (~latched) & (t >= cfg.predict_step)
        if valid.any():
          tc = torch.clamp(-bx / (vx - 1e-3), 0.0, 2.0)
          cy = (bp[:, 1] - org[:, 1]) + bv[:, 1] * tc
          cz = bp[:, 2] + bv[:, 2] * tc - 0.5 * g * tc * tc
          tf = t * 0.02 + tc
          q = (torch.stack([cy, cz, tf], -1) - fmean) / fstd      # (N,3) normalised
          nn = torch.cdist(q[valid], featn).argmin(1)             # nearest dive
          vi = valid.nonzero(as_tuple=False).squeeze(-1)
          my_mu[vi] = mu[nn]; latched[vi] = True
          s_dep = t + tc[vi] / 0.02                                # deploy crossing step
          shift[vi] = s_dep - scross[nn]                           # align dive crossing to ball crossing
        R = interp_shift(my_mu, t - shift).clamp(-cfg.clip, cfg.clip)
        a = (base(obs) + torch.where(latched[:, None], R, torch.zeros_like(R))).clamp(-100, 100)
        obs = env.step(a)[0]
        bp = ball.data.root_link_pos_w
        entered |= ((bp[:, 0] - org[:, 0]) <= -0.5) & ((bp[:, 1] - org[:, 1]).abs() <= 1.5) & (bp[:, 2] <= 1.8)
      blocked += int((~entered).sum()); total += N
      if region is not None:
        for i in range(N): rec.append((int(region[i]), bool(~entered[i])))
    print(f"  batch {b+1}/{cfg.batches}: {blocked}/{total} = {100*blocked/total:.1f}%", flush=True)
  print(f"\nRESIDLIB OVERALL: {blocked}/{total} = {100*blocked/total:.1f}%")
  for rg in range(6):
    rr = [x for x in rec if x[0] == rg]
    if rr: print(f"  {_REGION[rg]:<11} {100*sum(x[1] for x in rr)/len(rr):.1f}%")
  env.close()


if __name__ == "__main__":
  import mjlab.tasks, src.tasks  # noqa
  main(tyro.cli(Cfg, prog="eval_residlib"))
