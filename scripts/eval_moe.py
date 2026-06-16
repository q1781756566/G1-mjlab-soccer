"""Mixture-of-experts goalkeeper: a ballistic-crossing GATE routes each env to a
height-specialist (Up / Mid / Low). The single feed-forward MLP can't hold all
regions at once (its weak region shifts every run); specialists each beat the
general policy on their own region, and this gate combines them.

The gate computes the ball's ballistic crossing height z at the keeper plane
(x_rel=0) from the OBSERVED ball position+velocity (legitimate — a real keeper
does this), latches the height-group once the ball is clearly approaching, and
holds it for the rest of the episode. Deployable: scripts/eval_naive_goalkeeper
can load this same router.
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


def _load(env, ckpt, dev):
  runner = MjlabOnPolicyRunner(env, asdict(goalkeeper_train_runner_cfg()), device=dev)
  runner.load(ckpt, load_cfg={"actor": True})
  return runner.get_inference_policy(device=dev)


@dataclass
class Cfg:
  moe: str = ""             # single bundled checkpoint (overrides up/mid/low + z thresholds)
  up: str = "logs/repairs/spUp.pt"
  mid: str = "logs/repairs/spMid.pt"
  low: str = "logs/repairs/spLow.pt"
  num_envs: int = 256
  batches: int = 16
  seed: int = 2810
  z_low: float = 0.85       # z_cross < z_low -> Low specialist
  z_up: float = 1.35        # z_cross > z_up  -> Up specialist; else Mid
  land_x: float = 0.0       # extrapolate ball z to this x_rel (0=keeper plane, -0.35=landing region)
  perfect_gate: bool = False  # route by TRUE region (privileged) to upper-bound the MoE
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
  if cfg.moe:   # deployable single-file bundle: 3 expert ckpts + gate thresholds
    import tempfile, os
    b = torch.load(cfg.moe, map_location="cpu", weights_only=False)
    cfg.z_low = b.get("z_low", cfg.z_low); cfg.z_up = b.get("z_up", cfg.z_up)
    paths = {}
    for k in ("low", "mid", "up"):
      p = os.path.join(tempfile.gettempdir(), f"_moe_{k}.pt"); torch.save(b[k], p); paths[k] = p
    cfg.low, cfg.mid, cfg.up = paths["low"], paths["mid"], paths["up"]
  experts = [_load(env, cfg.low, dev), _load(env, cfg.mid, dev), _load(env, cfg.up, dev)]  # 0=Low,1=Mid,2=Up
  ball = env.unwrapped.scene["ball"]; org = env.unwrapped.scene.env_origins
  N = cfg.num_envs; g = 9.81

  def gate_group():
    bp = ball.data.root_link_pos_w; bv = ball.data.root_link_lin_vel_w
    bx = bp[:, 0] - org[:, 0]
    vx = bv[:, 0]
    t = torch.clamp(-(bx - cfg.land_x) / (vx - 1e-3), 0.0, 2.0)
    z_cross = bp[:, 2] + bv[:, 2] * t - 0.5 * g * t * t
    grp = torch.ones(N, dtype=torch.long, device=dev)            # default Mid
    grp = torch.where(z_cross < cfg.z_low, torch.zeros_like(grp), grp)
    grp = torch.where(z_cross > cfg.z_up, torch.full_like(grp, 2), grp)
    # only a valid estimate when the ball is approaching (vx<0) and still in front
    valid = (vx < -1.0) & (bx > 0.2) & (bx < 3.5)
    return grp, valid

  blocked = 0; total = 0
  rec = []
  for b in range(cfg.batches):
    obs = env.reset()
    if isinstance(obs, tuple): obs = obs[0]
    with torch.no_grad():
      latched = torch.full((N,), -1, dtype=torch.long, device=dev)
      entered = torch.zeros(N, dtype=torch.bool, device=dev)
      region = env.unwrapped._gk_region.clone().long() if hasattr(env.unwrapped, "_gk_region") else None
      # perfect-gate: route by the true region group [Low=0,Mid=1,Up=2]
      r2g = torch.tensor([1, 1, 2, 2, 0, 0], device=dev)
      pg = r2g[region] if (cfg.perfect_gate and region is not None) else None
      for _ in range(150):
        grp, valid = gate_group()
        newl = valid & (latched < 0)
        latched = torch.where(newl, grp, latched)
        use = pg if pg is not None else torch.where(latched < 0, torch.ones_like(latched), latched)  # default Mid pre-latch
        acts = torch.stack([e(obs) for e in experts], 0)   # (3,N,29)
        a = acts[use, torch.arange(N, device=dev)]
        res = env.step(a); obs = res[0]
        bp = ball.data.root_link_pos_w
        entered |= ((bp[:, 0] - org[:, 0]) <= -0.5) & ((bp[:, 1] - org[:, 1]).abs() <= 1.5) & (bp[:, 2] <= 1.8)
    blocked += int((~entered).sum()); total += N
    if region is not None:
      for i in range(N): rec.append((int(region[i]), bool(~entered[i])))
    print(f"  batch {b+1}/{cfg.batches}: {blocked}/{total} = {100*blocked/total:.1f}%", flush=True)
  print(f"\nMoE OVERALL: {blocked}/{total} = {100*blocked/total:.1f}%")
  if rec:
    for rg in range(6):
      rr = [x for x in rec if x[0] == rg]
      if rr: print(f"  {_REGION[rg]:<11} {sum(x[1] for x in rr)}/{len(rr)} = {100*sum(x[1] for x in rr)/len(rr):.1f}%")
  env.close()


if __name__ == "__main__":
  import mjlab.tasks, src.tasks  # noqa
  main(tyro.cli(Cfg, prog="eval_moe"))
