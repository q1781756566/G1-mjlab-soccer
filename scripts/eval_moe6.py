"""6-way mixture-of-experts: one specialist per region (R/L x Low/Mid/Up), routed
by a ballistic-crossing gate on BOTH the crossing height z and side y. Finer than
the 3-way height MoE — each expert focuses on a single region."""
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
  r = MjlabOnPolicyRunner(env, asdict(goalkeeper_train_runner_cfg()), device=dev)
  r.load(ckpt, load_cfg={"actor": True})
  return r.get_inference_policy(device=dev)


@dataclass
class Cfg:
  moe6: str = ""                 # single bundled checkpoint {sr:[6], z_low, z_up}
  dir: str = "logs/repairs"      # loads {dir}/sr0..sr5.pt
  prefix: str = "sr"
  num_envs: int = 256
  batches: int = 16
  seed: int = 2810
  z_low: float = 0.85
  z_up: float = 1.35
  land_x: float = 0.0           # extrapolate ball z to this x_rel (0=keeper plane, <0=landing)
  vz_low: float = -99.0         # if ball vz at crossing < this (steep descent) -> route Low
  latch_hi: float = 3.5         # latch the gate once bx drops below this (raise -> latch earlier)
  default_ckpt: str = ""        # generalist policy used BEFORE latching (avoids a wrong early commit)
  perfect_gate: bool = False    # route by TRUE region (upper bound)
  gate: str = ""                # learned gate checkpoint (ball pos+vel -> region)
  use_bundle_gate: bool = True  # read gate params (z_low/z_up/vz_low/latch_hi) from the bundle; set False to sweep via CLI
  mirror_map: str = ""          # replace experts via L/R mirror, e.g. "2:3" => expert2 := mirror(expert3)
  device: str = "cuda:0"


def main(cfg: Cfg):
  configure_torch_backends()
  dev = cfg.device; torch.manual_seed(cfg.seed)
  env_cfg = load_env_cfg("Eval-Goalkeeper", play=False)
  env_cfg.scene.num_envs = cfg.num_envs; env_cfg.seed = cfg.seed
  if "fell_over" in env_cfg.terminations: env_cfg.terminations["fell_over"] = None
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=dev), clip_actions=100.0)
  if cfg.moe6:
    import tempfile, os
    bd = torch.load(cfg.moe6, map_location="cpu", weights_only=False)
    if cfg.use_bundle_gate:
      cfg.z_low = bd.get("z_low", cfg.z_low); cfg.z_up = bd.get("z_up", cfg.z_up)
      cfg.vz_low = bd.get("vz_low", cfg.vz_low); cfg.latch_hi = bd.get("latch_hi", cfg.latch_hi)
      if not cfg.mirror_map: cfg.mirror_map = bd.get("mirror_map", "")
    cfg.dir = tempfile.mkdtemp(prefix="moe6_"); cfg.prefix = "_moe6_"  # unique per process (no parallel collision)
    for r in range(6): torch.save(bd["sr"][r], f"{cfg.dir}/_moe6_{r}.pt")
  experts = [_load(env, f"{cfg.dir}/{cfg.prefix}{r}.pt", dev) for r in range(6)]  # idx = region
  if cfg.mirror_map:  # exploit L/R symmetry: use a strong side's expert (mirrored) for the weak side
    from src.tasks.soccer.modules.symmetry import mirror_obs, mirror_action
    def _mir(p):
      def f(obs):  # obs may be a TensorDict; the actor accepts a plain {"actor": ...} dict
        return mirror_action(p({"actor": mirror_obs(obs["actor"])}))
      return f
    base_experts = list(experts)
    for pair in cfg.mirror_map.split(","):
      dst, src = (int(x) for x in pair.split(":"))
      experts[dst] = _mir(base_experts[src]); print(f"[INFO] expert {dst} := mirror(expert {src})", flush=True)
  default_policy = _load(env, cfg.default_ckpt, dev) if cfg.default_ckpt else None
  ball = env.unwrapped.scene["ball"]; org = env.unwrapped.scene.env_origins
  N = cfg.num_envs; g = 9.81

  gnet = gmean = gstd = None
  if cfg.gate:
    gd = torch.load(cfg.gate, map_location=dev, weights_only=False)
    gnet = torch.nn.Sequential(torch.nn.Linear(6, 128), torch.nn.ReLU(),
                               torch.nn.Linear(128, 128), torch.nn.ReLU(),
                               torch.nn.Linear(128, 6)).to(dev)
    gnet.load_state_dict(gd["state"]); gnet.eval()
    gmean = gd["mean"].to(dev); gstd = gd["std"].to(dev)

  def gate_region():
    bp = ball.data.root_link_pos_w; bv = ball.data.root_link_lin_vel_w
    bx = bp[:, 0] - org[:, 0]; vx = bv[:, 0]
    valid = (vx < -1.0) & (bx > 0.2) & (bx < cfg.latch_hi)
    if gnet is not None:
      f = torch.cat([torch.stack([bx, bp[:, 1] - org[:, 1], bp[:, 2]], -1), bv], -1)
      region = gnet((f - gmean) / gstd).argmax(1)
      return region, valid
    t = torch.clamp(-(bx - cfg.land_x) / (vx - 1e-3), 0.0, 2.0)
    cy = (bp[:, 1] - org[:, 1]) + bv[:, 1] * t
    cz = bp[:, 2] + bv[:, 2] * t - 0.5 * g * t * t
    base = torch.full_like(cy, 0, dtype=torch.long)               # Mid
    base = torch.where(cz < cfg.z_low, torch.full_like(base, 4), base)   # Low
    base = torch.where(cz > cfg.z_up, torch.full_like(base, 2), base)    # Up
    vz_cross = bv[:, 2] - g * t                                          # vertical vel at crossing
    base = torch.where(vz_cross < cfg.vz_low, torch.full_like(base, 4), base)  # steep descent -> Low
    region = base + (cy < 0).long()
    return region, valid

  blocked = 0; total = 0; rec = []
  for b in range(cfg.batches):
    obs = env.reset()
    if isinstance(obs, tuple): obs = obs[0]
    with torch.no_grad():
      latched = torch.full((N,), -1, dtype=torch.long, device=dev)
      entered = torch.zeros(N, dtype=torch.bool, device=dev)
      truereg = env.unwrapped._gk_region.clone().long() if hasattr(env.unwrapped, "_gk_region") else None
      for _ in range(150):
        reg, valid = gate_region()
        newl = valid & (latched < 0); latched = torch.where(newl, reg, latched)
        if cfg.perfect_gate and truereg is not None:
          use = truereg
        else:
          use = torch.where(latched < 0, torch.zeros_like(latched), latched)
        acts = torch.stack([e(obs) for e in experts], 0)   # (6,N,29)
        a = acts[use, torch.arange(N, device=dev)]
        if default_policy is not None and not cfg.perfect_gate:  # pre-latch: neutral generalist (no wrong early commit)
          a = torch.where((latched < 0).unsqueeze(1), default_policy(obs), a)
        obs = env.step(a)[0]
        bp = ball.data.root_link_pos_w
        entered |= ((bp[:, 0] - org[:, 0]) <= -0.5) & ((bp[:, 1] - org[:, 1]).abs() <= 1.5) & (bp[:, 2] <= 1.8)
      blocked += int((~entered).sum()); total += N
      usel = torch.where(latched < 0, torch.zeros_like(latched), latched)
      if truereg is not None:
        for i in range(N): rec.append((int(truereg[i]), int(usel[i]), bool(~entered[i])))
    print(f"  batch {b+1}/{cfg.batches}: {blocked}/{total} = {100*blocked/total:.1f}%", flush=True)
  print(f"\nMoE6 OVERALL: {blocked}/{total} = {100*blocked/total:.1f}%")
  for rg in range(6):
    rr = [x for x in rec if x[0] == rg]
    if rr: print(f"  {_REGION[rg]:<11} {100*sum(x[2] for x in rr)/len(rr):.1f}%")
  if rec and not cfg.perfect_gate:
    acc = sum(1 for x in rec if x[0] == x[1]) / len(rec)
    print(f"\n[DIAG] gate accuracy (latched==true): {100*acc:.1f}%")
    print(f"[DIAG] per true-region: n | route%correct | block(routed-correct) | block(misrouted) | misroute targets")
    for rg in range(6):
      rr = [x for x in rec if x[0] == rg]
      if not rr: continue
      cor = [x for x in rr if x[1] == rg]; mis = [x for x in rr if x[1] != rg]
      bc = 100 * sum(x[2] for x in cor) / len(cor) if cor else float("nan")
      bm = 100 * sum(x[2] for x in mis) / len(mis) if mis else float("nan")
      from collections import Counter
      tgt = Counter(_REGION[x[1]] for x in mis).most_common(3)
      tgtstr = ", ".join(f"{k}:{v}" for k, v in tgt)
      print(f"  {_REGION[rg]:<11} n={len(rr):<4} route={100*len(cor)/len(rr):5.1f}%  "
            f"ok={bc:5.1f}%  mis={bm:5.1f}%  -> [{tgtstr}]")
  env.close()


if __name__ == "__main__":
  import mjlab.tasks, src.tasks  # noqa
  main(tyro.cli(Cfg, prog="eval_moe6"))
