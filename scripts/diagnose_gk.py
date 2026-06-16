"""Diagnose the goalkeeper base policy's FAILURE STRUCTURE (vectorized).

Runs a checkpoint over many parallel envs and records, per trial:
  - region (env._gk_region)
  - blocked / conceded (exact eval criterion)
  - crossing point (y,z) of the ball at the keeper plane (x_rel=0)
  - min distance ANY body link reached to the ball over the episode (near-miss)
  - min distance any blocking link reached to the ball's keeper-plane crossing pt

Prints overall + per-region block rate, and for the MISSES the distribution of
near-miss distance and crossing point. This tells us whether failures are
concentrated (a targeted fix works) or diffuse (need a repair oracle).
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import torch, tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper, MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

_REGION_NAMES = ["Right-Mid", "Left-Mid", "Right-Up", "Left-Up", "Right-Low", "Left-Low"]
# Blocking links: hands/forearms/upper-arms, shins/feet, chest/pelvis.
_BLOCK_LINKS = (
  "left_wrist_yaw_link", "right_wrist_yaw_link",
  "left_elbow_link", "right_elbow_link",
  "left_shoulder_roll_link", "right_shoulder_roll_link",
  "left_ankle_roll_link", "right_ankle_roll_link",
  "left_knee_link", "right_knee_link",
  "torso_link", "pelvis",
)


@dataclass
class Cfg:
  checkpoint: str = "src/assets/soccer/weight/goalkeeper_distilled_v3.pt"
  num_envs: int = 256
  batches: int = 8          # total trials = num_envs * batches
  device: str = "cuda:0"
  steps: int = 150
  hidden: tuple[int, ...] = ()   # native checkpoint with custom net hidden_dims
  residual_base: str = ""        # eval a GoalkeeperResidual checkpoint on this frozen base
  residual_scale: float = 1.5
  residual_head: tuple[int, ...] = (512, 256, 128)
  seed: int | None = None        # fix env seed for cross-GPU diff-test (same balls)
  regions: tuple[int, ...] = ()  # restrict ball reset to these regions


def main(cfg: Cfg):
  configure_torch_backends()
  if cfg.seed is not None:
    import torch as _t; _t.manual_seed(cfg.seed)
  env_cfg = load_env_cfg("Eval-Goalkeeper", play=False)
  env_cfg.scene.num_envs = cfg.num_envs
  if cfg.seed is not None:
    env_cfg.seed = cfg.seed
  if "fell_over" in env_cfg.terminations:
    env_cfg.terminations["fell_over"] = None
  if cfg.regions:
    import copy as _c
    rb = env_cfg.events["reset_ball"]; vc = _c.copy(rb.params["vel_cfg"])
    vc.regions = [vc.regions[i] for i in cfg.regions]; rb.params["vel_cfg"] = vc
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device), clip_actions=100.0)

  import os, sys
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  if cfg.residual_base:   # GoalkeeperResidual: frozen base + trained head
    from dataclasses import asdict
    import src.tasks.soccer.modules.gk_residual as gkr
    gkr.BASE_CKPT = None; gkr.BASE_HIDDEN = (1024, 512, 256); gkr.RESIDUAL_SCALE = cfg.residual_scale
    from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
    agent = RslRlOnPolicyRunnerCfg(
      actor=RslRlModelCfg(class_name="src.tasks.soccer.modules.gk_residual.GoalkeeperResidual",
                          hidden_dims=cfg.residual_head, activation="elu", obs_normalization=False,
                          distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.1, "std_type": "scalar"}),
      critic=RslRlModelCfg(hidden_dims=(512, 256, 256), activation="elu", obs_normalization=False),
      algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
                                     entropy_coef=0.0, num_learning_epochs=5, num_mini_batches=4, learning_rate=3e-4,
                                     schedule="adaptive", gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0),
      experiment_name="g1_goalkeeper", save_interval=50, num_steps_per_env=24, max_iterations=1)
    runner = MjlabOnPolicyRunner(env, asdict(agent), device=cfg.device)
    runner.load(cfg.checkpoint, load_cfg={"actor": True})
    policy = runner.get_inference_policy(device=cfg.device)
  elif cfg.hidden:   # native checkpoint with a custom net size
    from dataclasses import asdict
    from src.tasks.soccer.config.g1.gk_train_cfg import goalkeeper_train_runner_cfg
    agent = goalkeeper_train_runner_cfg()
    agent.actor.hidden_dims = tuple(cfg.hidden); agent.critic.hidden_dims = tuple(cfg.hidden)
    runner = MjlabOnPolicyRunner(env, asdict(agent), device=cfg.device)
    runner.load(cfg.checkpoint, load_cfg={"actor": True})
    policy = runner.get_inference_policy(device=cfg.device)
  else:
    from eval_naive_goalkeeper import _load_policy
    policy = _load_policy(cfg.checkpoint, env, cfg.device)

  ball = env.unwrapped.scene["ball"]
  robot = env.unwrapped.scene["robot"]
  org = env.unwrapped.scene.env_origins  # (B,3)
  blk_idx = torch.as_tensor(robot.find_bodies(_BLOCK_LINKS, preserve_order=True)[0], device=cfg.device)

  recs = []  # (region, entered, min_body_dist, cross_y, cross_z, dist_to_cross)
  N = cfg.num_envs
  for b in range(cfg.batches):
    obs = env.reset()
    if isinstance(obs, tuple): obs = obs[0]
    entered = torch.zeros(N, dtype=torch.bool, device=cfg.device)
    min_bd = torch.full((N,), 1e9, device=cfg.device)
    best_xabs = torch.full((N,), 1e9, device=cfg.device)  # closest ball_x_rel to 0 seen
    cross_y = torch.zeros(N, device=cfg.device)
    cross_z = torch.zeros(N, device=cfg.device)
    dist_cross = torch.full((N,), 1e9, device=cfg.device)
    region = env.unwrapped._gk_region.clone() if hasattr(env.unwrapped, "_gk_region") else torch.zeros(N, dtype=torch.long, device=cfg.device)
    for _ in range(cfg.steps):
      with torch.inference_mode():
        a = policy(obs)
      res = env.step(a); obs = res[0]
      bp = ball.data.root_link_pos_w            # (B,3)
      bx = bp[:, 0] - org[:, 0]
      by = bp[:, 1] - org[:, 1]
      # eval block criterion
      ing = (bx <= -0.5) & (by.abs() <= 1.5) & (bp[:, 2] <= 1.8)
      entered |= ing
      # min body-link distance to ball
      body = robot.data.body_link_pos_w[:, blk_idx]   # (B,K,3)
      d = (body - bp.unsqueeze(1)).pow(2).sum(-1).min(1).values.sqrt()
      min_bd = torch.minimum(min_bd, d)
      # record ball (y,z) at keeper plane crossing (closest |bx| to 0)
      closer = bx.abs() < best_xabs
      best_xabs = torch.where(closer, bx.abs(), best_xabs)
      cross_y = torch.where(closer, by, cross_y)
      cross_z = torch.where(closer, bp[:, 2], cross_z)
      dist_cross = torch.where(closer, d, dist_cross)
    for i in range(N):
      recs.append((int(region[i]), bool(entered[i]), float(min_bd[i]),
                   float(cross_y[i]), float(cross_z[i]), float(dist_cross[i])))
    bl = sum(1 for r in recs if not r[1])
    print(f"  batch {b+1}/{cfg.batches}: cumulative block {bl}/{len(recs)} = {100*bl/len(recs):.1f}%", flush=True)
  env.close()

  total = len(recs)
  blocked = sum(1 for r in recs if not r[1])
  print(f"\n{'='*64}\n  OVERALL block rate: {blocked}/{total} = {100*blocked/total:.1f}%\n{'='*64}")
  print(f"  {'region':<11} {'n':>4} {'block%':>7} {'miss_mindist':>13} {'miss_cross(y,z)':>20}")
  for rg in range(6):
    rr = [r for r in recs if r[0] == rg]
    if not rr: continue
    bk = sum(1 for r in rr if not r[1])
    miss = [r for r in rr if r[1]]
    if miss:
      md = sum(r[2] for r in miss)/len(miss)
      cy = sum(r[3] for r in miss)/len(miss)
      cz = sum(r[4] for r in miss)/len(miss)
      ms = f"{md:.3f}m   ({cy:+.2f},{cz:.2f})"
    else:
      ms = "—"
    print(f"  {_REGION_NAMES[rg]:<11} {len(rr):>4} {100*bk/len(rr):>6.1f}% {ms:>34}")
  # near-miss histogram (misses only)
  miss_all = [r for r in recs if r[1]]
  print(f"\n  Misses: {len(miss_all)}. Near-miss min-body-dist distribution:")
  import bisect
  edges = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0, 10.0]
  buckets = [0]*len(edges)
  for r in miss_all:
    buckets[bisect.bisect_left(edges, r[2])] += 1
  lo = 0.0
  for e, c in zip(edges, buckets):
    print(f"    {lo:.2f}-{e:.2f}m: {c}  ({100*c/max(1,len(miss_all)):.0f}%)")
    lo = e


if __name__ == "__main__":
  import mjlab.tasks, src.tasks  # noqa
  main(tyro.cli(Cfg, prog="diagnose_gk"))
