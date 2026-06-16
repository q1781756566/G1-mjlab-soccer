"""Validate the G1 goalkeeper left/right mirror map (src/.../modules/symmetry.py)
against (a) the robot's forward kinematics and (b) a real observation vector."""
from __future__ import annotations
import torch
from src.tasks.soccer.modules.symmetry import (
  JOINT_PERM, JOINT_SIGN, mirror_obs, mirror_action)


def validate():
  import mjlab.tasks, src.tasks  # noqa
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg
  from mjlab.utils.torch import configure_torch_backends
  configure_torch_backends()
  dev = "cuda:0"
  c = load_env_cfg("Eval-Goalkeeper", play=False); c.scene.num_envs = 4
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=c, device=dev), clip_actions=100.0)
  r = env.unwrapped.scene["robot"]
  perm = torch.tensor(JOINT_PERM, device=dev); sign = torch.tensor(JOINT_SIGN, device=dev)

  # (a) KINEMATIC: mirrored qpos => geometrically mirrored pose
  names = r.body_names
  pairs = [("left_wrist_yaw_link", "right_wrist_yaw_link"),
           ("left_ankle_roll_link", "right_ankle_roll_link"),
           ("left_knee_link", "right_knee_link")]
  idx = {n: names.index(n) for p in pairs for n in p}
  torch.manual_seed(0); worst = 0.0
  for _ in range(4):
    q = (torch.rand(29, device=dev) - 0.5) * 1.2
    qm = q[perm] * sign
    jp = torch.stack([q, qm, q, qm], 0)
    r.write_joint_state_to_sim(jp, torch.zeros_like(jp)); env.unwrapped.sim.forward()
    P = r.data.body_link_pos_w.clone(); base = env.unwrapped.scene.env_origins
    flip = torch.tensor([1.0, -1.0, 1.0], device=dev)
    for (ln, rn) in pairs:
      a = P[1, idx[ln]] - base[1]
      b = (P[0, idx[rn]] - base[0]) * flip
      worst = max(worst, (a - b).norm().item())
  print(f"(a) kinematic max link error = {worst*1000:.2f} mm  {'OK' if worst<0.01 else 'BAD'}")

  # (b) OBS: run a few steps so the ball is off-centre and the robot has moved,
  # then check mirror_obs structure on a real observation.
  obs, _ = env.reset()
  for _ in range(15):
    obs = env.step(torch.zeros(4, 29, device=dev))[0]
  o = obs["actor"]                       # (4, 960)
  om = mirror_obs(o)
  inv = (mirror_obs(om) - o).abs().max().item()          # involution
  # ball_pos frame0 = o[:,0:3]; y must flip, x/z unchanged
  by = (om[:, 0:3] - o[:, 0:3] * torch.tensor([1.0, -1, 1], device=dev)).abs().max().item()
  # joint_pos frame0 = o[:,90:119]; mirrored = perm+sign
  jp0 = o[:, 90:119]; jp0m = jp0[:, perm] * sign
  jerr = (om[:, 90:119] - jp0m).abs().max().item()
  print(f"(b) involution err = {inv:.2e}   ball-y-flip err = {by:.2e}   joint_pos err = {jerr:.2e}")
  print("OBS MIRROR", "OK" if max(inv, by, jerr) < 1e-5 else "BAD")
  env.close()


if __name__ == "__main__":
  validate()
