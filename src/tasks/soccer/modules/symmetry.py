"""Left/right (sagittal y->-y) mirror map for the G1 goalkeeper.

The eval shows a consistent ~7% left<right asymmetry inherited from the teacher.
The task is perfectly left/right symmetric, so we can enforce policy equivariance
  pi(o) == mirror_action( pi(mirror_obs(o)) )
to remove the asymmetry. The joint sign/permutation below is VALIDATED against the
robot's forward kinematics in scripts/gk_symmetry.py (max link error 0.01 mm).

Actor obs (960) is TERM-major, each term = 10 history frames:
  ball_pos   0:30   (10x3)
  ang_vel    30:60  (10x3)
  proj_grav  60:90  (10x3)
  joint_pos  90:380 (10x29)
  joint_vel  380:670(10x29)
  actions    670:960(10x29)
"""
from __future__ import annotations
import torch

# left<->right joint permutation (29 DoF)
JOINT_PERM = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5,
              12, 13, 14,
              22, 23, 24, 25, 26, 27, 28,
              15, 16, 17, 18, 19, 20, 21]
# sign flip under the mirror: roll(x)/yaw(z) joints flip, pitch(y)/knee/elbow keep
_FLIP = {1, 2, 5, 7, 8, 11, 12, 13, 16, 17, 19, 21, 23, 24, 26, 28}
JOINT_SIGN = [(-1.0 if i in _FLIP else 1.0) for i in range(29)]

# per-frame 3-vector mirrors
_VEC_BALL = [1.0, -1.0, 1.0]    # position vector: flip y
_VEC_ANGV = [-1.0, 1.0, -1.0]   # angular-velocity pseudovector
_VEC_GRAV = [1.0, -1.0, 1.0]    # gravity direction vector

_cache: dict = {}


def _t(dev):
  if dev not in _cache:
    _cache[dev] = (
      torch.tensor(JOINT_PERM, device=dev),
      torch.tensor(JOINT_SIGN, device=dev),
      torch.tensor(_VEC_BALL, device=dev),
      torch.tensor(_VEC_ANGV, device=dev),
      torch.tensor(_VEC_GRAV, device=dev),
    )
  return _cache[dev]


def mirror_action(a: torch.Tensor) -> torch.Tensor:
  perm, sign, *_ = _t(a.device)
  return a[..., perm] * sign


def mirror_obs(obs: torch.Tensor) -> torch.Tensor:
  """obs: (..., 960), TERM-major (each term = 10 history frames)."""
  perm, sign, vb, va, vg = _t(obs.device)
  B = obs.shape[:-1]
  o = obs.clone()

  def vec(lo, hi, v):
    o[..., lo:hi] = (o[..., lo:hi].reshape(*B, 10, 3) * v).reshape(*B, hi - lo)

  def jnt(lo, hi):
    o[..., lo:hi] = (o[..., lo:hi].reshape(*B, 10, 29)[..., perm] * sign).reshape(*B, hi - lo)

  vec(0, 30, vb)       # ball_pos
  vec(30, 60, va)      # base_ang_vel
  vec(60, 90, vg)      # projected_gravity
  jnt(90, 380)         # joint_pos
  jnt(380, 670)        # joint_vel
  jnt(670, 960)        # actions
  return o
