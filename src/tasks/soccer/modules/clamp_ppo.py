"""PPO variant that anneals an upper bound on the policy's action std.

Why: with a coverage-style reward and learned std, vanilla PPO keeps the std
high (~0.6) because random high-variance "flailing" covers the goal and blocks
balls during *stochastic* rollouts — but the *deterministic* policy (the mean,
which is what eval uses) then blocks almost nothing. Capping the std and
annealing it toward ~0 forces the policy to block via its mean, so the
deterministic eval policy actually learns the skill.
"""

from __future__ import annotations

import torch

from rsl_rl.algorithms.ppo import PPO


class ClampStdPPO(PPO):
  # Annealing schedule for the std upper bound (per update / iteration).
  std_max_start: float = 0.5
  std_max_end: float = 0.1
  anneal_updates: int = 2000

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._upd = 0

  def update(self):
    out = super().update()
    self._upd += 1
    frac = min(1.0, self._upd / float(self.anneal_updates))
    std_max = self.std_max_start + (self.std_max_end - self.std_max_start) * frac
    dist = getattr(self.actor, "distribution", None)
    sp = getattr(dist, "std_param", None) if dist is not None else None
    if sp is not None:
      with torch.no_grad():
        sp.clamp_(min=1e-3, max=std_max)
    return out
