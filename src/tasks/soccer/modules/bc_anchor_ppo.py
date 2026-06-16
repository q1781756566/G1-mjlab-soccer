"""PPO that (1) clamps the action std to a tiny constant so the DETERMINISTIC
mean is what learns, and (2) adds a behavior-cloning ANCHOR pulling the actor
toward the repair-oracle's blocking actions every update — preventing the
drift/forgetting that collapsed prior RL fine-tunes of the distilled goalkeeper.

bc_obs/bc_act (GPU tensors) and bc_coef are set on the instance after the runner
is built (the runner cfg dataclass can't carry them).
"""
from __future__ import annotations

import torch
from rsl_rl.algorithms.ppo import PPO

from src.tasks.soccer.modules.symmetry import mirror_obs, mirror_action


class BCAnchorPPO(PPO):
  std_clamp: float = 0.06
  bc_obs = None
  bc_act = None
  bc_coef: float = 0.5
  bc_steps: int = 1
  bc_batch: int = 8192
  # Left/right symmetry: enforce equivariance pi(o)==mirror_action(pi(mirror_obs(o)))
  # to remove the ~7% left<right asymmetry. 0 disables.
  sym_coef: float = 0.0

  def update(self):
    out = super().update()
    dist = getattr(self.actor, "distribution", None)
    sp = getattr(dist, "std_param", None) if dist is not None else None
    if sp is not None:
      with torch.no_grad():
        sp.clamp_(min=1e-3, max=self.std_clamp)
    if self.bc_obs is not None and self.bc_coef > 0:
      n = self.bc_obs.shape[0]
      for _ in range(self.bc_steps):
        idx = torch.randint(0, n, (self.bc_batch,), device=self.bc_obs.device)
        o = self.bc_obs[idx]
        pred = self.actor({"actor": o}, stochastic_output=False)
        loss = self.bc_coef * torch.nn.functional.smooth_l1_loss(pred, self.bc_act[idx])
        if self.sym_coef > 0:
          # symmetric BC: mirrored obs -> mirrored target action
          loss = loss + self.bc_coef * torch.nn.functional.smooth_l1_loss(
            self.actor({"actor": mirror_obs(o)}, stochastic_output=False),
            mirror_action(self.bc_act[idx]))
          # equivariance consistency pi(o) == mirror(pi(mirror(o)))
          a = self.actor({"actor": o}, stochastic_output=False)
          am = self.actor({"actor": mirror_obs(o)}, stochastic_output=False)
          loss = loss + self.sym_coef * torch.nn.functional.smooth_l1_loss(
            a, mirror_action(am))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)  # prevent blow-up
        self.optimizer.step()
    return out
