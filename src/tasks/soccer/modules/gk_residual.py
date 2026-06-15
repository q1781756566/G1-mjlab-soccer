"""Residual-RL actor: a FROZEN distilled base policy + a small trainable residual.

Every attempt to PPO-fine-tune the distilled policy collapsed it, because PPO's
noisy gradient perturbs the precise, coordinated diving behavior. Residual RL
avoids this: the distilled policy (which dives, ~74%) is frozen and provides the
base action; a separate residual network (zero-initialized, tanh-bounded) adds a
small correction trained with the block reward. At init the residual is ~0 so the
policy == the distilled base; it can only refine timing/reach, never destroy the
diving skill. This is a known-stable way to improve a competent base policy.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable

# Set by the training/eval script before the runner is constructed (the runner
# cfg dataclass can't carry these extra fields).
BASE_CKPT = None
BASE_HIDDEN = (1024, 512, 256)
RESIDUAL_SCALE = 0.3


class GoalkeeperResidual(nn.Module):
  is_recurrent: bool = False

  def __init__(
    self, obs, obs_groups, obs_set, output_dim,
    hidden_dims=(512, 256, 128), activation="elu", obs_normalization=False,
    distribution_cfg=None, base_ckpt=None, base_hidden_dims=(1024, 512, 256),
    residual_scale=0.3, **kwargs,
  ):
    super().__init__()
    self.group = obs_groups[obs_set][0]
    if base_ckpt is None:
      base_ckpt = BASE_CKPT
      base_hidden_dims = BASE_HIDDEN
      residual_scale = RESIDUAL_SCALE
    self.residual_scale = residual_scale
    dcfg = dict(distribution_cfg or {"class_name": "GaussianDistribution", "init_std": 0.4, "std_type": "scalar"})

    # Frozen base = the distilled MLP policy.
    self.base = MLPModel(obs, obs_groups, obs_set, output_dim, base_hidden_dims, activation, False, dict(dcfg))
    if base_ckpt:
      ck = torch.load(base_ckpt, map_location="cpu", weights_only=False)
      self.base.load_state_dict(ck["actor_state_dict"])
      print(f"[GoalkeeperResidual] loaded frozen base from {base_ckpt}")
    for p in self.base.parameters():
      p.requires_grad_(False)
    self.base.eval()

    # Trainable residual (zero-initialized last layer → starts at the base).
    in_dim = obs[self.group].shape[-1]
    self.residual = MLP(in_dim, output_dim, hidden_dims, activation)
    for m in reversed([mod for mod in self.residual.modules() if isinstance(mod, nn.Linear)]):
      nn.init.zeros_(m.weight); nn.init.zeros_(m.bias); break

    # Our own action distribution.
    dist_class: type[Distribution] = resolve_callable(dcfg.pop("class_name"))
    self.distribution: Distribution = dist_class(output_dim, **dcfg)

  def _mean(self, obs):
    with torch.no_grad():
      base = self.base.forward(obs, stochastic_output=False)
    res = self.residual(obs[self.group])
    return base + self.residual_scale * torch.tanh(res)

  def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
    mean = self._mean(obs)
    self.distribution.update(mean)
    return self.distribution.sample() if stochastic_output else self.distribution.deterministic_output(mean)

  # rsl_rl interface
  def reset(self, dones=None, hidden_state=None): pass
  def get_hidden_state(self): return None
  def detach_hidden_state(self, dones=None): pass
  def update_normalization(self, obs): pass
  @property
  def output_mean(self): return self.distribution.mean
  @property
  def output_std(self): return self.distribution.std
  @property
  def output_entropy(self): return self.distribution.entropy
  @property
  def output_distribution_params(self): return self.distribution.params
  def get_output_log_prob(self, outputs): return self.distribution.log_prob(outputs)
  def get_kl_divergence(self, old, new): return self.distribution.kl_divergence(old, new)
