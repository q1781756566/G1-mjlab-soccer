"""Goalkeeper ActorCritic — matches Humanoid-Goalkeeper paper architecture.

Exact replica of the reference rsl_rl ActorCritic designed for HIMPPO.
The model takes 960D history-stacked actor observations and produces:
  - 29D action means (via history encoder + ball/region estimators → actor MLP)
  - 1D value (critic MLP on 113D privileged obs)

Architecture:
  history_encoder: 960 → 128 → 64 → 16  (ReLU)
  ball_estimator:  960 → 128 → 32 → 6   (ReLU)
  region_estimator: 960 → 128 → 32 → 6  (ReLU)
  actor:  119 → 512 → 256 → 256 → 29    (ELU)
  critic: 113 → 512 → 256 → 256 → 1     (ELU)
  std: (29,) learnable scalar
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


def _get_activation(act_name: str):
  if act_name == "elu":
    return nn.ELU()
  elif act_name == "relu":
    return nn.ReLU()
  elif act_name == "selu":
    return nn.SELU()
  elif act_name == "lrelu":
    return nn.LeakyReLU()
  elif act_name == "tanh":
    return nn.Tanh()
  elif act_name == "sigmoid":
    return nn.Sigmoid()
  else:
    return nn.ELU()


class GoalkeeperActorCritic(nn.Module):
  """Actor-Critic with history encoder, ball/region estimators.

  Designed to load reference Humanoid-Goalkeeper checkpoints directly.
  Compatible with RSL-RL's model interface (MLPModel-like).

  Observation history reordering:
    mjlab's ObservationManager stacks history as term-major:
      [ball_f0..ball_f9, ang_vel_f0..ang_vel_f9, ..., actions_f0..actions_f9]
    The pretrained reference model was trained with frame-major:
      [f0(96D), f1(96D), ..., f9(96D)]
    We transpose on-the-fly so the model always receives frame-major input.
  """

  is_recurrent = False

  # Term sizes and history length — used by _reorder_obs_history.
  _TERM_SIZES: tuple[int, ...] = (3, 3, 3, 29, 29, 29)
  _HISTORY_LEN: int = 10
  _ONE_STEP_DIM: int = sum(_TERM_SIZES)  # 96

  def __init__(
    self,
    obs,
    obs_groups,
    group_name,
    num_actions=29,
    num_one_step_obs=96,
    num_critic_obs=113,
    num_actor_obs=960,
    actor_history_length=10,
    actor_hidden_dims=(512, 256, 256),
    critic_hidden_dims=(512, 256, 256),
    activation="elu",
    init_noise_std=1.0,
    **kwargs,
  ):
    if kwargs:
      print(
        "GoalkeeperActorCritic.__init__ got unexpected arguments: "
        + str([key for key in kwargs.keys()])
      )
    super().__init__()

    # Try to infer dimensions from obs space if passed.
    if hasattr(obs, "spaces"):
      actor_space = obs.spaces.get("actor")
      critic_space = obs.spaces.get("critic")
      if actor_space is not None and hasattr(actor_space, "shape"):
        num_actor_obs = actor_space.shape[0]
        num_one_step_obs = num_actor_obs // actor_history_length
      if critic_space is not None and hasattr(critic_space, "shape"):
        num_critic_obs = critic_space.shape[0]
    elif isinstance(obs, dict):
      actor_space = obs.get("actor")
      critic_space = obs.get("critic")
      if actor_space is not None and hasattr(actor_space, "shape"):
        num_actor_obs = actor_space.shape[0]
        num_one_step_obs = num_actor_obs // actor_history_length
      if critic_space is not None and hasattr(critic_space, "shape"):
        num_critic_obs = critic_space.shape[0]

    self.num_actor_obs = num_actor_obs
    self.num_critic_obs = num_critic_obs
    self.num_one_step_obs = num_one_step_obs
    self.actor_history_length = actor_history_length
    self.num_actions = num_actions
    self.history_latent_dim = 16
    self.estimate_ball_dim = 6
    self.num_regions = 6

    # History encoder: 960 → 128 → 64 → 16  (ReLU)
    history_input_dim = num_one_step_obs * actor_history_length
    self.history_encoder = nn.Sequential(
      nn.Linear(history_input_dim, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, self.history_latent_dim),
    )

    # Ball estimator: 960 → 128 → 32 → 6  (ReLU)
    self.ball_estimator = nn.Sequential(
      nn.Linear(history_input_dim, 128),
      nn.ReLU(),
      nn.Linear(128, 32),
      nn.ReLU(),
      nn.Linear(32, self.estimate_ball_dim),
    )

    # Region estimator: 960 → 128 → 32 → 6  (ReLU)
    self.region_estimator = nn.Sequential(
      nn.Linear(history_input_dim, 128),
      nn.ReLU(),
      nn.Linear(128, 32),
      nn.ReLU(),
      nn.Linear(32, self.num_regions),
    )

    # Actor: 119 → 512 → 256 → 256 → 29  (ELU)
    act_fn = _get_activation(activation)
    mlp_input_dim_a = (
      num_one_step_obs + self.history_latent_dim + self.estimate_ball_dim + 1
    )
    self.num_actor_input = mlp_input_dim_a

    actor_layers = []
    actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
    actor_layers.append(act_fn)
    for l in range(len(actor_hidden_dims)):
      if l == len(actor_hidden_dims) - 1:
        actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
      else:
        actor_layers.append(
          nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1])
        )
        actor_layers.append(act_fn)
    self.actor = nn.Sequential(*actor_layers)

    # Critic: 113 → 512 → 256 → 256 → 1  (ELU)
    mlp_input_dim_c = num_critic_obs
    critic_layers = []
    critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
    critic_layers.append(act_fn)
    for l in range(len(critic_hidden_dims)):
      if l == len(critic_hidden_dims) - 1:
        critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
      else:
        critic_layers.append(
          nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1])
        )
        critic_layers.append(act_fn)
    self.critic = nn.Sequential(*critic_layers)

    # Learned action std.
    self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
    self.distribution = None
    self.estimate_ball = None
    self.estimate_region = None
    Normal.set_default_validate_args = False

    print(f"Actor MLP: {self.actor}")
    print(f"Critic MLP: {self.critic}")
    print(f"History MLP: {self.history_encoder}")
    print(f"Ball MLP: {self.ball_estimator}")
    print(f"Region MLP: {self.region_estimator}")

  def reset(self, dones=None):
    pass

  def _extract_tensor(self, obs, group="actor"):
    """Extract tensor from observation dict/TensorDict for given group.

    Handles plain dicts, tuples (obs, critic_obs), and TensorDict.
    """
    if isinstance(obs, (tuple, list)):
      return self._extract_tensor(obs[0], group)
    if isinstance(obs, dict):
      x = obs.get(group, obs)
    elif hasattr(obs, "get") and group in obs:
      x = obs[group]
    else:
      x = obs
    if hasattr(x, "get") and not isinstance(x, torch.Tensor):
      x = x[group] if group in x else x
    if isinstance(x, torch.Tensor) and x.dim() == 1:
      x = x.unsqueeze(0)
    return x

  def _reorder_obs_history(self, obs_history: torch.Tensor) -> torch.Tensor:
    """Transpose term-major 960D → frame-major 960D.

    mjlab history stacking (term-major):
      [ball_f0..ball_f9, ang_f0..ang_f9, grav_f0..grav_f9,
       jpos_f0..jpos_f9, jvel_f0..jvel_f9, actions_f0..actions_f9]

    Reference model expects (frame-major):
      [f0(ball,ang,grav,jpos,jvel,act), ..., f9(...)]
    """
    B = obs_history.shape[0]
    chunks = []
    offset = 0
    for sz in self._TERM_SIZES:
      block = obs_history[:, offset : offset + self._HISTORY_LEN * sz]
      chunks.append(block.view(B, self._HISTORY_LEN, sz))
      offset += self._HISTORY_LEN * sz
    # (B, 10, 3+3+3+29+29+29) = (B, 10, 96) → (B, 960)
    return torch.cat(chunks, dim=-1).reshape(B, -1)

  def forward(self, obs, **kwargs):
    return self.act_inference(self._extract_tensor(obs), **kwargs)

  @property
  def action_mean(self):
    return self.distribution.mean

  @property
  def action_std(self):
    return self.distribution.stddev

  @property
  def entropy(self):
    return self.distribution.entropy().sum(dim=-1)

  def update_distribution(self, obs_history):
    obs_history = self._reorder_obs_history(obs_history)
    history_latent = self.history_encoder(obs_history)
    self.estimate_ball = self.ball_estimator(obs_history)
    self.estimate_region = self.region_estimator(obs_history)
    actor_input = torch.cat(
      (
        obs_history[:, -self.num_one_step_obs :],
        history_latent,
        self.estimate_ball,
        torch.argmax(self.estimate_region, dim=-1, keepdim=True),
      ),
      dim=-1,
    )
    action_mean = self.actor(actor_input)
    self.distribution = Normal(action_mean, action_mean * 0.0 + self.std)

  def act(self, obs_history=None, **kwargs):
    self.update_distribution(obs_history)
    return self.distribution.sample(), self.estimate_ball, self.estimate_region

  def get_actions_log_prob(self, actions):
    return self.distribution.log_prob(actions).sum(dim=-1)

  def act_inference(self, obs_history, observations=None):
    """Deterministic inference — used at eval/play time."""
    obs_history = self._reorder_obs_history(obs_history)
    history_latent = self.history_encoder(obs_history)
    estimate_ball = self.ball_estimator(obs_history)
    estimate_region = self.region_estimator(obs_history)
    actor_input = torch.cat(
      (
        obs_history[:, -self.num_one_step_obs :],
        history_latent,
        estimate_ball,
        torch.argmax(estimate_region, dim=-1, keepdim=True),
      ),
      dim=-1,
    )
    return self.actor(actor_input)

  def evaluate(self, critic_observations, **kwargs):
    x = self._extract_tensor(critic_observations, group="critic")
    return self.critic(x)
