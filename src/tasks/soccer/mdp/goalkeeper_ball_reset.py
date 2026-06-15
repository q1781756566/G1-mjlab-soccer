"""Soccer-specific ball reset functions.

Includes:
- Legacy velocity-based launch (GoalkeeperBallVelCfg + reset_ball_with_goal_velocity)
- Parabolic trajectory model (RegionBallVelCfg + reset_ball_with_parabolic_trajectory)
  matching the Humanoid-Goalkeeper paper's assign_ball_states approach.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


# ---------------------------------------------------------------------------
# Legacy: velocity-based ball launch (simple shooter-style)
# ---------------------------------------------------------------------------


@dataclass
class GoalkeeperBallVelCfg:
  """Configuration for randomizing goalkeeper ball velocity (legacy).

  The ball is launched from the penalty spot toward the goal area.
  Velocity is randomized in speed, yaw direction, and pitch angle.
  """

  speed_min: float = 2.0
  speed_max: float = 4.5
  yaw_spread_deg: float = 18.0
  pitch_min_deg: float = 3.0
  pitch_max_deg: float = 8.0


def reset_ball_with_goal_velocity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  vel_cfg: GoalkeeperBallVelCfg,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> None:
  """Reset ball position and apply a randomized velocity toward goal (legacy)."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  asset: Entity = env.scene[ball_cfg.name]
  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  root_states = default_root_state[env_ids].clone()

  positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]

  speeds = (
    vel_cfg.speed_min
    + torch.rand(len(env_ids), device=env.device)
    * (vel_cfg.speed_max - vel_cfg.speed_min)
  )

  spread_rad = math.radians(vel_cfg.yaw_spread_deg)
  yaws = (torch.rand(len(env_ids), device=env.device) * 2 - 1) * spread_rad

  pitch_min_rad = math.radians(vel_cfg.pitch_min_deg)
  pitch_max_rad = math.radians(vel_cfg.pitch_max_deg)
  pitches = (
    pitch_min_rad
    + torch.rand(len(env_ids), device=env.device)
    * (pitch_max_rad - pitch_min_rad)
  )

  vx = speeds * torch.cos(pitches) * torch.cos(yaws)
  vy = speeds * torch.cos(pitches) * torch.sin(yaws)
  vz = speeds * torch.sin(pitches)

  ang_vel = torch.zeros(len(env_ids), 3, device=env.device)
  velocities = torch.cat(
    [torch.stack([vx, vy, vz], dim=-1), ang_vel], dim=-1
  )

  orientations = root_states[:, 3:7]

  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


# ---------------------------------------------------------------------------
# Parabolic trajectory model (6-region, matching Humanoid-GK paper)
# ---------------------------------------------------------------------------


@dataclass
class RegionBallVelCfg:
  """Configuration for region-conditioned parabolic ball trajectory.

  Matches the Humanoid-Goalkeeper paper's assign_ball_states approach:
  the ball is launched from a random start position and aimed at a
  specific landing point within a randomly chosen region behind the robot.

  Attributes:
    ball_start_x_range: Range of ball start distances in front of robot (m).
    ball_end_x_range: Range of ball end distances behind robot (m).
    t_flight_range: Ball flight duration range (s).
    regions: List of region definitions (height/width ranges in goal frame).
    ball_start_y_range: Init y range (computed from region y boundaries).
    ball_start_z_range: Init z range (computed from region z boundaries).
  """

  ball_start_x_range: tuple[float, float] = (3.0, 5.0)
  ball_end_x_range: tuple[float, float] = (0.1, 0.6)
  t_flight_range: tuple[float, float] = (0.5, 1.0)
  regions: list = None
  ball_start_y_range: tuple[float, float] = (-1.5, 1.5)
  ball_start_z_range: tuple[float, float] = (0.1, 1.8)

  # -- Difficulty curriculum (training only; 0 disables → always full range) --
  # The lateral spread (y) and vertical offset (z) of the ball's landing target
  # are scaled by a difficulty d in [difficulty_min, 1]. Early in training d is
  # small, so balls land near the keeper (y≈0, z≈block_center_z) and are easy to
  # block; d ramps to 1 over `curriculum_warmup_calls` reset calls, expanding to
  # the full 6-region range. Eval leaves curriculum_warmup_calls=0 ⇒ d=1 always.
  curriculum_warmup_calls: int = 0
  difficulty_min: float = 0.25
  # Flight-time (speed) curriculum: early training multiplies t_flight by
  # t_flight_slow_factor (slower balls → more time for the keeper to reach even
  # far crossing points → it learns the full reaching/diving skill), annealed to
  # ×1 (eval speed) over speed_warmup_calls steps. 0 disables (eval speed always).
  speed_warmup_calls: int = 0
  t_flight_slow_factor: float = 2.0
  block_center_z: float = 0.9  # comfortable hand height the keeper starts from

  def __post_init__(self):
    if self.regions is None:
      self.regions = []

  @property
  def num_regions(self) -> int:
    return len(self.regions)


def reset_ball_with_parabolic_trajectory(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  vel_cfg: RegionBallVelCfg,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> None:
  """Reset ball with a region-conditioned parabolic trajectory toward the robot.

  Matches the Humanoid-Goalkeeper paper's assign_ball_states:
  1. Randomly pick a landing region (uniform among k=6).
  2. Sample a random ball start position in front of the robot.
  3. Sample a random ball end position within the chosen region (behind robot).
  4. Compute launch velocity from the parabolic model:
       v_xy = delta_xy / t_flight
       v_z  = (delta_z + 0.5 * g * t_flight²) / t_flight
  5. Write position and velocity to sim.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  n = len(env_ids)
  device = env.device

  # -- Forced-scenario path (repair oracle) -----------------------------------
  # If env._gk_forced = {"start":(B,3) world, "vel":(B,3), "region":(B,)} is set,
  # write those balls directly THROUGH the normal reset so the observation
  # history is filled correctly (no stale-frame transient). Used to replay exact
  # scenarios for CEM search and eval-matched demonstration collection.
  forced = getattr(env, "_gk_forced", None)
  if forced is not None:
    asset: Entity = env.scene[ball_cfg.name]
    drs = asset.data.default_root_state
    quat = drs[env_ids, 3:7].clone()
    pos = forced["start"][env_ids]
    vel = torch.cat([forced["vel"][env_ids], torch.zeros(n, 3, device=device)], dim=-1)
    asset.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(vel, env_ids=env_ids)
    for key, val in (("_gk_region", forced["region"]),
                     ("_gk_ball_start_x", forced["start"][:, 0] - env.scene.env_origins[:, 0])):
      t = getattr(env, key, None)
      if t is None or t.shape[0] != env.num_envs:
        t = torch.zeros(env.num_envs, dtype=torch.float32, device=device)
      t[env_ids] = val[env_ids].float(); setattr(env, key, t)
    return

  num_regions = vel_cfg.num_regions
  assert num_regions > 0, "RegionBallVelCfg must have at least one region."

  # Pick a random region for each env.
  region_idx = torch.randint(0, num_regions, (n,), device=device)

  # -- Curriculum difficulty (training only) ----------------------------------
  warmup = getattr(vel_cfg, "curriculum_warmup_calls", 0)
  if warmup and warmup > 0:
    # Prefer the env-step counter (one tick per control step) for a schedule that
    # is independent of how often resets fire; fall back to a reset-call counter.
    step = getattr(env, "common_step_counter", None)
    if step is None:
      step = int(getattr(env, "_gk_curr_calls", 0)) + 1
      setattr(env, "_gk_curr_calls", step)
    progress = min(1.0, float(step) / float(warmup))
    difficulty = vel_cfg.difficulty_min + (1.0 - vel_cfg.difficulty_min) * progress
  else:
    difficulty = 1.0
  z_center = vel_cfg.block_center_z

  # Sample ball start position in local frame.
  # G1 faces +x (yaw=0). Ball comes from +x (front), lateral variation in y.
  start_x = (                          # front (+x)
    vel_cfg.ball_start_x_range[0]
    + torch.rand(n, device=device)
    * (vel_cfg.ball_start_x_range[1] - vel_cfg.ball_start_x_range[0])
  )
  start_y = (
    vel_cfg.ball_start_y_range[0]    # lateral
    + torch.rand(n, device=device)
    * (vel_cfg.ball_start_y_range[1] - vel_cfg.ball_start_y_range[0])
  )
  start_z = (
    vel_cfg.ball_start_z_range[0]
    + torch.rand(n, device=device)
    * (vel_cfg.ball_start_z_range[1] - vel_cfg.ball_start_z_range[0])
  )
  # Curriculum: pull the lateral spread toward center and the height toward the
  # keeper's comfortable block height when difficulty < 1 (easy early balls).
  start_y = start_y * difficulty
  start_z = z_center + (start_z - z_center) * difficulty
  ball_start_local = torch.stack([start_x, start_y, start_z], dim=-1)

  # Sample ball end within chosen region (behind robot, -y).
  region_height_low = torch.tensor(
    [vel_cfg.regions[i.item()]["height"][0] for i in region_idx],
    device=device, dtype=torch.float32,
  )
  region_height_high = torch.tensor(
    [vel_cfg.regions[i.item()]["height"][1] for i in region_idx],
    device=device, dtype=torch.float32,
  )
  region_width_low = torch.tensor(
    [vel_cfg.regions[i.item()]["width"][0] for i in region_idx],
    device=device, dtype=torch.float32,
  )
  region_width_high = torch.tensor(
    [vel_cfg.regions[i.item()]["width"][1] for i in region_idx],
    device=device, dtype=torch.float32,
  )

  # Behind robot (-x). Width (y) from region, height (z) from region.
  end_x = -(
    vel_cfg.ball_end_x_range[0]
    + torch.rand(n, device=device)
    * (vel_cfg.ball_end_x_range[1] - vel_cfg.ball_end_x_range[0])
  )
  end_y = region_width_low + torch.rand(n, device=device) * (
    region_width_high - region_width_low
  )
  end_z = region_height_low + torch.rand(n, device=device) * (
    region_height_high - region_height_low
  )
  # Curriculum scaling (see start position above).
  end_y = end_y * difficulty
  end_z = z_center + (end_z - z_center) * difficulty
  ball_end_local = torch.stack([end_x, end_y, end_z], dim=-1)

  # Convert to world frame (env_origins at robot base x=0).
  ball_start_w = ball_start_local + env.scene.env_origins[env_ids]
  ball_end_w = ball_end_local + env.scene.env_origins[env_ids]

  # Sample flight time.
  t_flight = (
    vel_cfg.t_flight_range[0]
    + torch.rand(n, device=device)
    * (vel_cfg.t_flight_range[1] - vel_cfg.t_flight_range[0])
  )

  # Flight-time (speed) curriculum: slower balls early, eval-speed later.
  speed_warmup = getattr(vel_cfg, "speed_warmup_calls", 0)
  if speed_warmup and speed_warmup > 0:
    step = getattr(env, "common_step_counter", None)
    if step is None:
      step = int(getattr(env, "_gk_speed_calls", 0)) + 1
      setattr(env, "_gk_speed_calls", step)
    sp = min(1.0, float(step) / float(speed_warmup))
    slow = vel_cfg.t_flight_slow_factor * (1.0 - sp) + 1.0 * sp
    t_flight = t_flight * slow

  # Parabolic trajectory velocity computation.
  delta_pos = ball_end_w - ball_start_w
  g = 9.81
  ball_vel_xy = delta_pos[:, :2] / t_flight.unsqueeze(-1)
  ball_vel_z = (delta_pos[:, 2] + 0.5 * g * t_flight ** 2) / t_flight
  ball_vel = torch.cat(
    [ball_vel_xy, ball_vel_z.unsqueeze(-1)], dim=-1
  )

  # Store the chosen region per environment for privileged critic observations.
  t = getattr(env, "_gk_region", None)
  if t is None or t.shape[0] != env.num_envs:
    t = torch.zeros(env.num_envs, dtype=torch.float32, device=device)
    setattr(env, "_gk_region", t)
  t[env_ids] = region_idx.float()
  setattr(env, "_gk_region", t)

  # Store the ball's start x (env-local) per env, for ball-synchronized motion phase.
  sx = getattr(env, "_gk_ball_start_x", None)
  if sx is None or sx.shape[0] != env.num_envs:
    sx = torch.full((env.num_envs,), 4.0, dtype=torch.float32, device=device)
    setattr(env, "_gk_ball_start_x", sx)
  sx[env_ids] = start_x
  setattr(env, "_gk_ball_start_x", sx)

  # Ball orientation unchanged from default.
  asset: Entity = env.scene[ball_cfg.name]
  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  root_states = default_root_state[env_ids].clone()
  orientations = root_states[:, 3:7]

  # Full velocity: 6-DOF (lin_vel + ang_vel). No initial spin.
  ang_vel = torch.zeros(n, 3, device=device)
  velocities = torch.cat([ball_vel, ang_vel], dim=-1)

  asset.write_root_link_pose_to_sim(
    torch.cat([ball_start_w, orientations], dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)
