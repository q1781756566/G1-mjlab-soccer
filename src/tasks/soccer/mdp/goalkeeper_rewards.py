"""Goalkeeper-specific reward functions for training.

Matches the Humanoid-Goalkeeper paper's reward set (24 terms), prioritized
to 7 goalkeeper-specific functions. The remaining regularization terms
(action_rate, joint_pos_limits, etc.) are reused from existing modules.

State tracking follows the pattern in training_rewards.py: per-environment
tensors stored on the env object via _gk_get_or_init_state, reset via a
dedicated reset event.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.entity import Entity


def _gk_get_or_init_state(
  env: ManagerBasedRlEnv, name: str, default: float, dtype=torch.float32,
) -> torch.Tensor:
  t = getattr(env, name, None)
  if t is None or t.shape[0] != env.num_envs:
    t = torch.full((env.num_envs,), default, dtype=dtype, device=env.device)
    setattr(env, name, t)
  return t


def _resolve_ee_indices(
  robot: Entity, ee_body_names: tuple[str, ...],
) -> torch.Tensor:
  """Look up body indices for named end-effector bodies at runtime."""
  return torch.as_tensor(
    robot.find_bodies(ee_body_names, preserve_order=True)[0],
    device=robot.data.body_link_pos_w.device,
  )


# -- Reset event ---------------------------------------------------------------


def _reset_gk_state(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
) -> None:
  """Reset per-environment GK tracking state on episode start."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  for key in ("_gk_max_ball_speed", "_gk_block_awarded"):
    t = getattr(env, key, None)
    if t is not None and t.shape[0] == env.num_envs:
      t[env_ids] = 0.0 if "speed" in key else False


# -- Reward functions ----------------------------------------------------------


def goalkeeper_ee_reach(
  env: ManagerBasedRlEnv,
  std: float = 0.3,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ee_body_names: tuple[str, ...] = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  ),
) -> torch.Tensor:
  """End-effector reach reward: exponential distance to ball.

  Paper's 'eereach' — encourages hands and feet to reach toward the ball.
  Rewards proximity of 4 end-effectors (both hands + both feet) to the ball
  using an exponential decay: sum_i exp(-||ee_i - ball||² / σ²).

  Weight: 10.0 (matches paper).
  """
  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]

  ball_pos = ball.data.root_link_pos_w  # (B, 3)
  ee_indices = _resolve_ee_indices(robot, ee_body_names)
  ee_pos = robot.data.body_link_pos_w[:, ee_indices]  # (B, N, 3)

  delta = ee_pos - ball_pos.unsqueeze(1)  # (B, N, 3)
  dist_sq = torch.sum(delta * delta, dim=-1)  # (B, N)
  reward_per_ee = torch.exp(-dist_sq / (std * std))  # (B, N)
  return reward_per_ee.sum(dim=-1)  # (B,)


def goalkeeper_stop_ball(
  env: ManagerBasedRlEnv,
  velocity_drop_threshold: float = 2.0,
  behind_robot_x_threshold: float = 0.0,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """One-shot block reward when ball velocity drops significantly.

  Paper's combined 'stopball' + 'success' — tracks per-environment max ball
  speed and awards 1.0 when:
    1. max_speed - current_speed > velocity_drop_threshold (2.0 m/s), AND
    2. ball has passed behind the robot (ball_x > robot_x + threshold).

  The reward is awarded at most once per episode per environment.

  Weight: 100.0 (matches paper's stopball=100).
  """
  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]

  ball_pos = ball.data.root_link_pos_w  # (B, 3)
  ball_vel = ball.data.root_link_lin_vel_w  # (B, 3)
  robot_pos = robot.data.root_link_pos_w  # (B, 3)

  current_speed = torch.norm(ball_vel, dim=-1)  # (B,)

  max_speed = _gk_get_or_init_state(env, "_gk_max_ball_speed", 0.0)
  block_awarded = _gk_get_or_init_state(env, "_gk_block_awarded", 0.0)

  # Update per-env max speed.
  max_speed = torch.maximum(max_speed, current_speed)
  setattr(env, "_gk_max_ball_speed", max_speed)

  speed_drop = (max_speed - current_speed) > velocity_drop_threshold
  ball_behind = ball_pos[:, 0] > (robot_pos[:, 0] + behind_robot_x_threshold)
  block_detected = speed_drop & ball_behind & (block_awarded < 0.5)

  reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  if torch.any(block_detected):
    ids = torch.nonzero(block_detected, as_tuple=False).squeeze(-1)
    reward[ids] = 1.0
    block_awarded[ids] = 1.0
    setattr(env, "_gk_block_awarded", block_awarded)

  return reward


def goalkeeper_stay_on_line(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize lateral deviation from goal center.

  Paper's 'stayonline' — discourages the robot from drifting sideways
  away from the center of the goal line.

  Weight: -2.0 (matches paper).
  """
  robot: Entity = env.scene[robot_cfg.name]
  robot_y = robot.data.root_link_pos_w[:, 1]
  return torch.abs(robot_y)


def goalkeeper_no_retreat(
  env: ManagerBasedRlEnv,
  goal_line_x: float = 0.0,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize retreating behind the goal line.

  Paper's 'noretreat' — discourages the robot from moving backward
  (increasing x) past the goal line. In our coordinate system, the
  robot faces -y with yaw=π; the ball comes from -x toward +x.

  Weight: -2.0 (matches paper).
  """
  robot: Entity = env.scene[robot_cfg.name]
  robot_x = robot.data.root_link_pos_w[:, 0]
  retreat = torch.clamp(robot_x - goal_line_x, min=0.0)
  return retreat * retreat


def goalkeeper_feet_slippage(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  foot_body_names: tuple[str, ...] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  ),
  contact_sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
  """Penalize foot sliding when feet are in ground contact.

  Paper's 'feet_slippage' — returns the sum of squared xy-velocities
  for feet that are currently touching the ground. This discourages
  slipping while moving laterally.

  Weight: -3.0 (exp(-10*vel) → direct penalty).
  """
  robot: Entity = env.scene[robot_cfg.name]
  foot_indices = _resolve_ee_indices(robot, foot_body_names)
  foot_vel_w = robot.data.body_link_lin_vel_w[:, foot_indices]  # (B, 2, 3)
  foot_vel_xy_sq = torch.sum(foot_vel_w[:, :, :2] ** 2, dim=-1)  # (B, 2)

  # Check foot-ground contact via sensor.
  # feet_ground_contact uses reduce="netforce", single slot.
  contact = env.scene.sensors.get(contact_sensor_name, None)
  if contact is not None:
    found = contact.data.force[:, 0]  # (B, 3) — net force on feet
    contact_found = torch.norm(found, dim=-1) > 1.0  # (B,)
  else:
    contact_found = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

  slip = foot_vel_xy_sq.sum(dim=-1)  # (B,)
  return slip * contact_found.float()


def goalkeeper_posture_orientation(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Reward upright posture via projected gravity.

  Paper's 'postorientation' — returns projected_gravity[:, 2] clamped
  to [0, 1]. When the robot is perfectly upright (gravity points along
  -z in local frame), projected_gravity_z ≈ 1.0. As the robot tilts,
  this value decreases.

  Weight: 3.0 (matches paper).
  """
  robot: Entity = env.scene[robot_cfg.name]
  grav_z = robot.data.projected_gravity_b[:, 2]
  return torch.clamp(grav_z, 0.0, 1.0)


def goalkeeper_ang_vel_xy(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize base angular velocity in the xy-plane.

  Paper's 'ang_vel_xy' — high angular velocity in pitch/roll axes
  indicates instability during reactive movements.

  Weight: -0.1 (matches paper).
  """
  robot: Entity = env.scene[robot_cfg.name]
  ang_vel = robot.data.root_link_ang_vel_b[:, :2]  # (B, 2)
  return torch.sum(ang_vel * ang_vel, dim=-1)  # (B,)
