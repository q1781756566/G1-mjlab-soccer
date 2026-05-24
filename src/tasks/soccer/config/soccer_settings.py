"""Load soccer settings from YAML and provide typed access."""

from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import get_type_hints, get_origin, get_args

import yaml


@dataclass
class AirDragSettings:
  enabled: bool = True
  fluid_coef: float = 0.10


@dataclass
class BallSettings:
  radius: float = 0.10
  mass: float = 0.35
  inertia: list[float] = field(
    default_factory=lambda: [0.0014, 0.0014, 0.0014, 0, 0, 0]
  )
  air_drag: AirDragSettings = field(default_factory=AirDragSettings)


@dataclass
class GoalSettings:
  width: float = 3.0
  height: float = 1.8


@dataclass
class PenaltySpotSettings:
  distance_from_goal: float = 4.0


@dataclass
class SceneLayoutSettings:
  goal_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
  goalkeeper_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.8])
  shooter_behind_ball: float = 0.5
  motion_origin_offset: list[float] = field(default_factory=lambda: [-5.0, 0.0, 0.0])
  motion_yaw_offset: float = 1.5708
  # Eval positions in motion-local coords (G1 near origin, facing -y).
  eval_ball_pos: list[float] = field(default_factory=lambda: [0.0, -1.5, 0.11])
  eval_goal_pos: list[float] = field(default_factory=lambda: [0.0, -5.5, 0.0])
  # ball_pos and shooter_pos are computed in load_settings() from
  # penalty_spot.distance_from_goal and ball.radius — they are set as
  # plain instance attributes after dataclass construction.


@dataclass
class GoalkeeperBallVelSettings:
  """Legacy — retained for SoccerSettings field default compatibility."""
  speed_min: float = 3.0
  speed_max: float = 5.0
  goal_margin: float = 0.2


@dataclass
class ShooterBallVelSettings:
  speed: float = 5.0  # m/s — initial ball speed for kick evaluation


@dataclass
class RegionSettings:
  """A single ball landing region: height (z) and width (y) ranges in goal frame."""

  height: list[float] = field(default_factory=lambda: [0.3, 1.5])
  width: list[float] = field(default_factory=lambda: [0.0, 1.5])


@dataclass
class BallTrajectorySettings:
  """Parameters controlling the parabolic ball trajectory model."""

  ball_start_distance: list[float] = field(default_factory=lambda: [3.0, 5.0])
  ball_end_distance: list[float] = field(default_factory=lambda: [0.1, 0.6])
  t_flight: list[float] = field(default_factory=lambda: [0.5, 1.0])


@dataclass
class GroundSettings:
  solref: list[float] = field(default_factory=lambda: [0.02, 0.07])
  solimp: list[float] = field(
    default_factory=lambda: [0.9, 0.95, 0.001, 0.5, 2]
  )
  friction: list[float] = field(default_factory=lambda: [1.0, 1.0])


@dataclass
class GoalkeeperTrainingSettings:
  ee_reach_std: float = 0.3
  stop_ball_vel_drop: float = 2.0
  behind_robot_x: float = 0.0
  episode_length_s: float = 3.0


@dataclass
class SoccerSettings:
  ball: BallSettings = field(default_factory=BallSettings)
  goal: GoalSettings = field(default_factory=GoalSettings)
  penalty_spot: PenaltySpotSettings = field(default_factory=PenaltySpotSettings)
  scene: SceneLayoutSettings = field(default_factory=SceneLayoutSettings)
  ground: GroundSettings = field(default_factory=GroundSettings)
  goalkeeper_ball_vel: GoalkeeperBallVelSettings = field(
    default_factory=GoalkeeperBallVelSettings
  )
  shooter_ball_vel: ShooterBallVelSettings = field(
    default_factory=ShooterBallVelSettings
  )
  goalkeeper_regions: list[RegionSettings] = field(
    default_factory=lambda: [RegionSettings() for _ in range(6)]
  )
  ball_trajectory: BallTrajectorySettings = field(
    default_factory=BallTrajectorySettings
  )
  goalkeeper_training: GoalkeeperTrainingSettings = field(
    default_factory=GoalkeeperTrainingSettings
  )
  episode_length_s: float = 10.0
  goalkeeper_episode_length_s: float = 3.0


_SETTINGS_PATH = Path(__file__).parent / "settings.yaml"


def _dict_to_dataclass(d: dict, dc: type) -> object:
  """Recursively convert a dict to a dataclass instance."""
  field_types = get_type_hints(dc)
  kwargs: dict = {}
  for key, value in d.items():
    if key in field_types:
      ft = field_types[key]
      if is_dataclass(ft) and isinstance(value, dict):
        kwargs[key] = _dict_to_dataclass(value, ft)
      elif get_origin(ft) is list and isinstance(value, list):
        inner = get_args(ft)[0]
        if is_dataclass(inner):
          kwargs[key] = [_dict_to_dataclass(v, inner) for v in value]
        else:
          kwargs[key] = value
      else:
        kwargs[key] = value
    else:
      kwargs[key] = value
  return dc(**kwargs)


def load_settings() -> SoccerSettings:
  with open(_SETTINGS_PATH) as f:
    raw = yaml.safe_load(f)
  settings = _dict_to_dataclass(raw, SoccerSettings)

  # Derive scene positions from distance_from_goal (single source of truth).
  d = settings.penalty_spot.distance_from_goal
  r = settings.ball.radius
  gx, gy, _ = settings.scene.goal_pos
  sb = settings.scene.shooter_behind_ball
  settings.scene.ball_pos = [gx - d, gy, r]
  settings.scene.shooter_pos = [gx - d - sb, gy, 0.8]

  return settings


# Module-level singleton — loaded once at import time.
SETTINGS = load_settings()
