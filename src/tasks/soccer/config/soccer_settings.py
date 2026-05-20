"""Load soccer settings from YAML and provide typed access."""

from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import get_type_hints

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
  distance_from_goal: float = 6.0


@dataclass
class SceneLayoutSettings:
  goal_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
  goalkeeper_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.8])
  shooter_behind_ball: float = 0.5
  # ball_pos and shooter_pos are computed in load_settings() from
  # penalty_spot.distance_from_goal and ball.radius — they are set as
  # plain instance attributes after dataclass construction.


@dataclass
class GoalkeeperBallVelSettings:
  speed_min: float = 2.0
  speed_max: float = 4.5
  pitch_min_deg: float = 3.0
  pitch_max_deg: float = 8.0
  goal_margin: float = 0.2


@dataclass
class GroundSettings:
  solref: list[float] = field(default_factory=lambda: [0.02, 0.07])
  solimp: list[float] = field(
    default_factory=lambda: [0.9, 0.95, 0.001, 0.5, 2]
  )
  friction: list[float] = field(default_factory=lambda: [1.0, 1.0])


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
  episode_length_s: float = 10.0


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
  gx, gy, gz = settings.scene.goal_pos
  sb = settings.scene.shooter_behind_ball
  settings.scene.ball_pos = [gx - d, gy, r]
  settings.scene.shooter_pos = [gx - d - sb, gy, 0.8]

  return settings


# Module-level singleton — loaded once at import time.
SETTINGS = load_settings()
