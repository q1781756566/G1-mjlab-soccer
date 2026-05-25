"""Cross-evaluate two teams: Shooter vs Goalkeeper in a shared MuJoCo scene.

Loads two independently-trained G1 policies into one simulation, routes
each robot's observations to its respective policy, concatenates actions,
and steps the physics.  Designed for Phase 2 all-to-all peer evaluation
(10 trials per matchup, per the CS2810 project rubric).

Scene layout (world frame, z-up):
  - Goalkeeper at (0, 0, 0.8), yaw=0, faces +x  (matches GK training frame)
  - Goal at (-0.5, 0, 0), behind the goalkeeper
  - Shooter at (4, 0, 0.8), yaw=pi, faces -x toward the goal
  - Ball at (3, 0, 0.1), in front of the shooter

Usage:
  # Headless batch (10 trials, the Phase 2 default)
  python scripts/compete.py \\
      --shooter-checkpoint <team_a_shooter.pt> \\
      --goalkeeper-checkpoint <team_b_goalkeeper.pt> \\
      --headless --num-trials 10

  # Interactive viewer (single episode, for debugging)
  python scripts/compete.py \\
      --shooter-checkpoint <path> \\
      --goalkeeper-checkpoint <path>

  # Zero-agent baseline (no checkpoints)
  python scripts/compete.py --headless --num-trials 10

-------------------------------------------------------------------------------
CUSTOMIZATION GUIDE (for Phase 2 cross-evaluation)
-------------------------------------------------------------------------------

When you load another team's checkpoint you MUST match the observation
space that their policy was trained with.  The sections marked
``CUSTOMIZE_OBSERVATIONS`` below show where to adjust the observation
terms for the shooter and goalkeeper.  In most cases you will need the
opponent team to provide:

  1. The observation term names, functions, and params they used
  2. The history_length for the actor observation group
  3. The action scale and default joint positions

This compete.py ships with DEFAULT observation configs that match the
template's eval scripts:

  Shooter  – proprioception + ball position  (``shared_obs`` functions)
  Goalkeeper – matches ``eval_goalkeeper_cfg``  (``goalkeeper_obs`` functions,
               960-D history-stacked actor input, HIMPPO architecture)

If your (or your opponent's) policy was trained with a different
observation space you MUST update ``make_compete_env_cfg()`` accordingly.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

# ---------------------------------------------------------------------------
# mjlab / project imports
# ---------------------------------------------------------------------------

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer, ViewerConfig

from src.assets.robots import get_g1_robot_cfg, G1_ACTION_SCALE
from src.assets.robots.unitree_g1.g1_constants import HOME_KEYFRAME, FULL_COLLISION
from src.tasks.soccer.ball import get_ball_cfg
from src.tasks.soccer.goal import get_goal_cfg
from src.tasks.soccer.ground import get_ground_cfg
from src.tasks.soccer.soccer_env_cfg import _add_soccer_scene_postproc
from src.tasks.soccer.config.soccer_settings import SETTINGS
from src.tasks.soccer import mdp
from src.tasks.soccer.mdp.shared_obs import (
    ball_pos_in_robot_frame,
    ball_vel_in_robot_frame,
)
from src.tasks.soccer.mdp.goalkeeper_obs import (
    _GK_DEFAULT_JOINT_POS,
    get_gk_robot_cfg,
    gk_ang_vel,
    gk_joint_pos_rel,
    gk_joint_vel_rel,
    gk_last_action,
)
from src.tasks.soccer.config.g1.rl_cfg import (
    GoalkeeperRunner,
    SoccerRecurrentRunner,
    unitree_g1_goalkeeper_ppo_runner_cfg,
    unitree_g1_soccer_recurrent_runner_cfg,
)

# =============================================================================
# Scene constants
# =============================================================================

# Goalkeeper at origin (matches training coordinate frame: faces +x).
GK_POS: tuple[float, float, float] = (0.0, 0.0, 0.8)
GK_YAW: float = 0.0

# Goal behind goalkeeper  (goal plane x = -0.5).
GOAL_POS: tuple[float, float, float] = (-0.5, 0.0, 0.0)

# Shooter at +x, facing -x toward the goal.
SHOOTER_POS: tuple[float, float, float] = (4.0, 0.0, 0.8)
SHOOTER_YAW: float = math.pi

# Ball in front of the shooter.
BALL_POS: tuple[float, float, float] = (3.0, 0.0, 0.1)

# Episode / control.
EPISODE_LENGTH_S: float = 10.0
CONTROL_DECIMATION: int = 4
MUJOCO_TIMESTEP: float = 0.005

# Goal geometry (behind goalkeeper, x = -0.5, width 3.0 m, height 1.8 m).
GOAL_X: float = -0.5
GOAL_HALF_WIDTH: float = 1.5
GOAL_HEIGHT: float = 1.8

# Commonly-used SceneEntityCfg instances.
_SHOOTER_CFG = SceneEntityCfg("shooter")
_GK_CFG = SceneEntityCfg("goalkeeper")
_BALL_CFG = SceneEntityCfg("ball")


# =============================================================================
# Helper: yaw -> quaternion  (rotation about world z-axis)
# =============================================================================

def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    half = yaw / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


# =============================================================================
# Robot entity factories
# =============================================================================

def _make_shooter_robot() -> Any:
    """Standard G1 at SHOOTER_POS, yaw=pi (faces -x toward goalkeeper)."""
    cfg = get_g1_robot_cfg()
    cfg.init_state = replace(
        HOME_KEYFRAME,
        pos=SHOOTER_POS,
        rot=_yaw_to_quat(SHOOTER_YAW),
    )
    cfg.collisions = (FULL_COLLISION,)
    return cfg


def _make_goalkeeper_robot() -> Any:
    """G1 with GK-specific PD gains and GK reference stance at GK_POS."""
    cfg = get_gk_robot_cfg()
    cfg.init_state = replace(
        cfg.init_state,
        pos=GK_POS,
        rot=_yaw_to_quat(GK_YAW),
        joint_pos=_GK_DEFAULT_JOINT_POS,
    )
    cfg.collisions = (FULL_COLLISION,)
    return cfg


# =============================================================================
# Compete Environment Config
# =============================================================================
#
# CUSTOMIZE_OBSERVATIONS: if your (or your opponent's) policy was trained with
# different observation terms, functions, scaling, or history length, edit the
# ``shooter_actor_terms``, ``goalkeeper_actor_terms``, and the corresponding
# critic terms below.  Also adjust history_length on each ObservationGroupCfg.
# =============================================================================

def make_compete_env_cfg() -> ManagerBasedRlEnvCfg:
    """Build the two-robot competition environment configuration.

    Returns a ``ManagerBasedRlEnvCfg`` whose scene contains both a "shooter"
    and a "goalkeeper" entity.  Four observation groups are defined so that
    each policy can be fed independently:

    * ``shooter_actor``  – observation for the shooter policy
    * ``shooter_critic`` – privileged obs (unused at inference; included for completeness)
    * ``goalkeeper_actor``  – observation for the goalkeeper policy (×10 history)
    * ``goalkeeper_critic`` – privileged obs (unused at inference)
    """

    # -- Scene ----------------------------------------------------------------

    entities: dict[str, Any] = {
        "ground": get_ground_cfg(),
        "ball": get_ball_cfg(pos=BALL_POS),
        "goal": get_goal_cfg(pos=GOAL_POS),
        "shooter": _make_shooter_robot(),
        "goalkeeper": _make_goalkeeper_robot(),
    }

    # ----------------------------------------------------------------------
    # CUSTOMIZE_OBSERVATIONS: Shooter actor terms
    #
    # Default: proprioception + ball position (~100-D).
    # If your shooter was trained with motion references, target-point
    # observations, or different noise / scaling, replace these terms.
    # ----------------------------------------------------------------------
    shooter_actor_terms: dict[str, ObservationTermCfg] = {
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "shooter/imu_ang_vel"},
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            params={"asset_cfg": _SHOOTER_CFG},
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": _SHOOTER_CFG},
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _SHOOTER_CFG},
        ),
        "actions": ObservationTermCfg(
            func=mdp.last_action,
            params={"action_name": "shooter_joint_pos"},
        ),
        "ball_pos_local": ObservationTermCfg(
            func=ball_pos_in_robot_frame,
            params={"ball_cfg": _BALL_CFG, "robot_cfg": _SHOOTER_CFG},
        ),
    }

    shooter_critic_terms: dict[str, ObservationTermCfg] = {
        **shooter_actor_terms,
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "shooter/imu_lin_vel"},
        ),
    }

    # ----------------------------------------------------------------------
    # CUSTOMIZE_OBSERVATIONS: Goalkeeper actor terms
    #
    # Default: matches ``eval_goalkeeper_cfg`` exactly (960-D history-stacked
    # input with GK-specific scaling).  If your opponent's goalkeeper uses a
    # different observation space, replace these terms and adjust
    # ``history_length`` on the ``goalkeeper_actor`` group below.
    # ----------------------------------------------------------------------
    goalkeeper_actor_terms: dict[str, ObservationTermCfg] = {
        "ball_pos_local": ObservationTermCfg(
            func=ball_pos_in_robot_frame,
            params={"ball_cfg": _BALL_CFG, "robot_cfg": _GK_CFG},
        ),
        "base_ang_vel": ObservationTermCfg(
            func=gk_ang_vel,
            params={"sensor_name": "goalkeeper/imu_ang_vel"},
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            params={"asset_cfg": _GK_CFG},
        ),
        "joint_pos": ObservationTermCfg(
            func=gk_joint_pos_rel,
            params={"asset_cfg": _GK_CFG},
        ),
        "joint_vel": ObservationTermCfg(
            func=gk_joint_vel_rel,
            params={"asset_cfg": _GK_CFG},
        ),
        "actions": ObservationTermCfg(
            func=gk_last_action,
            params={"action_name": "goalkeeper_joint_pos"},
        ),
    }

    goalkeeper_critic_terms: dict[str, ObservationTermCfg] = {
        **goalkeeper_actor_terms,
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "goalkeeper/imu_lin_vel"},
        ),
        "ball_vel_local": ObservationTermCfg(
            func=ball_vel_in_robot_frame,
            params={"ball_cfg": _BALL_CFG, "robot_cfg": _GK_CFG},
        ),
    }

    # -- Observation groups ---------------------------------------------------

    observations: dict[str, ObservationGroupCfg] = {
        "shooter_actor": ObservationGroupCfg(
            terms=shooter_actor_terms,
            concatenate_terms=True,
            enable_corruption=False,
            history_length=1,
        ),
        "shooter_critic": ObservationGroupCfg(
            terms=shooter_critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
            history_length=1,
        ),
        "goalkeeper_actor": ObservationGroupCfg(
            terms=goalkeeper_actor_terms,
            concatenate_terms=True,
            enable_corruption=False,
            history_length=10,  # CUSTOMIZE_OBSERVATIONS: 10-frame HIMPPO stack
        ),
        "goalkeeper_critic": ObservationGroupCfg(
            terms=goalkeeper_critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
            history_length=1,
        ),
    }

    # -- Actions --------------------------------------------------------------

    # CUSTOMIZE_OBSERVATIONS: action scales must match what each policy
    # expects.  Shooter uses per-joint G1_ACTION_SCALE; goalkeeper uses
    # uniform 0.25 (matching GK PD gains).
    actions: dict[str, ActionTermCfg] = {
        "shooter_joint_pos": JointPositionActionCfg(
            entity_name="shooter",
            actuator_names=(".*",),
            scale=G1_ACTION_SCALE,
            use_default_offset=True,
        ),
        "goalkeeper_joint_pos": JointPositionActionCfg(
            entity_name="goalkeeper",
            actuator_names=(".*",),
            scale=0.25,
            use_default_offset=True,
        ),
    }

    # -- Events (reset) -------------------------------------------------------

    events: dict[str, EventTermCfg] = {
        "reset_shooter_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "asset_cfg": _SHOOTER_CFG,
            },
        ),
        "reset_shooter_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.0, 0.0),
                "velocity_range": (-0.0, 0.0),
                "asset_cfg": SceneEntityCfg("shooter", joint_names=(".*",)),
            },
        ),
        "reset_goalkeeper_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "asset_cfg": _GK_CFG,
            },
        ),
        "reset_goalkeeper_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.0, 0.0),
                "velocity_range": (-0.0, 0.0),
                "asset_cfg": SceneEntityCfg("goalkeeper", joint_names=(".*",)),
            },
        ),
        "reset_ball": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "asset_cfg": _BALL_CFG,
            },
        ),
    }

    # -- Terminations ---------------------------------------------------------

    terminations: dict[str, TerminationTermCfg] = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "shooter_fell": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"asset_cfg": _SHOOTER_CFG, "limit_angle": math.radians(70.0)},
        ),
        "goalkeeper_fell": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"asset_cfg": _GK_CFG, "limit_angle": math.radians(70.0)},
        ),
    }

    # -- Rewards (placeholder — unused at inference) --------------------------

    rewards: dict[str, RewardTermCfg] = {
        "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    }

    # -- Assemble & return ----------------------------------------------------

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            entities=entities,
            num_envs=1,
            spec_fn=_add_soccer_scene_postproc,
        ),
        observations=observations,
        actions=actions,
        commands={},
        events=events,
        rewards=rewards,
        terminations=terminations,
        viewer=ViewerConfig(
            lookat=(2.0, 0.0, 1.0),
            distance=6.0,
            elevation=-15.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=256,
            njmax=3000,
            mujoco=MujocoCfg(
                timestep=MUJOCO_TIMESTEP,
                iterations=10,
                ls_iterations=20,
            ),
        ),
        decimation=CONTROL_DECIMATION,
        episode_length_s=EPISODE_LENGTH_S,
    )


# =============================================================================
# Policy loading
# =============================================================================

def _load_shooter_policy(
    checkpoint_path: str, env: ManagerBasedRlEnv, device: str
) -> Any:
    """Load a shooter checkpoint via SoccerRecurrentRunner (LSTM-based).

    The runner is initialised from the compete environment so that the model
    architecture matches the observation and action dimensions of the current
    env config.  If the checkpoint was trained with a different observation
    space you will see a size-mismatch error — update the shooter observation
    terms in ``make_compete_env_cfg()`` and retry.
    """
    print(f"[INFO] Loading shooter policy from: {checkpoint_path}")
    agent_cfg = unitree_g1_soccer_recurrent_runner_cfg()
    runner = SoccerRecurrentRunner(env, asdict(agent_cfg), log_dir=None, device=device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print("[INFO] Shooter policy loaded.")
    return policy


def _load_goalkeeper_policy(
    checkpoint_path: str, env: ManagerBasedRlEnv, device: str
) -> Any:
    """Load a goalkeeper checkpoint via GoalkeeperRunner (HIMPPO-based).

    Detects reference HIMPPO checkpoints (single ``model_state_dict`` for a
    unified ActorCritic) and loads them directly, bypassing the legacy
    migration path.  Otherwise falls back to the standard RSL-RL loader.
    """
    print(f"[INFO] Loading goalkeeper policy from: {checkpoint_path}")
    loaded = torch.load(checkpoint_path, map_location=device)

    agent_cfg = unitree_g1_goalkeeper_ppo_runner_cfg()
    runner = GoalkeeperRunner(env, asdict(agent_cfg), device=device)

    if "model_state_dict" in loaded and hasattr(runner.alg.actor, "history_encoder"):
        print("[INFO] Detected HIMPPO ActorCritic checkpoint — loading directly.")
        actor_state = {
            k: v
            for k, v in loaded["model_state_dict"].items()
            if not k.startswith("critic.")
        }
        runner.alg.actor.load_state_dict(actor_state, strict=False)
        print("[INFO] Goalkeeper policy loaded.")
    else:
        runner.load(checkpoint_path, load_cfg={"actor": True})
        print("[INFO] Goalkeeper policy loaded.")

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    return policy


# =============================================================================
# Zero-policy fallbacks
# =============================================================================

class _ZeroPolicy:
    """Policy that always outputs zeros (baseline / debug)."""

    def __init__(self, action_dim: int, device: str):
        self._zero = torch.zeros(1, action_dim, device=device)

    def __call__(self, obs: dict) -> torch.Tensor:
        del obs
        return self._zero

    def reset(self) -> None:
        pass


# =============================================================================
# Combined policy wrapper  (for viewer)
# =============================================================================

class CombinedPolicy:
    """Wraps two independent policies so the viewer sees a single policy.

    Extracts ``shooter_actor`` and ``goalkeeper_actor`` from the full
    observation dict, calls each sub-policy, and concatenates their actions
    into one tensor.
    """

    def __init__(self, shooter_policy: Any, goalkeeper_policy: Any):
        self._shooter = shooter_policy
        self._goalkeeper = goalkeeper_policy

    def __call__(self, obs: dict) -> torch.Tensor:
        s_act = self._shooter({"actor": obs["shooter_actor"]})
        g_act = self._goalkeeper({"actor": obs["goalkeeper_actor"]})
        return torch.cat([s_act, g_act], dim=-1)

    def reset(self) -> None:
        self._shooter.reset()
        self._goalkeeper.reset()


# =============================================================================
# Competition metrics
# =============================================================================

def _ball_entered_goal(ball_pos: torch.Tensor) -> bool:
    """Ball has crossed the goal plane (x <= GOAL_X) inside the goal frame."""
    x, y, z = ball_pos[0].item(), ball_pos[1].item(), ball_pos[2].item()
    return x <= GOAL_X and abs(y) <= GOAL_HALF_WIDTH and z <= GOAL_HEIGHT


def _ball_blocked(
    ball_vel: torch.Tensor,
    ball_pos: torch.Tensor,
    prev_speed: float,
    threshold: float = 2.0,
) -> bool:
    """Ball was blocked if its speed dropped by > threshold m/s behind GK."""
    speed = float(torch.norm(ball_vel))
    behind_gk = ball_pos[0].item() <= GK_POS[0]
    return behind_gk and (prev_speed - speed) > threshold


def run_trial(
    env: ManagerBasedRlEnv,
    shooter_policy: Any,
    goalkeeper_policy: Any,
    max_steps: int = 500,
) -> dict[str, Any]:
    """Run one competition episode.

    Returns a dict with keys:
      goal_scored, blocked, steps, ball_final_x, early_termination
    """
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    ball = env.unwrapped.scene["ball"]
    goal_scored = False
    blocked = False
    prev_ball_speed = 0.0
    steps = 0
    early_termination = False

    for _ in range(max_steps):
        with torch.inference_mode():
            s_act = shooter_policy({"actor": obs["shooter_actor"]})
            g_act = goalkeeper_policy({"actor": obs["goalkeeper_actor"]})
        action = torch.cat([s_act, g_act], dim=-1)

        result = env.step(action)
        obs = result[0]
        steps += 1

        ball_pos = ball.data.root_link_pos_w[0].cpu()
        ball_vel = ball.data.root_link_vel_w[0, :3].cpu()
        ball_speed = float(torch.norm(ball_vel))

        if _ball_entered_goal(ball_pos):
            goal_scored = True

        if _ball_blocked(ball_vel, ball_pos, prev_ball_speed):
            blocked = True

        prev_ball_speed = ball_speed

        # result[2] is the "terminated" signal (gym API).
        terminated = result[2]
        if hasattr(terminated, "item"):
            terminated = bool(terminated.item())
        else:
            terminated = bool(terminated)
        if terminated:
            # Distinguish timeout from early termination.
            if steps < max_steps - 1:
                early_termination = True
            break

    return {
        "goal_scored": goal_scored,
        "blocked": blocked,
        "steps": steps,
        "ball_final_x": float(ball.data.root_link_pos_w[0, 0].cpu()),
        "early_termination": early_termination,
    }


# =============================================================================
# Headless batch evaluation
# =============================================================================

def run_headless_eval(
    num_trials: int,
    env: ManagerBasedRlEnv,
    shooter_policy: Any,
    goalkeeper_policy: Any,
) -> dict[str, Any]:
    """Run multiple trials headless and return aggregate statistics."""
    print(f"\n[INFO] Running {num_trials} headless competition trials ...\n")

    goals = 0
    blocks = 0
    early_terminations = 0
    ball_crossed_goal_line = 0

    for trial in range(num_trials):
        stats = run_trial(env, shooter_policy, goalkeeper_policy)
        if stats["goal_scored"]:
            goals += 1
        if stats["blocked"]:
            blocks += 1
        if stats["early_termination"]:
            early_terminations += 1
        if stats["ball_final_x"] <= GOAL_X:
            ball_crossed_goal_line += 1

        interval = 1 if num_trials <= 10 else (num_trials // 10)
        if (trial + 1) % interval == 0 or trial == 0:
            print(
                f"  Trial {trial + 1:3d}/{num_trials}: "
                f"goal={stats['goal_scored']}, "
                f"blocked={stats['blocked']}, "
                f"early_term={stats['early_termination']}, "
                f"steps={stats['steps']}"
            )

    total = num_trials
    print(f"\n{'=' * 60}")
    print(f"  Competition Summary  ({total} trials)")
    print(f"{'=' * 60}")
    print(f"  Shooter goals:          {goals}/{total}  ({goals / total * 100:.1f}%)")
    print(f"  Goalkeeper blocks:      {blocks}/{total}  ({blocks / total * 100:.1f}%)")
    print(f"  Early terminations:     {early_terminations}/{total}")
    print(f"  Ball crossed goal line: {ball_crossed_goal_line}/{total}")
    print(f"{'=' * 60}\n")

    return {
        "num_trials": total,
        "goals": goals,
        "blocks": blocks,
        "early_terminations": early_terminations,
        "ball_crossed_goal_line": ball_crossed_goal_line,
    }


# =============================================================================
# Viewer
# =============================================================================

def run_viewer(
    viewer_type: str,
    env: ManagerBasedRlEnv,
    combined_policy: CombinedPolicy,
) -> None:
    """Launch an interactive viewer with the combined policy."""
    if viewer_type == "native":
        NativeMujocoViewer(env, combined_policy).run()
    elif viewer_type == "viser":
        ViserPlayViewer(env, combined_policy).run()
    else:
        raise RuntimeError(f"Unsupported viewer: {viewer_type}")


# =============================================================================
# CLI
# =============================================================================

@dataclass
class CompeteConfig:
    shooter_checkpoint: str | None = None
    """Path to the shooter policy checkpoint (.pt)."""

    goalkeeper_checkpoint: str | None = None
    """Path to the goalkeeper policy checkpoint (.pt)."""

    num_trials: int = 0
    """Number of evaluation trials (> 0 enables headless batch mode)."""

    headless: bool = False
    """Run without a viewer (required for multi-trial eval)."""

    video: bool = False
    """Record video of the first trial (requires --headless)."""

    video_length: int = 500
    """Video length in steps."""

    video_height: int = 480
    video_width: int = 640

    viewer: str = "auto"
    """Viewer type: 'auto', 'native', or 'viser'."""

    device: str | None = None
    """Torch device (auto-detected if omitted)."""

    seed: int = 2810
    """Random seed."""

    task_id: str = "Compete"


def run_compete(cfg: CompeteConfig) -> None:
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # -- Build environment ----------------------------------------------------
    print(f"Task: {cfg.task_id}  |  device: {device}")
    env_cfg = make_compete_env_cfg()
    env_cfg.scene.num_envs = 1
    env_cfg.viewer.height = cfg.video_height
    env_cfg.viewer.width = cfg.video_width

    # Print observation info for debugging.
    for grp_name in ("shooter_actor", "goalkeeper_actor"):
        grp = env_cfg.observations[grp_name]
        term_names = list(grp.terms.keys())
        print(
            f"{grp_name}: {len(term_names)} terms x {grp.history_length} history  "
            f"terms={term_names}"
        )

    render_mode = "rgb_array" if cfg.video else None
    env_base = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    # -- Load policies --------------------------------------------------------
    act_dim_shooter = env_base.action_manager.get_term("shooter_joint_pos").action_dim
    act_dim_goalkeeper = env_base.action_manager.get_term("goalkeeper_joint_pos").action_dim
    print(f"Action dims: shooter={act_dim_shooter}, goalkeeper={act_dim_goalkeeper}")

    if cfg.shooter_checkpoint:
        shooter_policy = _load_shooter_policy(cfg.shooter_checkpoint, env_base, device)
    else:
        shooter_policy = _ZeroPolicy(act_dim_shooter, device)
        print("[INFO] No shooter checkpoint — using zero policy.")

    if cfg.goalkeeper_checkpoint:
        goalkeeper_policy = _load_goalkeeper_policy(cfg.goalkeeper_checkpoint, env_base, device)
    else:
        goalkeeper_policy = _ZeroPolicy(act_dim_goalkeeper, device)
        print("[INFO] No goalkeeper checkpoint — using zero policy.")

    # Wrap after policy construction so runner sees the raw env.
    env = RslRlVecEnvWrapper(env_base, clip_actions=100.0)

    # Print observed shapes from one reset.
    obs_sample = env.reset()
    if isinstance(obs_sample, tuple):
        obs_sample = obs_sample[0]
    for key in ("shooter_actor", "goalkeeper_actor"):
        print(f"  {key} shape: {obs_sample[key].shape}")

    # -- Video recording (optional) -------------------------------------------
    if cfg.video and cfg.headless:
        from mjlab.utils.wrappers import VideoRecorder

        video_folder = Path("videos") / "compete"
        video_folder.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Recording video to: {video_folder}")
        env = VideoRecorder(
            env,
            video_folder=video_folder,
            step_trigger=lambda step: step == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )

    # -- Run ------------------------------------------------------------------
    if cfg.headless:
        if cfg.num_trials <= 0:
            print("[WARN] --headless without --num-trials; nothing to evaluate.")
        else:
            run_headless_eval(cfg.num_trials, env, shooter_policy, goalkeeper_policy)
    else:
        if cfg.num_trials > 0:
            print("[INFO] --num-trials set without --headless; launching viewer.")
        combined = CombinedPolicy(shooter_policy, goalkeeper_policy)

        if cfg.viewer == "auto":
            has_display = bool(
                os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
            )
            viewer_type = "native" if has_display else "viser"
        else:
            viewer_type = cfg.viewer
        run_viewer(viewer_type, env, combined)

    env.close()


def main() -> None:
    import mjlab.tasks  # noqa: F401
    import src.tasks    # noqa: F401

    args = tyro.cli(CompeteConfig, prog="compete")
    run_compete(args)


if __name__ == "__main__":
    main()
