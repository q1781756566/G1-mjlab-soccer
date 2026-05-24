"""Unitree G1 soccer task registration."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  unitree_g1_shooter_env_cfg,
  unitree_g1_goalkeeper_env_cfg,
)
from .rl_cfg import (
  GoalkeeperRunner,
  SoccerRecurrentRunner,
  unitree_g1_goalkeeper_ppo_runner_cfg,
  unitree_g1_soccer_ppo_runner_cfg,
  unitree_g1_soccer_recurrent_runner_cfg,
)
from .training_env_cfgs import (
  unitree_g1_goalkeeper_training_env_cfg,
  unitree_g1_stage1_env_cfg,
  unitree_g1_stage2_env_cfg,
)

# -- Naive (placeholder) tasks ------------------------------------------------

register_mjlab_task(
  task_id="Unitree-G1-Naive-Shooter",
  env_cfg=unitree_g1_shooter_env_cfg(),
  play_env_cfg=unitree_g1_shooter_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_ppo_runner_cfg(),
  runner_cls=None,
)

register_mjlab_task(
  task_id="Unitree-G1-Naive-Goalkeeper",
  env_cfg=unitree_g1_goalkeeper_env_cfg(),
  play_env_cfg=unitree_g1_goalkeeper_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_ppo_runner_cfg(),
  runner_cls=None,
)

# -- Shooter training tasks (two-stage) -----------------------------------------
# Stage I: MLP (motion tracking, no temporal dependency)
# Stage II: LSTM (perception-guided kicking with ball trajectory prediction)

register_mjlab_task(
  task_id="Unitree-G1-Shooter-Stage1",
  env_cfg=unitree_g1_stage1_env_cfg(),
  play_env_cfg=unitree_g1_stage1_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_ppo_runner_cfg(),
  runner_cls=None,
)

register_mjlab_task(
  task_id="Unitree-G1-Shooter-Stage2",
  env_cfg=unitree_g1_stage2_env_cfg(),
  play_env_cfg=unitree_g1_stage2_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_recurrent_runner_cfg(),
  runner_cls=SoccerRecurrentRunner,
)

# -- Goalkeeper training task (single-stage) ------------------------------------
# MLP [512,256,128] ELU with 10-frame history stacking on actor (96D×10=960D).
# Matches the Humanoid-Goalkeeper paper's observation paradigm.

register_mjlab_task(
  task_id="Unitree-G1-Goalkeeper",
  env_cfg=unitree_g1_goalkeeper_training_env_cfg(),
  play_env_cfg=unitree_g1_goalkeeper_training_env_cfg(play=True),
  rl_cfg=unitree_g1_goalkeeper_ppo_runner_cfg(),
  runner_cls=GoalkeeperRunner,
)
