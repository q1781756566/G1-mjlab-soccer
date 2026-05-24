"""Soccer task MDP terms — re-exported from sub-modules.

Self-contained MDP functions for observations, rewards, terminations,
reset events, domain randomization, and soccer-specific ball resets.
Mirrors the mjlab.tasks.velocity.mdp multi-file pattern.
"""

from mjlab.envs.mdp import *  # noqa: F401, F403

from .commands import MultiMotionSoccerCommand, MultiMotionSoccerCommandCfg  # noqa: F401
from .domain_randomization import *  # noqa: F403
from .kick_detection import KickContactTracker, KickContactEvent, ContactFootInfo  # noqa: F401
from .observations import *  # noqa: F403
from .reset_events import *  # noqa: F403
from .rewards import *  # noqa: F403
from .soccer_reset import *  # noqa: F403
from .terminations import *  # noqa: F403
from . import goalkeeper_obs  # noqa: F401
from . import goalkeeper_rewards  # noqa: F401
from . import training_obs  # noqa: F401
from . import training_rewards  # noqa: F401
