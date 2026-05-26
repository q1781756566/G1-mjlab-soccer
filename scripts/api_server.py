"""Reference policy server for Phase 2 tournament (robosuite format).

Implements the standard REST API that ``compete.py`` calls during cross-evaluation:

  POST /act    - receive observation, return action
  POST /reset  - reset policy hidden state

Usage:
  python scripts/api_server.py --checkpoint <path> --port 8000 --task shooter
  python scripts/api_server.py --checkpoint <path> --port 8001 --task goalkeeper

Test with curl:
  curl -X POST http://localhost:8000/act \\
       -H "Content-Type: application/json" \\
       -d '{"observation": [[0.0, ...]]}'
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Any

import torch
import tyro
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


# ---------------------------------------------------------------------------
# Request / response schemas (must match compete.py's ApiPolicy exactly)
# ---------------------------------------------------------------------------

class ActRequest(BaseModel):
    observation: list[list[float]]  # shape: [1, obs_dim]


class ActResponse(BaseModel):
    action: list[list[float]]  # shape: [1, act_dim]


# ---------------------------------------------------------------------------
# Policy loading (matches eval script logic for each task)
# ---------------------------------------------------------------------------

def _load_policy(checkpoint_path: str, task_id: str, device: str) -> Any:
    """Build env from task config, load checkpoint, return inference policy."""
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()

    env_cfg = load_env_cfg(task_id, play=False)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    print(f"[INFO] Task: {task_id}")
    actor_terms = list(env_cfg.observations["actor"].terms.keys())
    print(f"[INFO] Actor obs  ({len(actor_terms)} terms): {actor_terms}")
    print(f"[INFO] Action dim: {env.num_actions}")

    if task_id == "Eval-Goalkeeper":
        from src.tasks.soccer.config.g1.rl_cfg import (
            GoalkeeperRunner,
            unitree_g1_goalkeeper_ppo_runner_cfg,
        )
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
        else:
            runner.load(checkpoint_path, load_cfg={"actor": True})
    else:
        from src.tasks.soccer.config.g1.rl_cfg import (
            SoccerRecurrentRunner,
            unitree_g1_soccer_recurrent_runner_cfg,
        )
        agent_cfg = unitree_g1_soccer_recurrent_runner_cfg()
        runner = SoccerRecurrentRunner(env, asdict(agent_cfg), log_dir=None, device=device)
        runner.load(checkpoint_path)

    policy = runner.get_inference_policy(device=device)
    print(f"[INFO] Policy loaded from: {checkpoint_path}")
    return policy, env


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(checkpoint_path: str, task_id: str, device: str) -> FastAPI:
    """Build the FastAPI app with a loaded policy."""

    policy, env = _load_policy(checkpoint_path, task_id, device)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print(f"[INFO] Server ready — {task_id} policy on {device}")
        yield
        env.close()
        print("[INFO] Server shutting down.")

    app = FastAPI(title=f"CS2810 Phase 2 — {task_id}", lifespan=lifespan)

    # Allow compete.py from any origin.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.post("/act", response_model=ActResponse)
    async def act(req: ActRequest):
        obs = torch.tensor(req.observation, device=device, dtype=torch.float32)
        with torch.inference_mode():
            action = policy({"actor": obs})
        return ActResponse(action=action.cpu().tolist())

    @app.post("/reset")
    async def reset():
        policy.reset()
        return {"status": "ok"}

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclass
class ServerConfig:
    checkpoint: str
    """Path to the policy checkpoint (.pt)."""
    port: int = 8000
    """Port to listen on."""
    task: str = "shooter"
    """Task type: 'shooter' or 'goalkeeper'."""
    host: str = "0.0.0.0"
    """Host to bind to."""
    device: str | None = None
    """Torch device (auto-detected if omitted)."""


def main():
    import src.tasks  # noqa: F401  — register eval tasks

    args = tyro.cli(ServerConfig, prog="api_server")

    task_id = "Eval-Shooter" if args.task == "shooter" else "Eval-Goalkeeper"
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    app = create_app(args.checkpoint, task_id, device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
