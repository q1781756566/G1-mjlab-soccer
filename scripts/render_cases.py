"""Render N random goalkeeper episodes, one mp4 each, into an output dir."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import imageio, torch, tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from src.tasks.soccer.config.g1.gk_train_cfg import goalkeeper_train_runner_cfg


def main(ckpt: str = "logs/rsl_rl/g1_goalkeeper/distilled/model_distilled_v3.pt",
         out_dir: str = "/mnt/c/QBS/Courses/embodied_ai/proj/viz",
         device: str = "cuda:0", n: int = 20, seed: int = 100):
  configure_torch_backends()
  Path(out_dir).mkdir(parents=True, exist_ok=True)
  cfg = load_env_cfg("Eval-Goalkeeper", play=False)
  cfg.scene.num_envs = 1
  cfg.seed = seed
  if "fell_over" in cfg.terminations: cfg.terminations["fell_over"] = None
  cfg.viewer.width = 640; cfg.viewer.height = 480
  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array")
  wrapped = RslRlVecEnvWrapper(env, clip_actions=100.0)
  runner = MjlabOnPolicyRunner(wrapped, asdict(goalkeeper_train_runner_cfg()), device=device)
  runner.load(ckpt, load_cfg={"actor": True})
  policy = runner.get_inference_policy(device=device)
  ball = env.scene["ball"]; org = env.scene.env_origins[0]

  n_save = 0
  for ep in range(n):
    obs = wrapped.reset()
    if isinstance(obs, tuple): obs = obs[0]
    frames = []; entered = False
    for _ in range(150):
      with torch.inference_mode():
        a = policy(obs)
      res = wrapped.step(a); obs = res[0]
      frames.append(env.render())
      bp = ball.data.root_link_pos_w[0]
      if (bp[0]-org[0]).item() <= -0.5 and abs((bp[1]-org[1]).item()) <= 1.5 and bp[2].item() <= 1.8:
        entered = True
      if res[2].item(): break
    label = "goal" if entered else "save"
    n_save += (not entered)
    path = f"{out_dir}/case_{ep:02d}_{label}.mp4"
    imageio.mimsave(path, frames, fps=30, macro_block_size=1)
    print(f"case {ep:02d}: {label}  -> {path}", flush=True)
  print(f"DONE: {n_save}/{n} saves rendered to {out_dir}", flush=True)
  env.close()


if __name__ == "__main__":
  import mjlab.tasks, src.tasks  # noqa
  tyro.cli(main)
