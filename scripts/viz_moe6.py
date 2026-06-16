"""Render the MoE6 goalkeeper to an mp4, one frame at a time, with LIVE progress.

Unlike the VideoRecorder path (accumulates every frame in RAM, writes only at the
end, no progress), this:
  - caps thread pools (the VideoRecorder run thrashed on 162 threads);
  - streams frames straight to the mp4 via imageio's ffmpeg writer (flat memory,
    the output file grows as it renders -> progress is visible on disk);
  - prints flushed per-episode + per-30-frame progress with a running fps.

RENDER BACKEND NOTE: run this on WSL, NOT lab_1. lab_1's EGL offscreen rasterizer
produces flickering dark bands across the ground during dive scenes (a GL/driver
artifact in the raw rendered pixels, not the policy or the h264 encode). WSL's
GPU-accelerated mesa renders the identical scene cleanly *and* ~10x faster
(~22 fps vs ~1.9 fps). lab_1 stays correct for training/eval numbers (physics
are unaffected) — only its offscreen video output glitches.
  WSL: MUJOCO_GL=egl <unitree_rl_mjlab/.venv/bin/python> scripts/viz_moe6.py --out-dir viz_moe6_wsl
"""
from __future__ import annotations
import os, sys, time
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
  os.environ.setdefault(_v, "4")
from dataclasses import dataclass
import torch, tyro, imageio
torch.set_num_threads(4)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # so `eval_naive_goalkeeper` imports

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

_REGION = ["Right-Mid", "Left-Mid", "Right-Up", "Left-Up", "Right-Low", "Left-Low"]


@dataclass
class Cfg:
  checkpoint: str = "src/assets/soccer/weight/goalkeeper_moe6.pt"
  out_dir: str = "viz_moe6_wsl"   # one mp4 per episode here (relative -> repo dir; render on WSL, see module docstring)
  episodes: int = 12
  steps: int = 150
  seed: int = 2810
  width: int = 640
  height: int = 480
  fps: int = 30
  device: str = "cuda:0"


def main(cfg: Cfg):
  configure_torch_backends()
  dev = cfg.device
  env_cfg = load_env_cfg("Eval-Goalkeeper", play=False)
  env_cfg.scene.num_envs = 1; env_cfg.seed = cfg.seed
  env_cfg.viewer.width = cfg.width; env_cfg.viewer.height = cfg.height
  if "fell_over" in env_cfg.terminations: env_cfg.terminations["fell_over"] = None
  raw = ManagerBasedRlEnv(cfg=env_cfg, device=dev, render_mode="rgb_array")
  env = RslRlVecEnvWrapper(raw, clip_actions=100.0)

  from eval_naive_goalkeeper import MoE6Policy
  bundle = torch.load(cfg.checkpoint, map_location=dev, weights_only=False)
  print(f"[INFO] loaded MoE6 bundle (latch_hi={bundle.get('latch_hi')}, mirror={bundle.get('mirror_map')})", flush=True)
  policy = MoE6Policy(bundle, env, dev)

  ball = env.unwrapped.scene["ball"]; org = env.unwrapped.scene.env_origins
  os.makedirs(cfg.out_dir, exist_ok=True)
  print(f"[INFO] writing {cfg.episodes} episodes (one mp4 each) -> {cfg.out_dir}/", flush=True)

  blocked = 0; nframes = 0; t0 = time.time()
  for ep in range(cfg.episodes):
    obs = env.reset()
    if isinstance(obs, tuple): obs = obs[0]
    policy.reset()
    reg = int(env.unwrapped._gk_region[0]) if hasattr(env.unwrapped, "_gk_region") else -1
    rname = _REGION[reg].replace("-", "")
    tmp = f"{cfg.out_dir}/ep{ep+1:02d}_{rname}.tmp.mp4"
    writer = imageio.get_writer(tmp, fps=cfg.fps, codec="libx264", quality=8, macro_block_size=8)
    entered = False; epframes = 0
    for s in range(cfg.steps):
      with torch.no_grad():
        a = policy(obs)
      obs = env.step(a)[0]
      bp = ball.data.root_link_pos_w[0]
      x = (bp[0] - org[0, 0]).item(); y = (bp[1] - org[0, 1]).item(); z = bp[2].item()
      if x <= -0.5 and abs(y) <= 1.5 and z <= 1.8: entered = True
      frame = raw.render()
      if frame is not None:
        writer.append_data(frame); nframes += 1; epframes += 1
      if (s + 1) % 30 == 0:
        fps = nframes / max(time.time() - t0, 1e-6)
        print(f"    [ep {ep+1}/{cfg.episodes} {_REGION[reg]:<10}] step {s+1}/{cfg.steps}"
              f"  ({fps:.1f} fps)", flush=True)
    writer.close()
    tag = "SAVE" if not entered else "GOAL"
    final = f"{cfg.out_dir}/ep{ep+1:02d}_{rname}_{tag}.mp4"
    os.replace(tmp, final)
    blocked += (0 if entered else 1)
    print(f"[ep {ep+1:2d}/{cfg.episodes}] {_REGION[reg]:<10} {tag}   -> {os.path.basename(final)}"
          f"   (running {blocked}/{ep+1} saved)", flush=True)

  dt = time.time() - t0
  print(f"DONE: {blocked}/{cfg.episodes} saved.  {nframes} frames -> {cfg.out_dir}/  ({dt:.0f}s, {nframes/dt:.1f} fps)", flush=True)
  env.close()


if __name__ == "__main__":
  import mjlab.tasks, src.tasks  # noqa
  main(tyro.cli(Cfg, prog="viz_moe6"))
