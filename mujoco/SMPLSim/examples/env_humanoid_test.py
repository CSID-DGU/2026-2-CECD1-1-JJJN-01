"""Humanoid smoke test. On macOS use mjpython for human rendering."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import mujoco
import numpy as np
from importlib.resources import files
from omegaconf import DictConfig
from smpl_sim.envs.humanoid_env import HumanoidEnv


@hydra.main(version_base=None, config_path=str(files("smpl_sim").joinpath("data/cfg")), config_name="config")
def main(cfg: DictConfig) -> None:
    # This example is interactive by default; explicit headless=True still wins.
    if not any(
        item.startswith("headless=") for item in hydra.core.hydra_config.HydraConfig.get().overrides.task
    ):
        cfg.headless = cfg.env.render_mode != "human"
    max_steps = int(cfg.get("test_steps", 0))  # 0 means run until the window closes.
    if cfg.headless and max_steps <= 0:
        max_steps = 300
    if cfg.env.task != "HumanoidEnv":
        raise ValueError("This smoke test requires env.task=HumanoidEnv")
    print(f"MuJoCo {mujoco.__version__}; headless={cfg.headless}; render_mode={cfg.env.render_mode}", flush=True)
    env = HumanoidEnv(cfg)
    frames = []
    try:
        print("environment initialized; resetting", flush=True)
        env.reset(seed=cfg.seed)
        print("reset complete", flush=True)
        steps = 0
        while max_steps <= 0 or steps < max_steps:
            if env.viewer is not None and not env.viewer.is_running():
                break
            _, _, terminated, truncated, _ = env.step(np.zeros(env.action_space.shape))
            if not np.isfinite(env.mj_data.qpos).all() or not np.isfinite(env.mj_data.qvel).all():
                raise RuntimeError("Non-finite physics state")
            if not cfg.headless and cfg.env.render_mode == "rgb_array":
                frames.append(env.render())
            steps += 1
            if steps == 1 or steps % 30 == 0:
                print(f"step={steps} sim_time={env.mj_data.time:.6f} contacts={env.mj_data.ncon} ctrl_norm={np.linalg.norm(env.mj_data.ctrl):.3f}", flush=True)
            if (terminated or truncated) and not env.disable_reset:
                env.reset()
        if frames:
            import mediapy as media
            media.write_video("video.mp4", frames, fps=round(1 / env.dt))
        print(f"completed {steps} steps", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
