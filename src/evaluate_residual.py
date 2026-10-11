"""Evaluate a frozen residual PPO checkpoint and render its physical replay."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from free_throw_env import FreeThrowEnv


def run(checkpoint, output, render=False, bundle=None):
    checkpoint = Path(checkpoint).resolve()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not (checkpoint / "COMPLETE.json").exists():
        raise ValueError("Incomplete checkpoint")

    environment = Path(bundle).resolve() if bundle is not None else checkpoint / "environment"
    env = FreeThrowEnv(environment, record=True)
    vector = DummyVecEnv([lambda: env])
    norm = VecNormalize.load(checkpoint / "normalization.pkl", vector)
    norm.training = False
    norm.norm_reward = False
    model = PPO.load(checkpoint / "policy.zip", device="cpu")

    obs, _ = env.reset(seed=42)
    done = False
    actions = []
    total = 0.0
    while not done:
        action, _ = model.predict(norm.normalize_obs(obs[None, :]), deterministic=True)
        actions.append([float(env.d.time), *action[0].tolist()])
        obs, reward, done, _, info = env.step(action[0])
        total += reward

    metrics = dict(
        info["metrics"],
        return_value=float(total),
        checkpoint=str(checkpoint),
        environment=str(environment),
        training_steps=model.num_timesteps,
    )
    (output / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savetxt(output / "policy_actions.csv", actions, delimiter=",",
               header="time_s,elbow,wrist,forearm,fingers,detach", comments="")
    records = env.records
    history = env.ball_history
    np.savez_compressed(
        output / "evaluation.npz",
        times=[r[0] for r in records], qpos=[r[1] for r in records], qvel=[r[2] for r in records],
        errors=[r[3] for r in records], weld_active=[r[4] for r in records],
        ball_times=[b[0] for b in history], ball_positions=[b[1] for b in history],
        ball_velocities=[b[2] for b in history], filtered_actions=env.applied_history,
    )
    (output / "scene.xml").write_text(env.xml, encoding="utf-8")

    if render and records:
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw

        m = env.m
        data = mujoco.MjData(m)
        times = np.array([r[0] for r in records])
        positions = np.array([b[1] for b in history])
        ball_times = np.array([b[0] for b in history])

        camera = mujoco.MjvCamera()
        camera.lookat[:] = [1.8, 0, 1.6]
        camera.distance = 7.3
        camera.azimuth = 100
        camera.elevation = -12
        with mujoco.Renderer(m, height=640, width=960) as renderer, imageio.get_writer(
            str(output / "evaluation.mp4"), fps=30, codec="libx264", quality=8, macro_block_size=16
        ) as writer:
            for time in np.arange(0, times[-1], 1 / 30):
                k = int(np.argmin(abs(times - time)))
                data.qpos[:] = records[k][1]
                data.qvel[:] = records[k][2]
                data.eq_active[:] = records[k][4]
                mujoco.mj_forward(m, data)
                renderer.update_scene(data, camera)
                if metrics["release_time_s"] is not None:
                    points = positions[(ball_times >= metrics["release_time_s"]) & (ball_times <= time)][::25]
                    for point in points:
                        scene = renderer.scene
                        if scene.ngeom >= scene.maxgeom:
                            break
                        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                           np.array([0.012] * 3), point, np.eye(3).ravel(),
                                           np.array([1, 0.6, 0, 1], dtype=np.float32))
                        scene.ngeom += 1
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 960, 65), fill=(18, 27, 42))
                draw.text((15, 12), f"PPO checkpoint: {model.num_timesteps} steps | t={time:.2f}s | physical contact launch", fill="white")
                draw.text((15, 35), f'Scored={metrics["scored"]} | physics accepted={metrics["accepted"]} | apex={metrics["apex_m"]:.2f}m', fill=(255, 180, 65))
                writer.append_data(np.asarray(frame))
                if abs(time - (metrics["release_time_s"] or 1.7)) < 1 / 60:
                    frame.save(output / "release.png")

        release = metrics["release_time_s"] or 1.7
        close_camera = mujoco.MjvCamera()
        close_camera.lookat[:] = [0.30, -0.05, 2.04]
        close_camera.distance = 1.25
        close_camera.azimuth = 125
        close_camera.elevation = -7
        storyboard = Image.new("RGB", (1440, 480))
        with mujoco.Renderer(m, height=480, width=480) as renderer:
            panels = [("BEFORE", release - 0.12), ("RELEASE", release), ("AFTER", release + 0.12)]
            for panel, (label, time) in enumerate(panels):
                k = int(np.argmin(abs(times - time)))
                data.qpos[:] = records[k][1]
                data.qvel[:] = records[k][2]
                data.eq_active[:] = records[k][4]
                mujoco.mj_forward(m, data)
                renderer.update_scene(data, close_camera)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 480, 42), fill=(18, 27, 42))
                draw.text((14, 13), f"{label}  t={times[k]:.3f}s", fill=(255, 180, 65))
                storyboard.paste(frame, (panel * 480, 0))
        storyboard.save(output / "release_closeup.png")

        page = """<!doctype html><html lang="ko"><meta charset="utf-8">
<title>잔차 PPO 평가</title><body style="font:16px system-ui;max-width:1000px;margin:30px auto">
<h1>잔차 PPO 체크포인트 평가</h1><video src="evaluation.mp4" controls style="width:100%"></video>
<h2>릴리스 전후 손 자세</h2><img src="release_closeup.png" style="width:100%">
<p>결정론적 정책으로 실행한 실제 물리 궤적입니다. 준비 구간은 weld, 해제 이후에는 자유 비행 접촉을 사용합니다.</p>
<p><a href="evaluation.json">평가 수치</a> · <a href="policy_actions.csv">정책 행동</a> · <a href="evaluation.npz">물리 상태</a></p></body></html>"""
        (output / "review.html").write_text(page, encoding="utf-8")

    norm.close()
    print(json.dumps(metrics), flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--bundle", type=Path, help="Optional compatible reference bundle for transfer evaluation")
    args = parser.parse_args()
    run(args.checkpoint, args.output, args.render, args.bundle)
