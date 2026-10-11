"""Render labeled thumb markers to verify left/right hand handedness."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw


def render(evaluation, output):
    evaluation = Path(evaluation)
    output = Path(output)
    model = mujoco.MjModel.from_xml_string((evaluation / "scene.xml").read_text(encoding="utf-8"))
    run = np.load(evaluation / "evaluation.npz")
    metrics = json.loads((evaluation / "evaluation.json").read_text(encoding="utf-8"))
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.30, -0.05, 2.04]
    camera.distance = 1.25
    camera.azimuth = 125
    camera.elevation = -7
    release = float(metrics["release_time_s"])
    samples = [("BEFORE", release - .13), ("RELEASE", release), ("AFTER", release + .11)]
    canvas = Image.new("RGB", (1440, 520), (18, 27, 42))
    measurements = []
    with mujoco.Renderer(model, height=480, width=480) as renderer:
        for panel, (label, time) in enumerate(samples):
            k = int(np.argmin(abs(run["times"] - time)))
            data.qpos[:] = run["qpos"][k]
            data.qvel[:] = run["qvel"][k]
            data.eq_active[:] = run["weld_active"][k]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera)
            scene = renderer.scene
            for side, color in [("right", [1., .05, .05, 1.]), ("left", [0., .9, 1., 1.])]:
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                   np.array([.016] * 3), data.body(side + "_thumb1").xpos,
                                   np.eye(3).ravel(), np.array(color, dtype=np.float32))
                scene.ngeom += 1
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 480, 42), fill=(18, 27, 42))
            draw.text((14, 13), f"{label}  t={run['times'][k]:.3f}s", fill=(255, 180, 65))
            canvas.paste(frame, (panel * 480, 0))

            cam = scene.camera[0]
            screen_right = np.cross(np.asarray(cam.forward), np.asarray(cam.up))
            screen_right /= np.linalg.norm(screen_right)
            row = {"time_s": float(run["times"][k])}
            for side in ["right", "left"]:
                delta = data.body(side + "_thumb1").xpos - data.body(side + "_pinky1").xpos
                row[side + "_thumb_minus_pinky_screen_horizontal_m"] = float(delta @ screen_right)
            measurements.append(row)
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 492), "RED = anatomical RIGHT thumb (shooting hand)    CYAN = anatomical LEFT thumb (guide hand)", fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    output.with_suffix(".json").write_text(json.dumps(measurements, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.evaluation, args.output)
