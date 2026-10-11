"""Replay a recorded residual free throw in the interactive MuJoCo viewer."""
import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def replay(evaluation: Path, speed: float, loop: bool) -> None:
    if speed <= 0:
        raise ValueError("--speed must be positive")
    model = mujoco.MjModel.from_xml_path(str(evaluation / "scene.xml"))
    data = mujoco.MjData(model)
    with np.load(evaluation / "evaluation.npz") as trajectory:
        times = trajectory["times"]
        qpos = trajectory["qpos"]
        qvel = trajectory["qvel"]
        weld_active = trajectory["weld_active"]
    if not (len(times) == len(qpos) == len(qvel) == len(weld_active)):
        raise ValueError("Trajectory arrays have different lengths")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [1.8, 0, 1.6]
        viewer.cam.distance = 7.3
        viewer.cam.azimuth = 100
        viewer.cam.elevation = -12
        while viewer.is_running():
            started = time.monotonic()
            for i, t in enumerate(times):
                if not viewer.is_running():
                    return
                remaining = (float(t) - float(times[0])) / speed - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
                data.qpos[:] = qpos[i]
                data.qvel[:] = qvel[i]
                data.eq_active[:] = weld_active[i]
                mujoco.mj_forward(model, data)
                viewer.sync()
            if not loop:
                while viewer.is_running():
                    time.sleep(0.05)
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    replay(args.evaluation, args.speed, args.loop)
