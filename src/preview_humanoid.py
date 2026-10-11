"""Inspect the humanoid without simulation, or simulate its unbalanced PD pose.

Default: static interactive inspection. --simulate enables gravity and stepping.
Closing the window ends the process. No trained balance controller is present.
"""
import argparse
import time
import mujoco
import mujoco.viewer
from humanoid_model import MODEL_PATH, load_model, reset_pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=MODEL_PATH)
    parser.add_argument('--pose', choices=['standing', 't_pose'], default='standing')
    parser.add_argument('--simulate', action='store_true')
    args = parser.parse_args()
    model = load_model(args.model)
    data = mujoco.MjData(model)
    reset_pose(model, data, args.pose)
    print('Physics enabled; falling is expected without balance control.' if args.simulate
          else 'Static inspection only. Use --simulate to enable physics.')
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0, 0, .9]
        viewer.cam.distance = 3.1
        viewer.cam.azimuth = 130
        viewer.cam.elevation = -15
        viewer.opt.sitegroup[:] = 0
        while viewer.is_running():
            start = time.perf_counter()
            with viewer.lock():
                if args.simulate:
                    for _ in range(10):
                        mujoco.mj_step(model, data)
                else:
                    mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(max(0, .01 - (time.perf_counter() - start)))


if __name__ == '__main__':
    main()
