"""Live PD + COM preview in one GUI window; reset physics between repetitions.

This preview does not write evaluation CSVs. Use m3_spl_physics.py for metrics.
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import mujoco.viewer
import numpy as np
from examples.m3_control import BalancedPD, ground
from examples.m3_spl_physics import skeleton
from smpl_sim.motion.smplh_motion import SMPLHMotion


def run(args):
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    model = mujoco.MjModel.from_xml_path(str(args.model))
    model.opt.timestep = 1/450
    motion = SMPLHMotion.load(args.source)
    bound = motion.bind(model)
    data, reference = mujoco.MjData(model), mujoco.MjData(model)
    initial = bound.sample(0)
    data.qpos[:] = initial
    ground(model, data)
    offset = float(data.qpos[2]-initial[2])

    def target(t):
        q = bound.sample(t)
        q[2] += offset
        return q

    frame_steps = np.ceil(np.arange(len(motion.poses))/motion.fps/model.opt.timestep-1e-9).astype(int)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0, 0, 1]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3., 135., -15.
        repeat = 0
        while viewer.is_running() and (args.repeats == 0 or repeat < args.repeats):
            with viewer.lock():
                mujoco.mj_resetData(model, data)
                data.qpos[:] = target(0)
                mujoco.mj_forward(model, data)
                controller = BalancedPD(model, bound.mapping)
            repeat += 1
            print(f'Repetition {repeat} started (same GUI window)', flush=True)
            wall_start = time.monotonic()
            completed_steps = 0
            for goal_step in frame_steps:
                if not viewer.is_running():
                    return
                with viewer.lock():
                    while completed_steps < goal_step:
                        desired = target(data.time+model.opt.timestep)
                        data.ctrl[:], _ = controller.control(data, desired)
                        previous_time = float(data.time)
                        mujoco.mj_step(model, data)
                        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or data.time <= previous_time:
                            raise RuntimeError('Nonfinite state or MuJoCo time reset')
                        mujoco.mj_forward(model, data)
                        completed_steps += 1
                    viewer.user_scn.ngeom = 0
                    if not args.no_reference:
                        reference.qpos[:] = target(data.time)
                        mujoco.mj_forward(model, reference)
                        skeleton(viewer.user_scn, model, reference)
                viewer.sync()
                time.sleep(max(0, float(data.time)-(time.monotonic()-wall_start)))
            warnings = sum(w.number for w in data.warning)
            print(f'Repetition {repeat} finished: {data.time:.6f}s, MuJoCo warnings={warnings}', flush=True)
            if warnings:
                raise RuntimeError('MuJoCo physics warning; stopped preview')
            if hashlib.sha256(args.source.read_bytes()).hexdigest() != source_hash:
                raise RuntimeError('Source motion changed during preview')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'outputs/m3_spl_fitting/spl_P0002_T0049.npz')
    parser.add_argument('--model', type=Path, default=ROOT.parents[2]/'basketball-r1/SMPLSim/outputs/smplh_physics/body_01.xml')
    parser.add_argument('--repeats', type=int, default=0, help='0: until GUI closes; positive: repetition count')
    parser.add_argument('--no-reference', action='store_true', help='Hide green reference skeleton')
    args = parser.parse_args()
    if args.repeats < 0:
        parser.error('--repeats must be >= 0')
    try:
        run(args)
    except KeyboardInterrupt:
        print('Preview stopped.', flush=True)
