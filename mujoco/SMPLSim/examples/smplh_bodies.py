"""Generate, validate and view four SMPL+H physics bodies on CPU."""
import argparse
import csv
import json
import sys
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import mujoco.viewer
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from smpl_sim.envs.humanoid_env import HumanoidEnv
from smpl_sim.envs.controllers import StablePDController
from smpl_sim.smpllib.smpl_parser import SMPLH_Parser
from smpl_sim.smpllib.smpl_mujoco_new import smpl_to_qpose

SHAPES = {
    'body_01': [0.] * 16,
    'body_02': [2.] + [0.] * 15,
    'body_03': [-2.] + [0.] * 15,
    'body_04': [0., 2.] + [0.] * 14,
}


def initial_state(model, data, clearance=0.01):
    """Place the upright zero pose above the plane using collision geometry."""
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = 0
    data.qpos[3:7] = .5  # SMPL Y-up -> MuJoCo Z-up, same convention as SMPLSim.
    mujoco.mj_forward(model, data)
    bottom = []
    for i in range(model.ngeom):
        if model.geom_bodyid[i] == 0:
            continue
        size, rotation = model.geom_size[i], data.geom_xmat[i].reshape(3, 3)
        typ = model.geom_type[i]
        if typ == mujoco.mjtGeom.mjGEOM_BOX:
            extent = np.abs(rotation[2]) @ size
        elif typ == mujoco.mjtGeom.mjGEOM_CAPSULE:
            extent = size[0] + abs(rotation[2, 2]) * size[1]
        elif typ == mujoco.mjtGeom.mjGEOM_SPHERE:
            extent = size[0]
        else:
            raise ValueError(f'Unsupported collision type: {typ}')
        bottom.append(data.geom_xpos[i, 2] - extent)
    data.qpos[2] = clearance - min(bottom)
    mujoco.mj_forward(model, data)


def controller(model):
    ids = model.actuator_trnid[:, 0]
    gains = model.jnt_user[ids]
    # Preserve body gains; conservative finger gains for the isolated demo.
    gains = gains.copy()
    for i, jid in enumerate(ids):
        name = model.joint(int(jid)).name
        if any(f'_{finger}' in name for finger in ('Index', 'Middle', 'Pinky', 'Ring', 'Thumb')):
            gains[i, :2] = [5., .5]
            gains[i, 3] = 2.
    return StablePDController(np.ones(model.nu), np.zeros(model.nu), model.nv,
                              gains[:, 3], gains[:, 0], gains[:, 1])


def rollout(model, steps):
    data = mujoco.MjData(model)
    initial_state(model, data)
    ctrl = controller(model)
    start = data.qpos.copy()
    for _ in range(steps):
        data.ctrl[:] = ctrl.control(np.zeros(model.nu), model, data)
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    assert not data.warning.number.any(), data.warning.number
    return {'steps': steps, 'time_s': float(data.time),
            'qpos_delta_l2': float(np.linalg.norm(data.qpos - start)),
            'contacts': int(data.ncon), 'warnings': data.warning.number.tolist()}


def finger_test(model):
    fingers = [b for b in range(1, model.nbody)
               if any(f'_{f}' in model.body(b).name for f in ('Index', 'Middle', 'Pinky', 'Ring', 'Thumb'))]
    assert len(fingers) == 30
    joints = [j for j in range(model.njnt) if model.jnt_bodyid[j] in fingers]
    assert len(joints) == 90
    for j in joints:
        assert np.count_nonzero(model.actuator_trnid[:, 0] == j) == 1
    # Isolate actuator response in air with gravity disabled; all joints tested.
    gravity = model.opt.gravity.copy()
    model.opt.gravity[:] = 0
    try:
        baseline, driven = mujoco.MjData(model), mujoco.MjData(model)
        initial_state(model, baseline)
        initial_state(model, driven)
        baseline.qpos[2] += 3
        driven.qpos[2] += 3
        driven.ctrl[:] = 0
        for j in joints:
            aid = int(np.flatnonzero(model.actuator_trnid[:, 0] == j)[0])
            driven.ctrl[aid] = .002
        for _ in range(20):
            mujoco.mj_step(model, baseline)
            mujoco.mj_step(model, driven)
        diff = np.abs(driven.qpos[model.jnt_qposadr[joints]] - baseline.qpos[model.jnt_qposadr[joints]])
        assert np.isfinite(diff).all() and np.all(diff > 1e-8), diff
        assert not driven.warning.number.any()
    finally:
        model.opt.gravity[:] = gravity
    return {'bodies': len(fingers), 'hinges': len(joints), 'actuators': len(joints),
            'min_actuated_angle_delta_rad': float(diff.min()),
            'max_actuated_angle_delta_rad': float(diff.max())}


def generate(out, model_dir):
    out.mkdir(parents=True, exist_ok=True)
    snapshots, summary, body_rows, geom_rows = {}, {}, [], []
    for name, betas in SHAPES.items():
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'smpl_sim/data/cfg')):
            cfg = compose(config_name='config', overrides=['robot=smplh_humanoid', 'headless=True'])
        cfg.robot.smpl_data_dir = str(model_dir)
        cfg.robot.betas = betas
        env = HumanoidEnv(cfg)
        try:
            env.reset(seed=0)
            assert env.robot is not None and env.robot.smpl_parser.num_betas == 16
            torch.testing.assert_close(env.robot.beta, torch.tensor([betas], dtype=torch.float32))
            (out / f'{name}.xml').write_text(env.default_xml_str)
            m = mujoco.MjModel.from_xml_path(str(out / f'{name}.xml'))
            m.opt.timestep = env.sim_timestep
            d = mujoco.MjData(m)
            initial_state(m, d)
            # Converter is for pose ordering only; beta is applied by robot generation.
            converted = smpl_to_qpose(torch.zeros(1, 156), m, model='smplh', euler_order='XYZ')
            assert converted.shape == (1, m.nq)
            assert np.isfinite(m.body_mass).all() and np.all(m.body_mass[1:] > 0)
            assert np.isfinite(m.body_inertia).all() and np.all(m.body_inertia[1:] > 0)
            length = np.linalg.norm(m.body_pos[1:], axis=1)
            robot_geoms = np.flatnonzero(m.geom_bodyid > 0)
            snapshot = {'body_local_pos': m.body_pos[1:].copy(), 'body_world_pos': d.xpos[1:].copy(),
                        'segment_length': length, 'geom_size': m.geom_size[robot_geoms].copy(),
                        'geom_pos': m.geom_pos[robot_geoms].copy(), 'mass': m.body_mass[1:].copy(),
                        'inertia': m.body_inertia[1:].copy()}
            snapshots[name] = snapshot
            stats = {'betas': betas, 'gender': 'male', 'nbody_human': m.nbody-1, 'nq': m.nq,
                     'nv': m.nv, 'actuators': m.nu, 'collision_geoms': len(robot_geoms),
                     'mass_kg': float(m.body_mass.sum()), 'initial_root_z_m': float(d.qpos[2]),
                     'fingers': finger_test(m), 'physics': rollout(m, 1350)}
            # Also exercise full Gym/SMPLSim observations and its PD controller.
            for _ in range(30):
                obs, *_ = env.step(np.zeros(env.action_space.shape))
                assert np.isfinite(obs).all()
            assert not env.mj_data.warning.number.any()
            stats['humanoid_env_control_steps'] = 30
            summary[name] = stats
            for i in range(1, m.nbody):
                body_rows.append([name, m.body(i).name, *m.body_pos[i], *d.xpos[i],
                                  length[i-1], m.body_mass[i], *m.body_inertia[i]])
            for i in robot_geoms:
                geom_rows.append([name, m.body(int(m.geom_bodyid[i])).name, int(m.geom_type[i]),
                                  *m.geom_size[i], *m.geom_pos[i], int(m.geom_contype[i]), int(m.geom_conaffinity[i])])
            print(f'{name}: mass={stats["mass_kg"]:.3f}kg; fingers=90 hinges; physics passed', flush=True)
        finally:
            env.close()
            if env.robot is not None:
                env.robot.remove_geoms()
    base = snapshots['body_01']
    for name, snap in snapshots.items():
        delta = {key: float(np.max(np.abs(snap[key] - base[key]))) for key in base}
        summary[name]['max_abs_delta_from_body_01'] = delta
        if name != 'body_01':
            assert all(delta[key] > 1e-9 for key in ('body_local_pos', 'segment_length', 'geom_size', 'mass', 'inertia'))
    # Verify hand pose keyword correction actually changes vertices.
    parser = SMPLH_Parser(model_path=str(model_dir), gender='male', num_betas=16,
                          use_pca=False, create_transl=False)
    zero, hand = torch.zeros(1, 156), torch.zeros(1, 156)
    hand[0, 66] = .3
    with torch.no_grad():
        v0, _ = parser.get_joints_verts(zero)
        v1, _ = parser.get_joints_verts(hand)
    pose_delta = float((v1-v0).abs().max())
    assert pose_delta > 1e-5
    report = {'mujoco_version': mujoco.__version__, 'hand_pose_vertex_delta_m': pose_delta, 'bodies': summary}
    (out / 'validation.json').write_text(json.dumps(report, indent=2))
    for filename, header, rows in (
        ('body_properties.csv', ['shape','body','local_x','local_y','local_z','world_x','world_y','world_z','parent_offset_length_m','mass_kg','inertia_x','inertia_y','inertia_z'], body_rows),
        ('collision_properties.csv', ['shape','body','geom_type','size_0','size_1','size_2','local_x','local_y','local_z','contype','conaffinity'], geom_rows),
    ):
        with (out / filename).open('w', newline='') as f:
            writer = csv.writer(f); writer.writerow(header); writer.writerows(rows)
    print(f'Artifacts: {out}', flush=True)


def view(path, headless, steps):
    model = mujoco.MjModel.from_xml_path(str(path))
    model.opt.timestep = 1 / 450
    if headless:
        print(json.dumps(rollout(model, steps or 1350), indent=2))
        return
    data = mujoco.MjData(model)
    initial_state(model, data)
    ctrl = controller(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0, 0, 1]
            viewer.cam.distance = 3
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -15
        frame = 0
        while viewer.is_running() and (steps == 0 or frame < steps):
            begin = time.monotonic()
            with viewer.lock():
                for _ in range(15):
                    data.ctrl[:] = ctrl.control(np.zeros(model.nu), model, data)
                    mujoco.mj_step(model, data)
                assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
            viewer.sync()
            frame += 1
            if frame % 300 == 0:  # Reset periodically; no balance policy trained yet.
                with viewer.lock():
                    initial_state(model, data)
            time.sleep(max(0, 1/30 - (time.monotonic() - begin)))
        print(f'GUI completed {frame} control frames; warnings={data.warning.number.tolist()}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generate', action='store_true')
    p.add_argument('--body', choices=SHAPES, default='body_01')
    p.add_argument('--all', action='store_true', help='View all four bodies sequentially')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--steps', type=int, default=0, help='GUI control frames; headless physics steps (0 uses defaults)')
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/smplh_physics')
    p.add_argument('--model-dir', type=Path, default=ROOT / 'assets/body_models/smplh')
    args = p.parse_args()
    if args.steps < 0:
        p.error('--steps must be nonnegative')
    if args.generate:
        generate(args.output.resolve(), args.model_dir.resolve())
    elif args.all and not args.headless:
        # macOS passive-viewer teardown is asynchronous. Run each viewer in
        # its own process so the next window never races the previous one.
        launcher = Path(sys.prefix) / "bin" / ("mjpython" if sys.platform == "darwin" else "python")
        for name in SHAPES:
            subprocess.run([str(launcher), str(Path(__file__).resolve()), "--body", name,
                            "--output", str(args.output.resolve()), "--steps", str(args.steps)], check=True)
    else:
        for name in (SHAPES if args.all else [args.body]):
            path = args.output / f'{name}.xml'
            if not path.exists():
                p.error(f'{path} missing: run --generate first')
            print(f'Viewing {name}', flush=True)
            view(path, args.headless, args.steps)


if __name__ == '__main__':
    main()
