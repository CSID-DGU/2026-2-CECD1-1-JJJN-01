"""Headless physics checks and optional actual MuJoCo renders."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from humanoid_model import ROOT, MODEL_PATH, load_model, reset_pose


def run_checks(path=MODEL_PATH, seconds=2.):
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError('Duration must be positive and finite')
    model = load_model(path)
    data = mujoco.MjData(model)
    reset_pose(model, data)
    assert model.nbody == 53, 'Expected world plus 52 skeleton bodies'
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert not np.any(model.actuator_trnid[:, 0] == 0), 'Root must not be actuated'
    assert model.neq == 0 and not np.any(model.body_gravcomp)
    assert np.all(model.body_mass[1:] > 0) and np.all(model.body_inertia[1:] > 0)
    assert data.ncon == 0, 'Initial standing pose has unexpected penetrations'
    root_initial = float(data.qpos[2])
    peak_foot_force = 0.
    deepest_contact = 0.
    peak_speed = 0.
    wrench = np.zeros(6)
    floor = model.geom('floor').id
    feet = {model.geom(side + '_' + part + '_geom').id
            for side in ['left', 'right'] for part in ['ankle', 'foot']}
    steps = int(np.ceil(seconds / model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        assert not np.any(data.warning.number), 'MuJoCo numerical warning'
        peak_speed = max(peak_speed, float(np.max(np.abs(data.qvel))))
        for idx, contact in enumerate(data.contact):
            deepest_contact = max(deepest_contact, -float(contact.dist))
            if (contact.geom1 == floor and contact.geom2 in feet) or (contact.geom2 == floor and contact.geom1 in feet):
                mujoco.mj_contactForce(model, data, idx, wrench)
                peak_foot_force = max(peak_foot_force, float(wrench[0]))
        if model.nu:
            assert np.all(data.actuator_force <= model.actuator_forcerange[:, 1] + 1e-6)
            assert np.all(data.actuator_force >= model.actuator_forcerange[:, 0] - 1e-6)
    assert peak_foot_force > 0, 'No load-bearing foot contact detected'
    assert abs(data.time - steps * model.opt.timestep) < 1e-7, 'Simulation reset unexpectedly'
    assert deepest_contact < .04, 'Excessive contact penetration'
    return {
        'status': 'passed', 'mass_kg': float(model.body_mass.sum()),
        'skeleton_bodies': model.nbody - 1, 'nq': model.nq, 'nv': model.nv,
        'actuators': model.nu, 'simulated_seconds': float(data.time),
        'root_initial_height_m': root_initial, 'root_final_height_m': float(data.qpos[2]),
        'peak_individual_foot_contact_force_n': peak_foot_force,
        'deepest_contact_m': deepest_contact, 'peak_generalized_speed': peak_speed,
        'warning_counts': data.warning.number.tolist(),
        'balance_controller': False,
        'interpretation': 'Numerical and contact validation only. Falling under fixed joint targets is expected.',
    }


def render_preview(path, output):
    from PIL import Image, ImageDraw
    model = load_model(path)
    data = mujoco.MjData(model)
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0
    tiles = []
    with mujoco.Renderer(model, height=720, width=640) as renderer:
        for pose, azimuth, title in [
            ('t_pose', 0, 'T pose / front'),
            ('standing', 125, 'Initial pose / oblique'),
            ('standing', 90, 'Initial pose / side'),
        ]:
            reset_pose(model, data, pose)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [0, 0, .9]
            camera.distance = 3.2
            camera.azimuth = azimuth
            camera.elevation = -12
            renderer.update_scene(data, camera=camera, scene_option=option)
            tile = Image.fromarray(renderer.render())
            ImageDraw.Draw(tile).text((16, 16), title, fill='white')
            tiles.append(tile)
    canvas = Image.new('RGB', (1920, 720))
    for i, tile in enumerate(tiles):
        canvas.paste(tile, (640 * i, 0))
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=MODEL_PATH)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/humanoid_check')
    parser.add_argument('--seconds', type=float, default=2.)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    report = run_checks(args.model, args.seconds)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.render:
        render_preview(args.model, args.output / 'preview.png')
        report['preview'] = str(args.output / 'preview.png')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
