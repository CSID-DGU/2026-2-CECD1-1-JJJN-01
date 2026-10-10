"""M3-1: a shared 3-second synthetic free throw, PD + COM, no PPO."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from examples.m3_control import reference, run, write_csv
from examples.smplh_bodies import SHAPES
from smpl_sim.motion import SMPLHMotion
from smpl_sim.motion.synthetic_free_throw import generate, stage_at, STAGES, TRACKED_JOINTS


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def annotate_and_summarize(output, body):
    prefix = f'{body}_synthetic_free_throw'
    time_path = output / f'{prefix}_timeseries.csv'
    rows = read_csv(time_path)
    for row in rows:
        row['phase'] = stage_at(float(row['time_s']))
    write_csv(time_path, rows)
    joint_path = output / f'{prefix}_joint_tracking.csv'
    joints = read_csv(joint_path)
    for row in joints:
        row['phase'] = stage_at(float(row['time_s']))
    write_csv(joint_path, joints)
    phase_rows = []
    for phase, start, end in STAGES:
        measured = [r for r in rows if r['phase'] == phase]
        if not measured:
            continue
        phase_rows.append({
            'shape': body, 'phase': phase, 'start_s': start, 'end_s': end,
            'samples': len(measured),
            'body_joint_rmse_rad': float(np.sqrt(np.mean([float(r['body_joint_rmse_rad'])**2 for r in measured]))),
            'task_joint_rmse_rad': float(np.sqrt(np.mean([float(r['motion_joint_rmse_rad'])**2 for r in measured]))),
            'balance_fraction': float(np.mean([int(r['balanced']) for r in measured])),
            'both_feet_contact_fraction': float(np.mean([int(r['left_contact']) and int(r['right_contact']) for r in measured])),
            'max_pelvis_tilt_deg': max(float(r['pelvis_tilt_deg']) for r in measured),
            'fallen_any': max(int(r['fallen']) for r in measured),
        })
    write_csv(output / f'{body}_phase_summary.csv', phase_rows)
    joint_summary = []
    for name in TRACKED_JOINTS:
        entries = [r for r in joints if r['joint'] == name]
        errors = np.array([float(r['error_rad']) for r in entries])
        actual = np.array([float(r['actual_rad']) for r in entries])
        targets = np.array([float(r['target_rad']) for r in entries])
        joint_summary.append({'shape': body, 'joint': name, 'rmse_rad': float(np.sqrt(np.mean(errors**2))),
                              'max_abs_error_rad': float(np.abs(errors).max()),
                              'actual_range_rad': float(np.ptp(actual)), 'target_range_rad': float(np.ptp(targets))})
    write_csv(output / f'{body}_task_joint_summary.csv', joint_summary)
    return phase_rows, joint_summary


def check(summary, joints):
    # Validation criteria concern pipeline tracking, not shooting technique.
    if abs(float(summary['duration_s']) - 3.) > 1e-5:
        raise RuntimeError('Incomplete 3-second rollout')
    if int(summary['fallen_any']) or int(summary['mujoco_warnings']):
        raise RuntimeError(f'Fall or simulation warning: {summary}')
    if float(summary['balance_fraction']) < .95 or float(summary['both_feet_contact_fraction']) < .95:
        raise RuntimeError(f'Balance/contact check failed: {summary}')
    if float(summary['body_joint_rmse_rad']) > .10 or float(summary['motion_joint_rmse_rad']) > .20:
        raise RuntimeError(f'Tracking check failed: {summary}')
    # Confirm that physics actually responded to each requested movement.
    minima = {'Knee': .10, 'Shoulder': .65, 'Elbow': .40, 'Wrist': .15}
    for row in joints:
        for part, minimum in minima.items():
            if f'_{part}_' in row['joint'] and row['actual_range_rad'] < minimum:
                raise RuntimeError(f'Insufficient movement: {row}')


def write_report(output, summaries, source_hash):
    lines = [
        '# M3-1 Synthetic Free-Throw Validation', '',
        '**Temporary synthetic pipeline data. Not real basketball ground truth.**', '',
        'One shared SMPL+H NPZ: 3.0 seconds, 30 FPS, 91 samples including both endpoints.',
        '450 Hz physics, 30 Hz metrics, 1350 physics steps / 90 metric samples per body.',
        'M3 PD + COM feedback is reused. PPO, external support and mid-motion resets are absent.',
        'Only a constant initial root-floor offset is applied per shape.', '',
        f'Source SHA-256: `{source_hash}`', '',
        '| Shape | Body RMSE rad | Task RMSE rad | Balance % | Both feet % | Max pelvis tilt deg | Fall |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in summaries:
        lines.append(f"| {row['shape']} | {float(row['body_joint_rmse_rad']):.5f} | "
                     f"{float(row['motion_joint_rmse_rad']):.5f} | {100*float(row['balance_fraction']):.1f} | "
                     f"{100*float(row['both_feet_contact_fraction']):.1f} | "
                     f"{float(row['max_pelvis_tilt_deg']):.2f} | {int(row['fallen_any'])} |")
    lines += ['', '## Phases', '', '| Phase | Start s | End s |', '|---|---:|---:|']
    for name, start, end in STAGES:
        lines.append(f'| {name} | {start:.2f} | {end:.2f} |')
    lines += [
        '', '## Files and interpretation', '',
        '- summary.csv: complete 3-second rollouts; no warmup interval discarded.',
        '- phase_summary.csv: phase-specific error, balance, contacts and falls.',
        '- task_joint_summary.csv: Hip_x, Knee_x, Shoulder_x, Elbow_y, Wrist_x for both sides.',
        '- Per-body joint_tracking.csv: every hinge at 30 Hz, original/effective targets, angles and torques.',
        '- Per-body timeseries.csv: contact normal forces, support footprint margin, pelvis tilt and root height.',
        '- Per-body reference.npz: source angular motion plus constant floor alignment.',
        '- reference_metadata.json: explicit synthetic-data provenance and controller configuration.',
        '', 'Body RMSE excludes fingers and floating root. Task RMSE measures the 10 moving joints.',
        'Balance is a 30 Hz proxy using pelvis tilt, height, floor contacts and projected foot-box support.',
        'It is not a CoP/ZMP or perturbation robustness guarantee. Root trajectory is not servoed.',
        'Elbows flex during preparation before extending; wrists cock, flick and follow through.',
        'No basketball, hoop, ball release, finger-grasp validation, real motion capture or shot success is included.',
        '', 'See M3_1_SYNTHETIC_FREE_THROW.md in the repository for commands and full metric definitions.',
    ]
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--body', choices=SHAPES, default='body_01')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--source', type=Path, help='Existing shared synthetic NPZ; preserve its contents')
    parser.add_argument('--models', type=Path, default=ROOT / 'outputs/smplh_physics')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/m3_free_throw')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve() if args.source else (args.output / 'synthetic_free_throw.npz').resolve()
    if not args.source:
        model = mujoco.MjModel.from_xml_path(str(args.models / 'body_01.xml'))
        root = reference('stand', model).root_position[0]
        generate(root).save(source)
    motion = SMPLHMotion.load(source)
    if len(motion.poses) != 91 or motion.fps != 30 or abs(motion.duration - 3.) > 1e-9:
        parser.error('Expected a 91-sample, 30 FPS, 3-second synthetic source')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = {'purpose': 'SYNTHETIC_PIPELINE_VALIDATION_ONLY', 'basketball_ground_truth': False,
                'mujoco_version': mujoco.__version__, 'fps': 30, 'duration_s': 3., 'source_samples': 91,
                'source_path': str(source), 'source_sha256': source_hash,
                'phases': [{'name': n, 'start_s': s, 'end_s': e} for n, s, e in STAGES],
                'controller': 'M3 BalancedPD (unchanged gains and COM rule)',
                'root_retargeting': 'constant initial floor alignment only; no root actuator',
                'measurement_start_s': 0., 'physics_hz': 450, 'metrics_hz': 30}
    (args.output / 'reference_metadata.json').write_text(json.dumps(metadata, indent=2))
    bodies = list(SHAPES) if args.all else [args.body]
    if args.gui and len(bodies) > 1:
        launcher = Path(sys.prefix) / 'bin' / ('mjpython' if sys.platform == 'darwin' else 'python')
        for body in bodies:
            cmd = [str(launcher), str(Path(__file__).resolve()), '--body', body, '--gui', '--source', str(source),
                   '--models', str(args.models.resolve()), '--output', str(args.output.resolve())]
            if args.check:
                cmd.append('--check')
            subprocess.run(cmd, check=True)
        summaries = [read_csv(args.output / f'{b}_synthetic_free_throw_summary.csv')[0] for b in bodies]
        phase_rows = [r for b in bodies for r in read_csv(args.output / f'{b}_phase_summary.csv')]
        joint_rows = [r for b in bodies for r in read_csv(args.output / f'{b}_task_joint_summary.csv')]
    else:
        summaries, phase_rows, joint_rows = [], [], []
        for body in bodies:
            opts = SimpleNamespace(models=args.models, motion_file=source, output=args.output,
                                   align_floor=True, gui=args.gui, frames=None, no_balance=False,
                                   motion_label='synthetic_free_throw', summary_start_time=0.,
                                   tracking_joint_names=TRACKED_JOINTS, joint_log_stride=1)
            summary = run(body, 'stand', opts)
            phases, joints = annotate_and_summarize(args.output, body)
            if args.check:
                check(summary, joints)
            summaries.append(summary); phase_rows.extend(phases); joint_rows.extend(joints)
    write_csv(args.output / 'summary.csv', summaries)
    write_csv(args.output / 'phase_summary.csv', phase_rows)
    write_csv(args.output / 'task_joint_summary.csv', joint_rows)
    # Only the root position may change per body; the source sequence stays identical.
    for body in bodies:
        applied = SMPLHMotion.load(args.output / f'{body}_synthetic_free_throw_reference.npz')
        np.testing.assert_array_equal(applied.poses, motion.poses)
        offset = applied.root_position - motion.root_position
        np.testing.assert_allclose(offset, np.tile(offset[0], (91, 1)), atol=1e-12)
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError('Source motion was modified during tracking')
    write_report(args.output, summaries, source_hash)
    print(f'{len(bodies)} continuous synthetic rollouts completed; common source verified', flush=True)


if __name__ == '__main__':
    main()
