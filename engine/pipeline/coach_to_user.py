"""Coach motion -> user's body with PAN (same topology, user's bone lengths).

Both runs come from `pipeline.run_video` (any estimator). Each holds the person's own
skeleton (BigVegas topology, measured bone lengths: observation.npz `person_offsets`)
and the IK motion on it (source_person.bvh). The coach motion is brought to BigVegas
scale, encoded by PAN and decoded with the user's offsets (group-A decoder,
`PanRuntime.retarget_intra`); validated on Mixamo ground truth by `pipeline.intra_check`.

  python -m pipeline.coach_to_user --coach runs/video_wham/squat_diag --user runs/video_wham/<user>
  python -m pipeline.coach_to_user --coach runs/video_wham/squat_diag --user-scale legs=1.15 arms=0.9 \
      --name synth_long_legs        # no user video: coach body with scaled bones

Outputs runs/coach_user/<name>/:
  coach.bvh                coach motion on the coach's body (copy of source_person.bvh)
  coach_on_user_pan.bvh    PAN result on the user's body  <- main output
  coach_on_user_copy.bvh   baseline: coach rotations copied, root scaled by the leg ratio
  user.bvh                 the user's own motion (only with --user)
  report.json, comparison.html
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.core.paths import REPO, enter_repo  # noqa: E402

GROUPS = {'legs': ('LeftUpLeg', 'LeftLeg', 'LeftFoot', 'RightUpLeg', 'RightLeg', 'RightFoot'),
          'spine': ('Spine', 'Spine1', 'Spine2', 'Neck'),
          'arms': ('LeftShoulder', 'LeftArm', 'LeftForeArm', 'RightShoulder', 'RightArm', 'RightForeArm')}
TOES = ('LeftToeBase', 'RightToeBase')


def load_person(run):
    from data_preprocess.Mixamo.bvh_parser import BVH_file
    obs = np.load(run / 'observation.npz')
    raw = BVH_file(str(run / 'source_person.bvh')).to_tensor(quater=True).double()
    return obs['person_offsets'].astype(np.float64), raw


def parse_scale(items):
    out = {}
    for it in items:
        k, v = it.split('=')
        if k not in ('legs', 'arms', 'spine'):
            raise ValueError(f'Unknown scale group {k} (legs/arms/spine)')
        out[k] = float(v)
    return out


def foot_metrics(skel, pos, floor_margin=3.):
    """Lowest toe height and horizontal toe travel while the toe is near the floor (cm)."""
    toes = [skel.index(n) for n in TOES]
    y = pos[:, toes, 1]
    floor = float(y.min())
    slide = []
    for k in range(len(toes)):
        near = y[:, k] < floor + floor_margin
        both = near[1:] & near[:-1]
        step = np.linalg.norm(np.diff(pos[:, toes[k]][:, [0, 2]], axis=0), axis=1)
        slide.append(float(step[both].sum()))
    return dict(toe_min_y=floor, toe_slide_cm=slide)


def rotation_gap(skel, qa, qb):
    from pipeline.core import quat as Q
    d = np.degrees(Q.angle_between(qa, qb).numpy())
    return {g: float(d[:, [skel.index(n) for n in names]].mean()) for g, names in GROUPS.items()}


def main(argv=None):
    from pipeline.core import quat as Q
    from pipeline.harness import perturbed_offsets
    from pipeline.core.pan_runtime import PanRuntime, canonical_sign, ood_stats
    from pipeline.core.skeleton import Skeleton
    from pipeline.run_video import person_writer

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--coach', required=True, help='run_video output directory of the coach')
    ap.add_argument('--user', help='run_video output directory of the user')
    ap.add_argument('--user-scale', nargs='+', help='synthetic user: coach bones scaled, e.g. legs=1.15 arms=0.9')
    ap.add_argument('--name', default=None)
    ap.add_argument('--no-viz', action='store_true')
    ap.add_argument('--out', default=None, help='output directory (default runs/coach_user/<name>)')
    ap.add_argument('--device', default='cuda:0')
    opt = ap.parse_args(argv)
    if bool(opt.user) == bool(opt.user_scale):
        raise SystemExit('Give exactly one of --user or --user-scale')
    coach_run = (Path.cwd() / opt.coach).resolve()
    user_run = (Path.cwd() / opt.user).resolve() if opt.user else None
    out_arg = Path(opt.out).resolve() if opt.out else None       # relative to the caller's cwd
    enter_repo()

    skel = Skeleton.from_character('BigVegas')
    coach_off, coach_raw = load_person(coach_run)
    coach_skel = skel.with_offsets(coach_off)
    if user_run:
        user_off, user_raw = load_person(user_run)
    else:
        user_off, user_raw = perturbed_offsets(coach_skel, parse_scale(opt.user_scale)), None
    user_skel = skel.with_offsets(user_off)
    name = opt.name or (f'{coach_run.name}_to_{user_run.name}' if user_run else
                        f'{coach_run.name}_synth_' + '_'.join(s.replace('=', '') for s in opt.user_scale))
    out = out_arg or REPO / 'runs/coach_user' / name
    out.mkdir(parents=True, exist_ok=True)

    runtime = PanRuntime(source='BigVegas', device=opt.device)
    pan_stats = tuple(s.cpu() for s in runtime.source_stats)
    coach_ratio, user_ratio = runtime.root_ratio(coach_off), runtime.root_ratio(user_off)

    # Coach motion at BigVegas scale (PAN's input domain), quaternion signs as in training.
    src = coach_raw.clone()
    src[-3:] /= coach_ratio
    src = canonical_sign(src, *pan_stats)
    pan = runtime.retarget_intra(src.float(), user_off).cpu().double()
    copy = torch.cat([coach_raw[:-3], coach_raw[-3:] * user_ratio / coach_ratio])

    shutil.copy(coach_run / 'source_person.bvh', out / 'coach.bvh')
    person_writer(skel, user_skel).write_raw(pan.float(), 'quaternion', str(out / 'coach_on_user_pan.bvh'))
    person_writer(skel, user_skel).write_raw(copy.float(), 'quaternion', str(out / 'coach_on_user_copy.bvh'))
    if user_run:
        shutil.copy(user_run / 'source_person.bvh', out / 'user.bvh')

    def motion(sk, raw):
        q, root = sk.raw_to_joint(raw)
        q = Q.normalize(q)
        return q, sk.fk(q, root).numpy()

    q_pan, p_pan = motion(user_skel, pan)
    q_copy, p_copy = motion(user_skel, copy)
    _, p_coach = motion(coach_skel, coach_raw)
    lengths = lambda off: {n: float(np.linalg.norm(off[skel.index(n)])) for n in
                           ('LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftArm', 'LeftForeArm', 'Spine1')}
    report = dict(
        coach=str(coach_run), user=str(user_run) if user_run else None,
        user_scale=parse_scale(opt.user_scale) if opt.user_scale else None,
        frames=int(coach_raw.shape[1]),
        height_cm=dict(coach=coach_skel.height(), user=user_skel.height()),
        leg_ratio_to_bigvegas=dict(coach=coach_ratio, user=user_ratio),
        bone_length_cm=dict(coach=lengths(coach_off), user=lengths(user_off)),
        pan_input=ood_stats(runtime.normalize_source(src.float())),
        pan_vs_copy_rot_deg=rotation_gap(skel, q_pan, q_copy),
        pan_vs_copy_pos_cm=float(np.linalg.norm(p_pan - p_copy, axis=-1).mean()),
        feet=dict(coach=foot_metrics(skel, p_coach), pan=foot_metrics(skel, p_pan), copy=foot_metrics(skel, p_copy)),
    )
    if user_raw is not None:
        report['user_frames'] = int(user_raw.shape[1])
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    if not opt.no_viz:
        from pipeline.coach_user_viz import render
        render(out, coach_run, user_run)
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
