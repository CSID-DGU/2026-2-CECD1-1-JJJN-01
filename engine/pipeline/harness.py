"""Validation harness: synthetic "video" observations with known ground truth.

For each Mixamo test motion X (never in BigVegas' training npy):
  source  = BigVegas/X.bvh  -> ground-truth rotations, used ONLY to synthesise observations
  answer  = <target>/X.bvh  -> ground truth of the retargeted motion (same Mixamo animation)

Observation levels (what a pose estimator would hand to the pipeline):
  L0   exact joint positions, BigVegas body
  L0b  exact joint positions of a different body (bone lengths perturbed); the IK runs on
       that person's own skeleton estimated from the observations
  L1   L0b + per-frame noise (extra along the camera depth axis) + spine joints unobserved
  L2   MediaPipe-like input run through the real S1 code: the 14 joints MediaPipe provides
       (no pelvis/spine/neck/head/clavicles), hip-centred camera-aligned coordinates (no
       global trajectory), toe tip instead of ball of foot, image-plane jitter, temporally
       correlated depth error, far-side joints less visible/noisier. S1 then re-orients the
       motion to face +Z and rebuilds the root path from foot contact; evaluation aligns
       yaw + translation per clip (Procrustes) because that global frame is unknowable.

Conditions: gt (original rotations, upper bound) and each IK preset (free/init/limits/full);
'full_nocanon' re-uses the 'full' solution without the quaternion sign canonicalisation.

Metrics are reported per clip in results.json and aggregated in summary.md:
  fit_obs_cm / fit_true_cm  IK position error vs observations / vs clean truth
  rot_deg, twist_deg, swing_deg  local-rotation error vs ground truth
  ood_frac       fraction of normalised PAN input channels with |z| > 4
  pan_err        official-style error vs target ground truth: mean(dpos^2) / height^2 (x1e3)
  pan_gap        same formula vs the PAN output obtained from the original rotations
  pan_err_al     pan_err after per-clip yaw + translation alignment to the answer

Usage (from anywhere):
  python -m pipeline.harness --name first_run              # all test clips, all levels
  python -m pipeline.harness --max-clips 10 --levels L0 --presets free full
"""
import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.core.paths import MIXAMO_DIR, REPO, TRAIN_NPY_DIR, enter_repo  # noqa: E402

TARGETS = ('Mousey_m', 'Goblin_m', 'Mremireh_m', 'Vampire_m')
SPINE = ('Hips', 'Spine', 'Spine1', 'Spine2')
VIEWS = {'diagonal': (1., 0., 1.), 'side': (1., 0., 0.), 'front': (0., 0., 1.)}
GROUPS = {'root': ('Pelvis',),
          'legs': ('LeftUpLeg', 'LeftLeg', 'LeftFoot', 'RightUpLeg', 'RightLeg', 'RightFoot'),
          'spine': ('Spine', 'Spine1', 'Spine2', 'Neck'),
          'arms': ('LeftShoulder', 'LeftArm', 'LeftForeArm',
                   'RightShoulder', 'RightArm', 'RightForeArm')}


# ----------------------------------------------------------------------------- data
def training_first_frames(source):
    clips = np.load(TRAIN_NPY_DIR / f'{source}.npy', allow_pickle=True)
    return np.stack([np.asarray(c)[0] for c in clips]), [len(c) for c in clips]


def load_clips(source, targets, max_clips=None, files=None, min_frames=16):
    from data_preprocess.Mixamo.bvh_parser import BVH_file
    names = sorted(p.name for p in (MIXAMO_DIR / source).glob('*.bvh'))
    if files:
        missing = set(files) - set(names)
        if missing:
            raise FileNotFoundError(f'Not in {source}: {sorted(missing)}')
        names = [n for n in names if n in files]
    first, lengths = training_first_frames(source)
    clips, skipped = [], {}
    for name in names:
        if max_clips and len(clips) >= max_clips:
            break
        f = BVH_file(str(MIXAMO_DIR / source / name))
        euler = f.to_tensor().numpy().T                       # 60 fps, as stored in the npy
        d = np.abs(first - euler[0]).max(1)
        if d.min() < 1e-3 and lengths[int(d.argmin())] == len(euler):
            skipped[name] = 'in training npy'
            continue
        raws = {source: f.to_tensor(quater=True)[:, ::2]}
        for t in targets:
            p = MIXAMO_DIR / t / name
            if not p.exists():
                skipped[name] = f'missing for {t}'
                break
            raws[t] = BVH_file(str(p)).to_tensor(quater=True)[:, ::2]
        else:
            T = min(r.shape[1] for r in raws.values()) // 4 * 4
            if T < min_frames:
                skipped[name] = f'too short ({T})'
                continue
            clips.append(dict(name=name, src=raws[source][:, :T].double(),
                              gt={t: raws[t][:, :T] for t in targets}))
    return clips, skipped


# ------------------------------------------------------------------------- alignment
def yaw_align(est, ref):
    """Yaw rotation R (3x3) and translation t minimising |R est + t - ref|^2 (all frames)."""
    a = est.reshape(-1, 3)
    b = ref.reshape(-1, 3)
    ac, bc = a - a.mean(0), b - b.mean(0)
    C = (ac[:, 0] * bc[:, 0] + ac[:, 2] * bc[:, 2]).sum()
    S = (ac[:, 2] * bc[:, 0] - ac[:, 0] * bc[:, 2]).sum()
    th = math.atan2(S, C)
    R = np.array([[math.cos(th), 0., math.sin(th)], [0., 1., 0.], [-math.sin(th), 0., math.cos(th)]])
    return R, b.mean(0) - a.mean(0) @ R.T, th


# ----------------------------------------------------------------------- observations
def perturbed_offsets(skel, scales):
    off = skel.offsets.copy()
    for j, n in enumerate(skel.names):
        for key, s in scales.items():
            if key == 'legs' and any(k in n for k in ('UpLeg', 'Leg', 'Foot', 'Toe')) \
                    or key == 'arms' and any(k in n for k in ('Arm', 'Hand')) \
                    or key == 'spine' and any(k in n for k in ('Spine', 'Neck', 'Head')):
                off[j] *= s
    return off


def correlated_noise(rng, shape, sigma, frames):
    """Gaussian noise low-passed along axis 0 and rescaled to std `sigma`."""
    from scipy.ndimage import gaussian_filter1d
    x = rng.normal(0., 1., shape)
    if frames > 0:
        x = gaussian_filter1d(x, frames, axis=0, mode='nearest')
        x /= max(x.std(), 1e-9)
    return x * sigma


def synth_mediapipe(truth, skel, opt, rng):
    """Clean world positions of one clip (T, J, 3) -> MediaPipe-format world landmarks + visibility."""
    from pipeline.core.canonicalize import facing_yaw
    from pipeline.core.mediapipe_format import LANDMARKS, TO_MIXAMO
    T = len(truth)
    theta = facing_yaw(truth, np.ones(truth.shape[:2]), skel)
    view = {'front': 0., 'diagonal': 45., 'side': 90.}[opt.view]
    a = theta + math.radians(view)
    to_cam = np.array([math.sin(a), 0., math.cos(a)])          # person -> camera
    z = -to_cam
    y = np.array([0., -1., 0.])
    x = np.cross(y, z)
    cam = np.stack([x, y, z])                                    # rows: camera axes in world
    hips = truth[:, [skel.index('LeftUpLeg'), skel.index('RightUpLeg')]].mean(1)
    world = np.full((T, 33, 3), np.nan)
    vis = np.zeros((T, 33))
    for joint, (lm, _) in TO_MIXAMO.items():
        p = truth[:, skel.index(joint)]
        if lm.endswith('foot_index'):                            # toe tip, not ball of foot
            foot = truth[:, skel.index(joint.replace('ToeBase', 'Foot'))]
            p = p + 0.35 * (p - foot)
        c = (p - hips) @ cam.T                                   # camera-aligned, cm
        far = c[:, 2] > 5.
        depth_sigma = opt.l2_depth_noise_cm * np.where(far, 1.5, 1.)
        c[:, :2] += rng.normal(0., opt.l2_jitter_cm, (T, 2))
        c[:, 2] += correlated_noise(rng, T, 1., opt.l2_depth_corr_frames) * depth_sigma
        c[:, 2] += rng.normal(0., opt.l2_jitter_cm, T)
        k = LANDMARKS[lm]
        world[:, k] = c / 100. * np.array([1., 1., 1.])          # already MediaPipe axes
        vis[:, k] = np.where(far, 0.6, 0.98)
    return world, vis


def observe_l2(clips, truth, bounds, skel, priors, opt, rng):
    """Synthetic MediaPipe -> real S1 conversion. Returns obs, weight, person skeleton."""
    from pipeline.core.canonicalize import anchor_root, face_forward, fill_gaps
    from pipeline.core.mediapipe_format import UNRELIABLE_LENGTH, landmarks_to_joints
    rel, weights = [], []
    for k in range(len(clips)):
        world, vis = synth_mediapipe(truth[bounds[k]:bounds[k + 1]], skel, opt, rng)
        o, w = landmarks_to_joints(world, vis, skel)
        o, w = fill_gaps(o, w)
        o, _ = face_forward(o, w, skel)
        rel.append(o)
        weights.append(w)
    weight = np.concatenate(weights)
    person = estimate_skeleton(skel, np.concatenate(rel), weight > 0, UNRELIABLE_LENGTH)
    ground = priors['ground']['lowest_foot_y_median'] * person.height() / priors['ground']['height']
    obs = np.concatenate([anchor_root(o, w, skel, ground) for o, w in zip(rel, weights)])
    return obs, weight, person


def observe(level, clips, skel, opt, rng, priors=None, bounds=None):
    """-> obs (T, J, 3), weight (T, J), clean truth (T, J, 3), generating offsets, person or None."""
    from pipeline.core import quat as Q
    if level == 'L0':
        offsets = skel.offsets
    elif level in ('L0b', 'L1', 'L2'):
        offsets = perturbed_offsets(skel, opt.body_scale)
    else:
        raise ValueError(level)
    ratio = skel.height(offsets) / skel.height()
    truth = []
    for c in clips:
        q, root = skel.raw_to_joint(c['src'])
        truth.append(skel.fk(Q.normalize(q), root * ratio, offsets))
    truth = torch.cat(truth).numpy()
    if level == 'L2':
        obs, weight, person = observe_l2(clips, truth, bounds, skel, priors, opt, rng)
        return obs, weight, truth, offsets, person
    obs = truth.copy()
    mask = np.ones(truth.shape[:2], dtype=bool)
    if level == 'L1':
        depth = np.asarray(VIEWS[opt.view]) / np.linalg.norm(VIEWS[opt.view])
        obs += rng.normal(0., opt.noise_cm, obs.shape)
        obs += rng.normal(0., opt.depth_noise_cm, obs.shape[:2])[..., None] * depth
        for n in SPINE:
            mask[:, skel.index(n)] = False
    return obs, mask.astype(np.float64), truth, offsets, None


from pipeline.core.person import estimate_skeleton  # noqa: E402  (re-exported)


# --------------------------------------------------------------------------- metrics
def rotation_errors(skel, q_est, q_gt, fixed):
    from pipeline.core import quat as Q
    rows = {}
    for j in skel.rotating_joints:
        name = skel.names[j]
        if name in fixed:
            continue
        ang = Q.angle_between(q_est[:, j], q_gt[:, j])
        rec = dict(total=float(torch.rad2deg(ang).mean()))
        axis = skel.bone_axis(j)
        if axis is not None:
            d = Q.mul(Q.conj(Q.normalize(q_gt[:, j])), Q.normalize(q_est[:, j]))
            tw, sw = Q.twist_swing_angles(d, torch.as_tensor(axis, dtype=d.dtype))
            rec.update(twist=float(torch.rad2deg(tw).mean()), swing=float(torch.rad2deg(sw).mean()))
        rows[name] = rec
    return rows


def pan_metrics(runtime, raw, clip, ref_pos, heights):
    from pipeline.core.pan_runtime import ood_stats
    out = runtime.retarget(raw)
    res = dict(ood=ood_stats(runtime.normalize_source(raw.float())), pan_err={}, pan_gap={}, pan_err_al={})
    pos = {}
    for i, t in enumerate(runtime.targets):
        pos[t] = runtime.positions(out[t], 1, i)
        gt = runtime.positions(clip['gt'][t], 1, i)
        h2 = heights[t] ** 2
        res['pan_err'][t] = float(((pos[t] - gt) ** 2).mean() / h2 * 1e3)
        P, G = pos[t].double().cpu().numpy(), gt.double().cpu().numpy()
        R, tr, _ = yaw_align(P, G)
        res['pan_err_al'][t] = float(((P @ R.T + tr - G) ** 2).mean() / h2 * 1e3)
        if ref_pos is not None:
            res['pan_gap'][t] = float(((pos[t] - ref_pos[t]) ** 2).mean() / h2 * 1e3)
    return res, pos, out


def clip_report(rows, clips_of_interest):
    lines = []
    for name in clips_of_interest:
        sel = [r for r in rows if r['clip'] == name]
        if not sel:
            continue
        lines += [f'\n### {name}\n', '| level | condition | rot_deg | pan_err×1e3 | pan_err_al×1e3 |', '|---|---|---|---|---|']
        for r in sel:
            rot = np.mean([v['total'] for v in r['rot'].values()]) if r['rot'] else float('nan')
            lines.append(f"| {r['level']} | {r['condition']} | {rot:.1f} | "
                         f"{np.mean(list(r['pan_err'].values())):.3f} | {np.mean(list(r['pan_err_al'].values())):.3f} |")
    return '\n'.join(lines)


def summarize(rows, levels, conditions):
    lines = ['| level | condition | fit_obs_cm | fit_true_cm | rot_deg | twist_deg | swing_deg | '
             + ' | '.join(f'rot_{g}' for g in GROUPS) + ' | ood_frac | pan_err×1e3 | pan_gap×1e3 | pan_err_al×1e3 |',
             '|' + '---|' * (12 + len(GROUPS))]
    table = {}
    for lv in levels:
        for cond in conditions:
            sel = [r for r in rows if r['level'] == lv and r['condition'] == cond]
            if not sel:
                continue
            def m(f):
                v = [f(r) for r in sel]
                v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
                return float(np.mean(v)) if v else float('nan')
            def joint_mean(r, key, names=None):
                vals = [v[key] for n, v in r['rot'].items() if key in v and (names is None or n in names)]
                return float(np.mean(vals)) if vals else None
            s = dict(fit_obs_cm=m(lambda r: r.get('fit_obs_cm')),
                     fit_true_cm=m(lambda r: r.get('fit_true_cm')),
                     rot_deg=m(lambda r: joint_mean(r, 'total')),
                     twist_deg=m(lambda r: joint_mean(r, 'twist')),
                     swing_deg=m(lambda r: joint_mean(r, 'swing')),
                     **{f'rot_{g}': m(lambda r, n=n: joint_mean(r, 'total', n)) for g, n in GROUPS.items()},
                     ood_frac=m(lambda r: r['ood']['frac_abs_z_gt']),
                     pan_err=m(lambda r: np.mean(list(r['pan_err'].values()))),
                     pan_gap=m(lambda r: np.mean(list(r['pan_gap'].values())) if r['pan_gap'] else None),
                     pan_err_al=m(lambda r: np.mean(list(r['pan_err_al'].values()))))
            table[f'{lv}/{cond}'] = s
            fmt = lambda x: '–' if x is None or math.isnan(x) else f'{x:.3f}' if abs(x) < 10 else f'{x:.1f}'
            lines.append(f'| {lv} | {cond} | ' + ' | '.join(fmt(v) for v in s.values()) + ' |')
    return '\n'.join(lines), table


# ------------------------------------------------------------------------------ main
def parse_scales(text):
    out = {}
    for item in text.split(','):
        k, v = item.split('=')
        if k not in ('legs', 'arms', 'spine'):
            raise argparse.ArgumentTypeError(f'unknown body part {k}')
        out[k] = float(v)
    return out


def main(argv=None):
    from pipeline.core.ik import PRESETS, solve
    from pipeline.core.filters import smooth_positions
    from pipeline.core import quat as Q
    from pipeline.core.pan_runtime import PanRuntime, canonical_sign
    from pipeline.core.priors import load_priors
    from pipeline.core.skeleton import Skeleton
    from dataclasses import replace

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', default=None, help='run directory name (default: timestamp)')
    ap.add_argument('--source', default='BigVegas')
    ap.add_argument('--targets', nargs='+', default=list(TARGETS))
    ap.add_argument('--levels', nargs='+', default=['L0', 'L0b', 'L1'], choices=['L0', 'L0b', 'L1', 'L2'])
    ap.add_argument('--presets', nargs='+', default=list(PRESETS), choices=list(PRESETS))
    ap.add_argument('--max-clips', type=int, default=None)
    ap.add_argument('--files', nargs='+', default=None, help='test BVH file names to use')
    ap.add_argument('--steps', type=int, default=None, help='override IK steps')
    ap.add_argument('--w-limit', type=float, default=None)
    ap.add_argument('--w-prior', type=float, default=None)
    ap.add_argument('--body-scale', type=parse_scales, default=parse_scales('legs=1.08,arms=0.94,spine=1.05'))
    ap.add_argument('--noise-cm', type=float, default=1.0)
    ap.add_argument('--depth-noise-cm', type=float, default=3.0)
    ap.add_argument('--view', default='diagonal', choices=list(VIEWS))
    ap.add_argument('--l2-jitter-cm', type=float, default=1.0)
    ap.add_argument('--l2-depth-noise-cm', type=float, default=4.0)
    ap.add_argument('--l2-depth-corr-frames', type=float, default=8.0,
                    help='temporal correlation (Gaussian sigma, frames) of the L2 depth error')
    ap.add_argument('--report-clips', nargs='*', default=['Standing To Crouch.bvh'])
    ap.add_argument('--obs-sigma', type=float, default=0.,
                    help='Gaussian temporal filter on observations (frames, 0 = off)')
    ap.add_argument('--save-bvh', nargs='*', default=['Standing To Crouch.bvh'],
                    help='clips whose BVHs are written for inspection')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    opt = ap.parse_args(argv)

    enter_repo()
    if opt.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.manual_seed(opt.seed)
    rng = np.random.default_rng(opt.seed)
    out_dir = REPO / 'runs/harness' / (opt.name or datetime.now().strftime('%Y%m%d_%H%M%S'))
    if out_dir.exists():
        raise FileExistsError(f'{out_dir} exists; pick another --name')
    out_dir.mkdir(parents=True)

    skel = Skeleton.from_character(opt.source)
    priors = load_priors(opt.source)
    runtime = PanRuntime(source=opt.source, targets=opt.targets, device=opt.device)
    pan_stats = tuple(s.cpu() for s in runtime.source_stats)
    heights = {t: Skeleton.from_character(t).height() for t in opt.targets}
    clips, skipped = load_clips(opt.source, opt.targets, opt.max_clips, opt.files)
    if not clips:
        raise RuntimeError('No usable clips')
    lengths = [c['src'].shape[1] for c in clips]
    segments = np.repeat(np.arange(len(clips)), lengths)
    bounds = np.cumsum([0] + lengths)
    print(f'{len(clips)} clips, {sum(lengths)} frames; skipped: {skipped}', flush=True)

    presets = {}
    for k in opt.presets:
        cfg = PRESETS[k]
        upd = {f: v for f, v in (('steps', opt.steps), ('w_limit', opt.w_limit if cfg.w_limit else None),
                                 ('w_prior', opt.w_prior if cfg.w_prior else None)) if v is not None}
        presets[k] = replace(cfg, **upd)
    config = dict(vars(opt), clips=[c['name'] for c in clips], skipped=skipped, priors=priors['source'],
                  started=datetime.now().isoformat(timespec='seconds'))
    config['presets'] = {k: dict(v.__dict__) for k, v in presets.items()}
    (out_dir / 'config.json').write_text(json.dumps(config, indent=1, default=str))

    save = set(opt.save_bvh or [])
    rows, ref_pos = [], {}
    bv_height = skel.height()
    started = time.monotonic()

    # Upper bound: original rotations.
    for c in clips:
        res, pos, out = pan_metrics(runtime, c['src'], c, None, heights)
        ref_pos[c['name']] = pos
        rows.append(dict(level='-', condition='gt', clip=c['name'], rot={}, **res))
        if c['name'] in save:
            d = out_dir / 'bvh' / Path(c['name']).stem
            d.mkdir(parents=True, exist_ok=True)
            runtime.writer(0, 0).write_raw(c['src'].float(), 'quaternion', str(d / 'source_gt.bvh'))
            for i, t in enumerate(opt.targets):
                runtime.writer(1, i).write_raw(c['gt'][t].float(), 'quaternion', str(d / f'{t}_answer.bvh'))
                runtime.writer(1, i).write_raw(out[t], 'quaternion', str(d / f'{t}_gt.bvh'))

    for level in opt.levels:
        obs, weight, truth, gen_offsets, person = observe(level, clips, skel, opt, rng, priors, bounds)
        mask = weight > 0
        obs = smooth_positions(obs, segments, opt.obs_sigma)
        if person is None:
            person = estimate_skeleton(skel, obs, mask)
        ratio = bv_height / person.height()
        print(f'\n[{level}] person height {person.height():.1f} cm (generated '
              f'{skel.height(gen_offsets):.1f}, BigVegas {bv_height:.1f})', flush=True)
        for name, cfg in presets.items():
            if level == 'L2' and not cfg.free_root:     # no observed root: optimise it
                cfg = replace(cfg, free_root=True, w_root_acc=PRESETS['full_root'].w_root_acc)
            print(f'[{level}/{name}] solving IK on {len(obs)} frames (free_root={cfg.free_root})', flush=True)
            sol = solve(obs, weight, person, priors, pan_stats, segments, cfg, device=opt.device)
            print(f'  done in {sol["seconds"]:.0f}s, final {sol["final"]}', flush=True)
            variants = [(name, True)] + ([('full_nocanon', False)] if name == 'full' else [])
            for k, c in enumerate(clips):
                a, b = bounds[k], bounds[k + 1]
                q, root = sol['q'][a:b].double(), sol['root'][a:b].double()
                pos = sol['positions'][a:b].numpy()
                m = mask[a:b]
                fit_obs = float(np.linalg.norm(pos - obs[a:b], axis=-1)[m].mean())
                q_gt, _ = skel.raw_to_joint(c['src'])
                if level == 'L2':          # unknown global frame: align yaw + translation first
                    R, tr, th = yaw_align(pos, truth[a:b])
                    fit_true = float(np.linalg.norm(pos @ R.T + tr - truth[a:b], axis=-1).mean())
                    qy = torch.tensor([math.cos(th / 2), 0., math.sin(th / 2), 0.], dtype=q.dtype)
                    q_eval = q.clone()
                    q_eval[:, 0] = Q.mul(qy.expand_as(q[:, 0]), q[:, 0])
                else:
                    fit_true = float(np.linalg.norm(pos - truth[a:b], axis=-1).mean())
                    q_eval = q
                rot = rotation_errors(skel, q_eval, q_gt, [])   # same joint set for every condition
                raw0 = skel.joint_to_raw(Q.normalize(q), root * ratio)
                for cond, canon in variants:
                    raw = canonical_sign(raw0, *pan_stats) if canon else raw0
                    res, _, out = pan_metrics(runtime, raw, c, ref_pos[c['name']], heights)
                    rows.append(dict(level=level, condition=cond, clip=c['name'], fit_obs_cm=fit_obs,
                                     fit_true_cm=fit_true, rot=rot, **res))
                    if c['name'] in save:
                        d = out_dir / 'bvh' / Path(c['name']).stem
                        runtime.writer(0, 0).write_raw(raw.float(), 'quaternion', str(d / f'source_{level}_{cond}.bvh'))
                        for i, t in enumerate(opt.targets):
                            runtime.writer(1, i).write_raw(out[t], 'quaternion', str(d / f'{t}_{level}_{cond}.bvh'))
            (out_dir / 'results.json').write_text(json.dumps(rows, indent=1))

    conditions = ['gt'] + list(presets) + (['full_nocanon'] if 'full' in presets else [])
    md, table = summarize(rows, ['-'] + opt.levels, conditions)
    md += '\n' + clip_report(rows, opt.report_clips or [])
    header = (f'# Harness `{out_dir.name}`\n\n{len(clips)} test clips ({sum(lengths)} frames @30fps), '
              f'targets {", ".join(opt.targets)}, {time.monotonic() - started:.0f}s.\n'
              f'Levels: L0 exact/BigVegas body; L0b exact/perturbed body {opt.body_scale}; '
              f'L1 = L0b + noise {opt.noise_cm}cm + depth {opt.depth_noise_cm}cm ({opt.view}), spine unobserved. '
              f'Observation filter sigma {opt.obs_sigma} frames. L2: MediaPipe-like, view {opt.view}, '
              f'jitter {opt.l2_jitter_cm}cm, depth {opt.l2_depth_noise_cm}cm correlated over '
              f'{opt.l2_depth_corr_frames} frames.\n\n')
    (out_dir / 'results.json').write_text(json.dumps(rows, indent=1))
    (out_dir / 'summary.json').write_text(json.dumps(table, indent=1))
    (out_dir / 'summary.md').write_text(header + md + '\n')
    print('\n' + md)
    print(f'\nWrote {out_dir}')


if __name__ == '__main__':
    main()
