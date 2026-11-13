"""Can 2D keypoints undo WHAM's regression of bone lengths toward the mean body?

Keypoint sources (same frames as WHAM):
  vitpose    ViTPose-H COCO-17, WHAM's own tracking_results.pth
  mediapipe  MediaPipe Pose (heavy) image landmarks, mapped to COCO-17 (+ mouth for the face)
  both       per bone, the frame ratios of both sources pooled

For every frame and bone, compare the bone's length in the image with the length of the same
bone of WHAM's SMPL mesh projected with WHAM's camera (verts_cam, focal = image diagonal;
joints from WHAM's COCO regressor):

    r_t = len_2d / len_wham_projected

Both share the same view, so foreshortening cancels to first order (WHAM's bone directions
are good, its lengths are pulled to the mean). median_t r = how much longer/shorter the real
bone is than WHAM's; dividing by the median over all bones removes the global scale, which a
single camera cannot observe. Frames are used only where both keypoints are confident and the
WHAM bone is within `max_tilt` degrees of the image plane.

Head metrics (pure 2D ratios within a frame, frames with the torso near the image plane):
  head_neck  nose to shoulder midpoint / torso (shoulder-hip midpoints)   both sources
  face       eye midpoint to mouth midpoint / torso                       mediapipe only
Children have a larger head relative to the torso; this is the cue that adult-only SMPL loses.

Checks:
  stability   spread of r over frames (IQR), left/right agreement
  consistency same coach, two cameras (squat_diag vs squat_side): values should agree
  effect      do other people differ from the coach by more than the consistency noise?

Run with the WHAM venv (joblib/cv2), from engine/:
  ../.venv-wham/bin/python pipeline/proportions_check.py \
      coach_diag=runs/video_wham/squat_diag_allclav coach_side=runs/video_wham/squat_side_allclav \
      child=runs/video_wham/child_squat [--mediapipe-dir runs/proportions/mediapipe] [--out DIR]
Each argument is label=<dir with wham_work/ and report.json (video path)>; MediaPipe landmarks
are read from <mediapipe-dir>/<label>.npz (tools/pose_estimation/mediapipe_extract.py on the
same video file) when present.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

# $WHAM_DIR or <repo>/third_party/WHAM (same rule as pipeline/core/paths.py; kept standalone for the WHAM venv)
WHAM = Path(os.environ.get('WHAM_DIR', Path(__file__).resolve().parents[2] / 'third_party/WHAM'))
COCO = ['nose', 'leye', 'reye', 'lear', 'rear', 'lsho', 'rsho', 'lelb', 'relb', 'lwri', 'rwri',
        'lhip', 'rhip', 'lknee', 'rknee', 'lank', 'rank']
MP_FROM_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]   # MediaPipe 33 -> COCO 17
MP_MOUTH = (9, 10)
BONES = {  # name: (a, b)
    'thigh_L': ('lhip', 'lknee'), 'thigh_R': ('rhip', 'rknee'),
    'shin_L': ('lknee', 'lank'), 'shin_R': ('rknee', 'rank'),
    'torso_L': ('lsho', 'lhip'), 'torso_R': ('rsho', 'rhip'),
    'upperarm_L': ('lsho', 'lelb'), 'upperarm_R': ('rsho', 'relb'),
    'forearm_L': ('lelb', 'lwri'), 'forearm_R': ('relb', 'rwri'),
    'shoulder_w': ('lsho', 'rsho'), 'hip_w': ('lhip', 'rhip'),
}
GROUPS = ['thigh', 'shin', 'torso', 'upperarm', 'forearm', 'shoulder_w', 'hip_w']
LIMBS = ['thigh', 'shin', 'torso', 'upperarm', 'forearm']
SOURCES = ['vitpose', 'mediapipe', 'both']


def load(run, mp_dir=None, label=None):
    import cv2
    import joblib
    run = Path(run)
    work = run / 'wham_work'
    out, trk = joblib.load(work / 'wham_output.pkl'), joblib.load(work / 'tracking_results.pth')
    sid = max(out, key=lambda k: len(out[k]['frame_ids']))
    o, t = out[sid], trk[sid]
    assert np.array_equal(o['frame_ids'], t['frame_id'])
    video = json.loads((run / 'report.json').read_text())['video']
    cap = cv2.VideoCapture(video)
    w, h = cap.get(3), cap.get(4)
    f = (w * w + h * h) ** .5
    reg = np.load(WHAM / 'dataset/body_models/J_regressor_coco.npy')
    X = np.einsum('jv,tvc->tjc', reg, o['verts'])                 # camera frame, metres
    uv = f * X[..., :2] / X[..., 2:] + np.array([w / 2, h / 2])
    d = dict(X=X, uv=uv, kp={'vitpose': t['keypoints']}, video=video)
    mp_file = Path(mp_dir) / f'{label}.npz' if mp_dir else None
    if mp_file and mp_file.exists():
        m = np.load(mp_file)
        if Path(str(m['source'])).resolve() != Path(video).resolve():
            raise ValueError(f'{mp_file} was made from {m["source"]}, WHAM from {video}')
        ids = o['frame_ids']
        img = m['image'][ids] * np.array([m['width'], m['height'], 0.])
        vis = np.nan_to_num(m['visibility'][ids])
        kp = np.concatenate([img[:, MP_FROM_COCO, :2], vis[:, MP_FROM_COCO, None]], -1)
        d['kp']['mediapipe'] = kp
        d['mouth'] = np.concatenate([img[:, MP_MOUTH, :2], vis[:, MP_MOUTH, None]], -1)
    return d


def tilt_deg(X, i, k):
    bone = X[:, k] - X[:, i]
    ray = (X[:, k] + X[:, i]) / 2
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
    return np.degrees(np.arcsin(np.clip(np.abs((bone * ray).sum(-1)) / np.linalg.norm(bone, axis=-1), 0, 1)))


def bone_ratios(d, src, conf, max_tilt):
    """name -> per-frame ratios r (usable frames only), WHAM 3D length, median tilt."""
    X, uv, kp = d['X'], d['uv'], d['kp'][src]
    res = {}
    for name, (a, b) in BONES.items():
        i, k = COCO.index(a), COCO.index(b)
        tilt = tilt_deg(X, i, k)
        l2d = np.linalg.norm(kp[:, k, :2] - kp[:, i, :2], axis=-1)
        lpr = np.linalg.norm(uv[:, k] - uv[:, i], axis=-1)
        ok = (kp[:, i, 2] > conf) & (kp[:, k, 2] > conf) & (tilt < max_tilt) & (lpr > 5)
        res[name] = dict(r=l2d[ok] / lpr[ok], len3d=float(np.median(np.linalg.norm(X[:, k] - X[:, i], axis=-1))),
                         tilt=float(np.median(tilt)))
    return res


def summarize(raw):
    out = {}
    for n, v in raw.items():
        r = v['r']
        out[n] = dict(n=int(len(r)), r=float(np.median(r)) if len(r) else np.nan,
                      iqr=float(np.subtract(*np.percentile(r, [75, 25]))) if len(r) > 3 else np.nan,
                      len3d=v['len3d'], tilt=v['tilt'])
    return out


def measure(d, conf=0.6, max_tilt=45.):
    srcs = {s: bone_ratios(d, s, conf, max_tilt) for s in d['kp']}
    res = {s: summarize(v) for s, v in srcs.items()}
    if len(srcs) == 2:
        res['both'] = summarize({n: dict(srcs['vitpose'][n], r=np.concatenate([srcs[s][n]['r'] for s in srcs]))
                                 for n in BONES})
    return res


def group(res, key):
    out = {}
    for g in GROUPS:
        v = [res[n][key] for n in res if (n == g or n.rsplit('_', 1)[0] == g) and np.isfinite(res[n][key])]
        out[g] = float(np.mean(v)) if v else np.nan
    return out


def factors(res):
    c = group(res, 'r')
    s = np.nanmedian([c[g] for g in GROUPS])
    return {g: c[g] / s for g in GROUPS}


def proportions(res, corrected):
    """Bone lengths per group as a fraction of torso + thigh + shin (a stature proxy)."""
    L = group(res, 'len3d')
    if corrected:
        c = factors(res)
        L = {g: L[g] * c[g] for g in GROUPS}
    H = L['torso'] + L['thigh'] + L['shin']
    return {g: L[g] / H for g in GROUPS}


def head_metrics(d, conf=0.5, max_tilt=30.):
    """Median 2D head ratios over frames with the torso within max_tilt of the image plane."""
    X = d['X']
    upright = (tilt_deg(X, COCO.index('lsho'), COCO.index('lhip')) < max_tilt) & \
              (tilt_deg(X, COCO.index('rsho'), COCO.index('rhip')) < max_tilt)
    out = {}
    for src, kp in d['kp'].items():
        P, c = kp[..., :2], kp[..., 2]
        sho = (P[:, 5] + P[:, 6]) / 2
        hip = (P[:, 11] + P[:, 12]) / 2
        torso = np.linalg.norm(sho - hip, axis=-1)
        base = upright & (c[:, [5, 6, 11, 12]].min(1) > conf) & (torso > 5)
        ok = base & (c[:, 0] > conf)
        hn = np.linalg.norm(P[:, 0] - sho, axis=-1)[ok] / torso[ok]
        out[f'head_neck/{src}'] = (float(np.median(hn)) if ok.any() else np.nan, int(ok.sum()),
                                   float(np.subtract(*np.percentile(hn, [75, 25]))) if ok.sum() > 3 else np.nan)
        if src == 'mediapipe':
            M = d['mouth']
            ok = base & (c[:, [1, 2]].min(1) > conf) & (M[..., 2].min(1) > conf)
            fl = np.linalg.norm((P[:, 1] + P[:, 2]) / 2 - M[..., :2].mean(1), axis=-1)[ok] / torso[ok]
            out['face/mediapipe'] = (float(np.median(fl)) if ok.any() else np.nan, int(ok.sum()),
                                     float(np.subtract(*np.percentile(fl, [75, 25]))) if ok.sum() > 3 else np.nan)
    return out


def table(title, runs, rows, cell):
    lines = [title, '', '| | ' + ' | '.join(runs) + ' |', '|---|' + '---|' * len(runs)]
    lines += [f'| {r} | ' + ' | '.join(cell(lab, r) for lab in runs) + ' |' for r in rows]
    return lines + ['']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', help='label=run_dir')
    ap.add_argument('--mediapipe-dir', default='runs/proportions/mediapipe')
    ap.add_argument('--out', default='runs/proportions/check')
    ap.add_argument('--conf', type=float, default=0.6)
    ap.add_argument('--max-tilt', type=float, default=45.)
    ap.add_argument('--ref', nargs='+', default=['coach_diag', 'coach_side'],
                    help='labels of the same person seen twice (consistency) / reference body')
    opt = ap.parse_args()
    runs = dict(a.split('=', 1) for a in opt.runs)
    data = {lab: load(run, opt.mediapipe_dir, lab) for lab, run in runs.items()}
    meas = {lab: measure(d, opt.conf, opt.max_tilt) for lab, d in data.items()}
    heads = {lab: head_metrics(d) for lab, d in data.items()}
    labs = list(runs)
    srcs = [s for s in SOURCES if all(s in m for m in meas.values())]

    lines = ['# 2D-keypoint bone-length correction of WHAM\n',
             f'conf > {opt.conf}, bone tilt < {opt.max_tilt}° (WHAM). r = 2D length / WHAM projected length.\n']
    for s in srcs:
        lines += table(f'## r per bone, {s}: median ±IQR (frames)', labs, list(BONES),
                       lambda lab, b: f'{meas[lab][s][b]["r"]:.3f} ±{meas[lab][s][b]["iqr"]:.3f} ({meas[lab][s][b]["n"]})')
    lr = {s: {lab: float(np.nanmean([abs(meas[lab][s][f'{g}_L']['r'] - meas[lab][s][f'{g}_R']['r'])
                                     for g in LIMBS])) for lab in labs} for s in srcs}
    lines += table('## Left/right disagreement (mean |r_L - r_R| over limb groups; should be 0)', labs, srcs,
                   lambda lab, s: f'{lr[s][lab]:.3f}')
    props = {s: {lab: proportions(meas[lab][s], True) for lab in labs} for s in srcs}
    props['wham'] = {lab: proportions(meas[lab][srcs[0]], False) for lab in labs}
    lines += table('## Proportions (fraction of torso+thigh+shin)', labs,
                   [f'{s}:{g}' for s in ['wham'] + srcs for g in LIMBS],
                   lambda lab, sg: f'{props[sg.split(":")[0]][lab][sg.split(":")[1]]:.3f}')

    a, b = opt.ref[:2]
    cons = {s: float(np.mean([abs(props[s][a][g] - props[s][b][g]) / props[s][a][g] for g in LIMBS]))
            for s in ['wham'] + srcs}
    others = [lab for lab in labs if lab not in opt.ref]
    ref = {s: {g: (props[s][a][g] + props[s][b][g]) / 2 for g in LIMBS} for s in ['wham'] + srcs}
    diff = {s: {lab: float(np.mean([abs(props[s][lab][g] - ref[s][g]) / ref[s][g] for g in LIMBS]))
                for lab in others} for s in ['wham'] + srcs}
    lines += ['## Signal vs noise (limb proportions, mean relative difference)', '',
              f'| source | noise: {a} vs {b} | ' + ' | '.join(f'{o} vs coach' for o in others) + ' |',
              '|---|---|' + '---|' * len(others)]
    lines += [f'| {s} | {cons[s]:.1%} | ' + ' | '.join(f'{diff[s][o]:.1%}' for o in others) + ' |'
              for s in ['wham'] + srcs] + ['']
    hk = sorted({k for h in heads.values() for k in h})
    lines += table('## Head metrics (2D ratio to torso length): median ±IQR (frames)', labs, hk,
                   lambda lab, k: (f'{heads[lab][k][0]:.3f} ±{heads[lab][k][2]:.3f} ({heads[lab][k][1]})'
                                   if k in heads[lab] else '-'))

    out = Path(opt.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(dict(runs=runs, bones=meas, proportions=props, head=heads,
                                                      consistency=cons, difference=diff), indent=1, default=float))
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
