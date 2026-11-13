"""End-to-end: video -> pose estimator -> S1 -> constrained IK (person's own skeleton) -> PAN.

  python -m pipeline.run_video VIDEO --targets Mousey_m Abe_m [--name squat01]
  python -m pipeline.run_video VIDEO --estimator wham          # -> runs/video_wham/<name>/

Estimators: mediapipe (default; 14 joints, hip-centred, root path from foot contact) or
wham (SMPL 24 joints incl. pelvis/spine/neck/head/clavicles, constant body shape, world
trajectory; only re-grounded and re-oriented here).

Steps / outputs in runs/video[_wham]/<name>/:
  landmarks.npz / wham.npz  S0 output (MediaPipe in .venv-pose, WHAM in .venv-wham)
  observation.npz       S1, Mixamo-frame joints (cm, Y-up, 30 fps, faces +Z, root path);
                        also `landmarks` (T, 33, 3): all MediaPipe points with the same
                        transforms but no smoothing/gap filling, and `landmark_visibility`
  source_person.bvh     IK result on the person's own skeleton (sanity check vs. the video)
  source_pan_input.bvh  the same motion on BigVegas as fed to PAN
  <target>.bvh          retargeted motion for each target character
  report.json           detection rate, person height, IK terms, PAN input OOD statistics
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.core.paths import POSE_PYTHON, REPO, WHAM_PYTHON, enter_repo  # noqa: E402

EXTRACT = REPO / 'tools/pose_estimation/mediapipe_extract.py'
WHAM_EXTRACT = REPO / 'tools/pose_estimation/wham_extract.py'
DEFAULT_TARGETS = ('Abe_m', 'Goblin_m', 'Mousey_m', 'Mremireh_m', 'Vampire_m')


def run_s0(video, out):
    if not POSE_PYTHON.exists():
        raise FileNotFoundError(f'{POSE_PYTHON} (pose venv) missing')
    subprocess.run([str(POSE_PYTHON), '-I', str(EXTRACT), str(video), str(out)], check=True)


def run_s1(lm, skel, priors, obs_sigma=1.5):
    from pipeline.core.canonicalize import anchor_root, face_forward, fill_gaps, resample, yaw_matrix
    from pipeline.core.filters import smooth_positions
    from pipeline.core.mediapipe_format import UNRELIABLE_LENGTH, landmarks_to_joints, mp_to_yup_cm
    from pipeline.core.person import estimate_skeleton
    det = lm['detected']
    if det.mean() < 0.5:
        raise RuntimeError(f'Person detected in only {det.mean():.0%} of frames')
    first, last = np.flatnonzero(det)[[0, -1]]                 # trim undetected ends
    world, vis = lm['world'][first:last + 1], lm['visibility'][first:last + 1]
    obs, weight = landmarks_to_joints(np.nan_to_num(world), vis, skel)
    obs, weight = fill_gaps(obs, weight)
    fps = float(lm['fps'])
    obs, weight = resample(obs, fps), resample(weight, fps)
    obs, yaw = face_forward(obs, weight, skel)
    person = estimate_skeleton(skel, obs, weight > 0, UNRELIABLE_LENGTH)
    ground = priors['ground']['lowest_foot_y_median'] * person.height() / priors['ground']['height']
    rel = obs
    obs = anchor_root(obs, weight, skel, ground)
    # All 33 MediaPipe points in the same frame (for inspection): same axis change, resampling,
    # yaw and per-frame root offset as the joints, but no gap filling or smoothing.
    pts = resample(mp_to_yup_cm(world), fps) @ yaw_matrix(-yaw).T
    pts = pts + (obs[:, :1] - rel[:, :1])
    pts_vis = resample(vis, fps)
    obs = smooth_positions(obs, np.zeros(len(obs)), obs_sigma)
    from pipeline.core.mediapipe_format import LANDMARKS, TO_MIXAMO
    from pipeline.viz import MP_EDGES
    extra = dict(landmarks=pts, landmark_visibility=pts_vis, landmark_edges=np.array(MP_EDGES),
                 landmark_fitted=np.array(sorted({LANDMARKS[lm] for lm, _ in TO_MIXAMO.values()})),
                 landmark_shown=np.array(sorted({LANDMARKS[lm] for lm, _ in TO_MIXAMO.values()} | {0, 7, 8, 29, 30})))
    return obs, weight, person, extra, dict(fps_in=fps, frames_used=[int(first), int(last)],
                                            detection_rate=float(det.mean()), initial_yaw_deg=float(np.degrees(yaw)))


def wham_input(video, out, fps_max=31.):
    """WHAM's temporal model is trained at 30 fps and OpenCV cannot decode AV1: give WHAM an H.264
    30-fps copy (out/video_wham.mp4) when the source is faster or not decodable; else the source.
    Halves the per-frame cost (2D pose, features) for 60-fps phone videos."""
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                            'stream=codec_name,avg_frame_rate', '-of', 'csv=p=0', str(video)],
                           capture_output=True, text=True).stdout.strip().split(',')
    codec, rate = probe[0], probe[1] if len(probe) > 1 else '0/1'
    num, den = (float(x) for x in rate.split('/')) if '/' in rate else (float(rate), 1.)
    fps = num / den if den else 0.
    if codec in ('h264', 'hevc', 'mpeg4', 'vp8', 'vp9') and fps <= fps_max:
        return video
    dst = out / 'video_wham.mp4'
    if not dst.exists():
        print(f'[S0] {video.name}: {codec} @ {fps:.2f} fps -> H.264 @ {min(fps, 30.):.0f} fps ({dst.name})', flush=True)
        vf = ['-vf', 'fps=30'] if fps > fps_max else []
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(video), *vf, '-c:v', 'libx264', '-crf', '18',
                        '-pix_fmt', 'yuv420p', '-an', str(dst)], check=True)
    return dst


def run_s0_wham(video, out):
    if not WHAM_PYTHON.exists():
        raise FileNotFoundError(f'{WHAM_PYTHON} (WHAM venv) missing')
    subprocess.run([str(WHAM_PYTHON), '-W', 'ignore', str(WHAM_EXTRACT), str(video), str(out)], check=True)


def run_s1_wham(wh, skel, priors, obs_sigma=1.5, skip=None, ground_mode='clip'):
    """WHAM already gives a world trajectory: only resample, face +Z and put feet on the floor."""
    from pipeline.core.canonicalize import face_forward, fill_gaps, ground_align, resample, yaw_matrix
    from pipeline.core.filters import smooth_positions
    from pipeline.core.person import estimate_skeleton
    from pipeline.core.wham_format import (DEFAULT_SKIP, SMPL_EDGES, SMPL_NAMES, TO_MIXAMO,
                                           UNRELIABLE_LENGTH, joints_to_obs)
    skip = DEFAULT_SKIP if skip is None else tuple(skip)
    det = wh['detected']
    if det.mean() < 0.5:
        raise RuntimeError(f'Person tracked in only {det.mean():.0%} of frames')
    first, last = np.flatnonzero(det)[[0, -1]]
    joints, det = wh['joints'][first:last + 1], det[first:last + 1]
    obs, weight = joints_to_obs(joints, det, skel, skip)
    obs, weight = fill_gaps(obs, weight)
    fps = float(wh['fps'])
    obs, weight = resample(obs, fps), resample(weight, fps)
    obs, yaw = face_forward(obs, weight, skel)
    person = estimate_skeleton(skel, obs, weight > 0, UNRELIABLE_LENGTH)
    ground = priors['ground']['lowest_foot_y_median'] * person.height() / priors['ground']['height']
    before = obs
    obs = ground_align(obs, weight, skel, ground, mode=ground_mode)
    pts = resample(np.nan_to_num(joints) * 100., fps) @ yaw_matrix(-yaw).T + (obs[:, :1] - before[:, :1])
    obs = smooth_positions(obs, np.zeros(len(obs)), obs_sigma)
    fitted = sorted({SMPL_NAMES.index(n) for j, (n, _) in TO_MIXAMO.items() if j not in skip})
    extra = dict(landmarks=pts, landmark_visibility=np.ones(pts.shape[:2]), landmark_edges=np.array(SMPL_EDGES),
                 landmark_fitted=np.array(fitted), landmark_shown=np.arange(24))
    return obs, weight, person, extra, dict(fps_in=fps, frames_used=[int(first), int(last)],
                                            detection_rate=float(wh['detected'].mean()),
                                            initial_yaw_deg=float(np.degrees(yaw)), estimator='wham', ground_mode=ground_mode,
                                            wham_betas=wh['betas'].tolist(), wham_skipped=list(skip))


def person_writer(skel, person):
    from data_preprocess.Mixamo.bvh_parser import BVH_file
    from data_preprocess.Mixamo.bvh_writer import BVH_writer
    from pipeline.core.paths import STD_BVH_DIR
    f = BVH_file(str(STD_BVH_DIR / 'BigVegas.bvh'))
    edges = [(e[0], e[1], person.offsets[e[1]].copy()) for e in f.edges]
    return BVH_writer(edges, f.names)


def main(argv=None):
    from pipeline.core import quat as Q
    from pipeline.core.ik import PRESETS, solve
    from pipeline.core.pan_runtime import PanRuntime, canonical_sign, ood_stats
    from pipeline.core.priors import load_priors
    from pipeline.core.skeleton import Skeleton

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('video')
    ap.add_argument('--name', default=None)
    ap.add_argument('--targets', nargs='*', default=list(DEFAULT_TARGETS),
                    help='group-B characters to retarget to (pass none to skip, e.g. for coach_to_user)')
    ap.add_argument('--out', default=None, help='output directory (default runs/video[_wham]/<name>)')
    ap.add_argument('--preset', default='full_root', choices=list(PRESETS))
    ap.add_argument('--obs-sigma', type=float, default=1.5)
    ap.add_argument('--estimator', default='mediapipe', choices=['mediapipe', 'wham'])
    ap.add_argument('--wham-skip', nargs='*', default=None,
                    help='Mixamo joints not fitted from WHAM (default: clavicles); pass nothing to fit all')
    ap.add_argument('--ground', default='clip', choices=['clip', 'frame'],
                    help='WHAM only: floor from the whole clip (default) or per frame (removes vertical '
                         'drift; needs a foot on the ground in every frame, e.g. squats, not jumps)')
    ap.add_argument('--reuse-landmarks', action='store_true', help='skip S0 if its output exists')
    ap.add_argument('--device', default='cuda:0')
    opt = ap.parse_args(argv)

    video = Path(opt.video).resolve()
    out_arg = Path(opt.out).resolve() if opt.out else None       # relative to the caller's cwd
    enter_repo()
    out = (out_arg if out_arg else
           REPO / ('runs/video' if opt.estimator == 'mediapipe' else 'runs/video_wham') / (opt.name or video.stem))
    out.mkdir(parents=True, exist_ok=True)
    s0_path = out / ('landmarks.npz' if opt.estimator == 'mediapipe' else 'wham.npz')
    video_original = video
    if opt.estimator == 'wham':
        video = wham_input(video, out)          # frames, fps_in and report['video'] refer to this file
    if not (opt.reuse_landmarks and s0_path.exists()):
        print(f'[S0] {opt.estimator}', flush=True)
        (run_s0 if opt.estimator == 'mediapipe' else run_s0_wham)(video, s0_path)
    s0 = dict(np.load(s0_path, allow_pickle=False))

    skel = Skeleton.from_character('BigVegas')
    priors = load_priors('BigVegas')
    print('[S1] canonicalise', flush=True)
    s1 = run_s1 if opt.estimator == 'mediapipe' else run_s1_wham
    obs, weight, person, extra, info = (s1(s0, skel, priors, opt.obs_sigma) if opt.estimator == 'mediapipe'
                                        else s1(s0, skel, priors, opt.obs_sigma, opt.wham_skip, opt.ground))
    np.savez_compressed(out / 'observation.npz', obs=obs, weight=weight, person_offsets=person.offsets, **extra)

    runtime = PanRuntime(source='BigVegas', targets=opt.targets or ['Mousey_m'], device=opt.device)
    pan_stats = tuple(s.cpu() for s in runtime.source_stats)
    cfg = PRESETS[opt.preset]
    if not cfg.free_root:
        from dataclasses import replace
        cfg = replace(cfg, free_root=True, w_root_acc=PRESETS['full_root'].w_root_acc)
    print(f'[S2] IK ({opt.preset}) on {len(obs)} frames, person height {person.height():.1f} cm', flush=True)
    sol = solve(obs, weight, person, priors, pan_stats, None, cfg, device=opt.device)

    q = Q.normalize(sol['q'].double())
    root = sol['root'].double()
    person_writer(skel, person).write_raw(skel.joint_to_raw(q, root).float(), 'quaternion',
                                          str(out / 'source_person.bvh'))
    raw = canonical_sign(skel.joint_to_raw(q, root * skel.height() / person.height()), *pan_stats)
    runtime.writer(0, 0).write_raw(raw.float(), 'quaternion', str(out / 'source_pan_input.bvh'))
    if opt.targets:
        print('[S3/S4] PAN', flush=True)
        res = runtime.retarget(raw)
        for i, t in enumerate(opt.targets):
            runtime.writer(1, i).write_raw(res[t], 'quaternion', str(out / f'{t}.bvh'))
    fit = np.linalg.norm(sol['positions'].numpy() - obs, axis=-1)[weight > 0].mean()
    report = dict(video=str(video), video_original=str(video_original), **info, frames=len(obs), person_height_cm=person.height(),
                  ik_preset=opt.preset, ik_final=sol['final'], ik_fit_cm=float(fit),
                  ik_seconds=sol['seconds'], pan_input=ood_stats(runtime.normalize_source(raw.float())),
                  targets=opt.targets)
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
