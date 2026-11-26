"""SMPL overlay of a scoring run (PLAN.md). Run with the WHAM venv (smplx) from engine/:

  ../.venv-wham/bin/python -m pipeline.overlay.compare --run NAME [--exercise squat] [--anchor feet|pelvis]

NAME is a pipeline.validation.score_user output (runs/scoring/NAME/inputs.json names the two run_video runs;
their wham.npz must carry the camera fields of tools/pose_estimation/wham_extract.py, i.e. made with
--rerun after that change). The playback pb comes from the same scoring as the API (backend
build_result: scorer_setup(exercise) + score(..., try_mirror=True)), not from score_user's score.json.

Writes runs/overlay/NAME/:
  playback.json     pb as used (student, coach, segments, dt), BVH frame numbers of user.bvh / coach BVH
  student_self.mp4  the student's own SMPL (own pose, own betas) placed by the common path, over the video
  overlay_m1.mp4    method 1 coach (coach WHAM rotations, student betas / heading / feet / camera), blue
  overlay_m2.mp4    method 2 coach (PAN output coach_on_user_pan.bvh converted to SMPL), teal. The conversion is
                    calibrated on the coach's own IK BVH against the coach's WHAM (coach_pose.calibrate_from_wham:
                    one fixed rotation per joint, how BigVegas and SMPL define their joints differently)
  overlay_both.mp4  both coaches on one picture (method 1 drawn first, method 2 over it)
  coach_self.mp4    the coach's own WHAM SMPL (own betas, heading, feet, camera) over the coach's video, in its
                    own time (no playback): how well WHAM saw the coach, what method 1 starts from. Blue
  coach_self_m2.mp4 the coach's IK BVH (source_pan_input.bvh, before PAN) through method 2's conversion, coach
                    rest joints, placed the same way: method 2's conversion path on the coach's own video. Teal.
                    Calibrated on the same frames it shows, so it is the best case for method 2
                    (--self-only: these two and playback.json only)
  scene_m1.npz, scene_m2.npz
                    per playback frame k, world frame (WHAM's, Y up, metres): SMPL params and SMPL-22 joints
                    of the student and the coach, betas (student), joint_names, student_frame / coach_frame
                    (npz frames used), mirrored, anchor. What a 3D viewer draws; the videos draw the same
                    placement after the camera move.
  compare.json      method 1 vs method 2 over every coach BVH frame (no mirroring, no placement): rotation
                    difference per joint, bone direction difference (motion_scoring's bones, student betas,
                    heading taken out); the round trip of the conversion (the coach's IK BVH
                    source_pan_input.bvh -> SMPL vs the coach's WHAM rotations); the part of the PAN pelvis /
                    chest motion one SMPL rotation cannot follow (fit_residual_deg); a mirror check; the
                    calibration (per-joint angle, spread over the frames); and m1 vs m2 / the round trip
                    without the calibration (without_calibration).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.common.paths import PROJECT, REPO  # noqa: E402
from pipeline.overlay import coach_pose as C  # noqa: E402
from pipeline.overlay import place as P  # noqa: E402
from pipeline.overlay import render as R  # noqa: E402
from pipeline.overlay.smpl_np import SMPL22, Body, aa_to_mat, mat_to_aa  # noqa: E402


def api_playback(retarget_dir, exercise):
    """res.playback() and res.to_dict()['mirrored'] exactly as backend build_result makes them."""
    if str(PROJECT / 'backend') not in sys.path:
        sys.path.insert(0, str(PROJECT / 'backend'))
    from app.result.result_builder import scorer_setup
    from pipeline.bvh_to_npz import bvh_joints
    cfg, weights = scorer_setup(exercise)
    from motion_scoring import score, track_from_joints
    coach, names, fps_c = bvh_joints(retarget_dir / 'coach_on_user_pan.bvh')
    user, _, fps_u = bvh_joints(retarget_dir / 'user.bvh')
    track = lambda j, fps: track_from_joints(j, np.arange(len(j)) / fps, 'y', None, names)
    res = score(track(coach, fps_c), track(user, fps_u), cfg, weights, try_mirror=True)
    return res.playback(), bool(res.to_dict()['mirrored'])


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


class Run:
    """A run_video output: wham.npz, report.json (frames_used, fps_in, the video WHAM read)."""

    def __init__(self, path):
        self.dir = Path(path)
        self.report = json.loads((self.dir / 'report.json').read_text())
        self.npz = dict(np.load(self.dir / 'wham.npz'))
        missing = [k for k in ('pose_cam', 'trans_cam', 'focal', 'img_size') if k not in self.npz]
        if missing:
            raise SystemExit(f'{self.dir}/wham.npz has no {missing}: rerun WHAM (score_user --rerun)')
        self.video = Path(self.report['video'])          # video_wham.mp4 when converted, else the source

    def frame(self, bvh_i):
        return min(P.npz_frame(bvh_i, self.report), len(self.npz['detected']) - 1)


def bvh_frames(path):
    for line in open(path):
        if line.startswith('Frames:'):
            return int(line.split()[1])
    raise ValueError(f'{path}: no Frames line')


def run_overlay(body, student, coaches, pb, mirrored, out, anchor, alpha):
    """One pass over the playback. coaches: {method: pose (coach BVH frame indexing)}. Writes student_self.mp4,
    overlay_<method>.mp4 and scene_<method>.npz."""
    w, h = student.npz['img_size']
    cam = R.Camera(w, h, student.npz['focal'])
    s_pose = P.pose_from_npz(student.npz)
    frames = R.FrameReader(student.video)
    fps = 1. / pb['dt']
    writers = {'self': R.writer(out / 'student_self.mp4', fps)}
    writers.update({m: R.writer(out / f'overlay_{m}.mp4', fps) for m in coaches})
    if len(coaches) > 1:
        writers['both'] = R.writer(out / 'overlay_both.mp4', fps)
    color = {'m1': R.COLORS['coach'], 'm2': R.COLORS['coach_m2']}
    K = len(pb['student'])
    nan = lambda *shape: np.full((K,) + shape, np.nan)
    scene = {m: dict(student_global_orient=nan(3), student_body_pose=nan(69), student_transl=nan(3),
                     coach_global_orient=nan(3), coach_body_pose=nan(69), coach_transl=nan(3),
                     student_joints=nan(22, 3), coach_joints=nan(22, 3),
                     student_frame=np.zeros(K, int), coach_frame=np.zeros(K, int)) for m in coaches}
    cams, st, t0 = {}, None, time.time()

    def mesh(params):
        c = P.to_camera(params, st)
        return body(c['global_orient'], c['body_pose'], c['betas'], c['transl'])[0][0]

    for k, (si, ci) in enumerate(zip(pb['student'], pb['coach'])):
        f = student.frame(int(si))
        if st is None or st.f != f:
            st = P.StudentFrame(body, student.npz, f)
            st.f = f
            cams = {}
            if st.ok:
                sp, sj = P.place_world(body, s_pose['tilt'][f], s_pose['body_pose'][f], st, anchor)
                cams['self'] = (sp, sj, mesh(sp))
        bg = frames.get(f)
        drawn = {'self': [(cams['self'][2], R.COLORS['student'])] if 'self' in cams else []}
        for m, pose in coaches.items():
            sc = scene[m]
            sc['student_frame'][k], sc['coach_frame'][k] = f, pose['npz_frame'][int(ci)]
            drawn[m] = []
            if not st.ok:
                continue
            sp, sj, _ = cams['self']
            cp, cj = P.place_world(body, pose['tilt'][int(ci)], pose['body_pose'][int(ci)], st, anchor)
            for who, prm, j in (('student', sp, sj), ('coach', cp, cj)):
                sc[f'{who}_global_orient'][k], sc[f'{who}_body_pose'][k] = prm['global_orient'], prm['body_pose']
                sc[f'{who}_transl'][k], sc[f'{who}_joints'][k] = prm['transl'], j[:22]
            drawn[m] = [(mesh(cp), color[m])]
        if 'both' in writers:
            drawn['both'] = [mesh_ for m in coaches for mesh_ in drawn[m]]
        for m, wr in writers.items():
            wr.append_data(R.composite(bg, drawn[m], body.faces, cam, alpha)[..., ::-1])
    for wr in writers.values():
        wr.close()
    for m, sc in scene.items():
        np.savez_compressed(out / f'scene_{m}.npz', **sc, betas=student.npz['betas'],
                            joint_names=np.array(SMPL22), mirrored=mirrored, anchor=anchor, fps=fps)
    names = ['student_self.mp4'] + [f'overlay_{m}.mp4' for m in writers if m != 'self'] + \
        [f'scene_{m}.npz' for m in coaches]
    print(f'{", ".join(names)}: {K} frames in {time.time() - t0:.0f} s')


def bvh_to_npz_frames(pose, run):
    """A pose indexed by BVH frame (run_video: from frames_used[0], 30 fps) -> indexed by the run's npz /
    video frame (NaN where the BVH has no frame), the inverse of place.npz_frame."""
    T = len(run.npz['detected'])
    i = np.round((np.arange(T) - run.report['frames_used'][0]) * P.BVH_FPS / run.report['fps_in']).astype(int)
    ok = (i >= 0) & (i < len(pose['body_pose']))
    out = dict(tilt=np.full((T, 3, 3), np.nan), body_pose=np.full((T, 69), np.nan))
    out['tilt'][ok], out['body_pose'][ok] = pose['tilt'][i[ok]], pose['body_pose'][i[ok]]
    return out


def self_overlay(body, run, poses, out, anchor, alpha):
    """A run's own body on its own video, in the video's own time (no playback): poses {file name: (pose
    indexed by npz frame, rgb)}, each placed like student_self (the run's betas, heading, feet, camera).
    Frames WHAM missed, or a pose has no frame for, show the video alone."""
    w, h = run.npz['img_size']
    cam = R.Camera(w, h, run.npz['focal'])
    frames = R.FrameReader(run.video)
    writers = {name: R.writer(out / name, run.report['fps_in']) for name in poses}
    T, t0 = len(run.npz['detected']), time.time()
    for f in range(T):
        bg = frames.get(f)
        st = P.StudentFrame(body, run.npz, f)
        for name, (pose, rgb) in poses.items():
            meshes = []
            if st.ok and np.isfinite(pose['body_pose'][f]).all() and np.isfinite(pose['tilt'][f]).all():
                prm, _ = P.place_world(body, pose['tilt'][f], pose['body_pose'][f], st, anchor)
                c = P.to_camera(prm, st)
                meshes = [(body(c['global_orient'], c['body_pose'], c['betas'], c['transl'])[0][0], rgb)]
            writers[name].append_data(R.composite(bg, meshes, body.faces, cam, alpha)[..., ::-1])
    for wr in writers.values():
        wr.close()
    print(f'{", ".join(poses)}: {T} frames in {time.time() - t0:.0f} s')


SMPL24 = SMPL22 + ['left_hand', 'right_hand']
SWAP24 = list(range(24))
for _a, _b in C.MIRROR_PAIRS:
    SWAP24[_a], SWAP24[_b] = _b, _a


def stats(x, axis=0):
    x = np.asarray(x)
    return dict(mean=round(float(np.mean(x)), 3), median=round(float(np.median(x)), 3), max=round(float(np.max(x)), 3))


def joint_rotations(pose):
    """(T, 24, 3, 3): root tilt and the 23 local rotations."""
    T = len(pose['body_pose'])
    return np.concatenate([pose['tilt'][:, None], aa_to_mat(pose['body_pose'].reshape(T, 23, 3))], 1)


def rotation_diff(a, b):
    """Per frame and joint, degrees between the two rotations. (T, 24)"""
    Ra, Rb = joint_rotations(a), joint_rotations(b)
    cos = (np.trace(np.swapaxes(Ra, -1, -2) @ Rb, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def bone_diff(body, a, b, betas):
    """Per frame and motion_scoring bone, degrees between the bone directions (SMPL FK, the same betas,
    heading taken out). -> {bone: (T,)}"""
    from motion_scoring.step1_preprocess import SMPL_BONES
    ja = body(mat_to_aa(a['tilt']), a['body_pose'], betas)[1]
    jb = body(mat_to_aa(b['tilt']), b['body_pose'], betas)[1]
    out = {}
    for name, p, c in SMPL_BONES:
        da, db = C.unit(ja[:, c] - ja[:, p]), C.unit(jb[:, c] - jb[:, p])
        out[name] = np.degrees(np.arccos(np.clip((da * db).sum(-1), -1, 1)))
    return out


def comparison(body, a, b, betas):
    rot = rotation_diff(a, b)
    bones = bone_diff(body, a, b, betas)
    return dict(rotation_deg=dict(all=stats(rot[:, :22]),
                                  per_joint={('root_tilt' if j == 0 else SMPL24[j]): stats(rot[:, j])
                                             for j in range(24)}),
                bone_direction_deg=dict(all=stats(np.concatenate(list(bones.values()))),
                                        per_bone={k: stats(v) for k, v in bones.items()}))


def compare_json(body, m1, m2, roundtrip, student, coach, mirrored, rt_source):
    rot = rotation_diff(m1, m2)
    rot_mirrored = rotation_diff(C.mirror(m1), C.mirror(m2))[:, SWAP24]       # the same, joint for joint
    k = dict(tilt=m1['tilt'], body_pose=m1['body_pose'])
    return dict(
        frames=len(m1['body_pose']), mirrored_in_overlay=mirrored,
        note='rotation_deg: root_tilt + local rotations; "all" covers the root and SMPL-22 joints 1-21 '
             '(hands 22-23 have no BigVegas joint: identity in method 2). bone_direction_deg: motion_scoring '
             'SMPL_BONES on SMPL FK with the student betas, heading taken out.',
        m1_vs_m2=comparison(body, m1, m2, student.npz['betas']),
        roundtrip=dict(source=rt_source, note='coach IK BVH (before PAN) -> SMPL with the coach rest joints, '
                                              'against the coach WHAM rotations; coach betas',
                       **comparison(body, k, roundtrip, coach.npz['betas'])),
        fit_residual_deg=dict(pan=m2['fit_residual_deg'], ik_roundtrip=roundtrip['fit_residual_deg']),
        mirror_check_max_deg=round(float(np.abs(rot - rot_mirrored).max()), 6))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, help='runs/scoring/<NAME>')
    ap.add_argument('--exercise', default='squat', help='for scorer_setup (bone weights), as in the API')
    ap.add_argument('--anchor', default='feet', choices=['feet', 'pelvis'])
    ap.add_argument('--alpha', type=float, default=0.6, help='mesh opacity')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--self-only', action='store_true',
                    help='write coach_self.mp4 / coach_self_m2.mp4 (and playback.json) only')
    opt = ap.parse_args(argv)

    inputs = json.loads((REPO / 'runs/scoring' / opt.run / 'inputs.json').read_text())
    student = Run(inputs['user_run'])
    out = REPO / 'runs/overlay' / opt.run
    out.mkdir(parents=True, exist_ok=True)

    pb, mirrored = api_playback(Path(inputs['retarget_dir']), opt.exercise)
    (out / 'playback.json').write_text(json.dumps(jsonable(dict(pb, mirrored=mirrored)), indent=1))
    n_s = len(pb['student'])
    print(f'playback: {n_s} frames, dt {pb["dt"]:.4f}, mirrored {mirrored}, '
          f'student BVH {pb["student"][0]}..{pb["student"][-1]}, coach BVH {pb["coach"][0]}..{pb["coach"][-1]}')

    coach = Run(inputs['coach_run'])
    n_bvh = bvh_frames(Path(inputs['retarget_dir']) / 'coach_on_user_pan.bvh')
    m1 = C.from_wham(coach.npz, coach.report, n_bvh)
    print(f'coach: PAN BVH {n_bvh} frames, WHAM npz {len(coach.npz["detected"])} frames '
          f'(BVH frames from npz {m1["npz_frame"][0]}..{m1["npz_frame"][-1]})')
    retarget = Path(inputs['retarget_dir'])
    rt_source = coach.dir / 'source_pan_input.bvh'
    calib, spread = C.calibrate_from_wham(rt_source, coach.npz, coach.report)
    m2 = C.from_pan_bvh(retarget / 'coach_on_user_pan.bvh', None, calib=calib)
    m2['npz_frame'] = m1['npz_frame']                    # the same coach BVH frames
    m2_rest = C.from_pan_bvh(retarget / 'coach_on_user_pan.bvh', student.npz['rest_joints'])
    roundtrip = C.from_pan_bvh(rt_source, None, calib=calib)
    roundtrip_rest = C.from_pan_bvh(rt_source, coach.npz['rest_joints'])
    if len(roundtrip['body_pose']) != n_bvh:
        raise SystemExit(f'{rt_source}: {len(roundtrip["body_pose"])} frames, PAN BVH {n_bvh}')
    calibration = dict(source=f'{rt_source} vs {coach.dir}/wham.npz pose_world',
                       angle_deg={n: round(float(np.degrees(np.arccos(np.clip((np.trace(k) - 1) / 2, -1, 1)))), 2)
                                  for n, k in zip(C.SMPL_NAMES24, calib)},
                       spread_deg=spread)
    print('calibration: per-joint fixed rotation deg', calibration['angle_deg'])

    body = Body(device=opt.device)
    self_overlay(body, coach, {'coach_self.mp4': (P.pose_from_npz(coach.npz), R.COLORS['coach']),
                               'coach_self_m2.mp4': (bvh_to_npz_frames(roundtrip, coach), R.COLORS['coach_m2'])},
                 out, opt.anchor, opt.alpha)
    if opt.self_only:
        print(f'Wrote {out} (--self-only)')
        return
    report = compare_json(body, m1, m2, roundtrip, student, coach, mirrored, str(rt_source))
    k = dict(tilt=m1['tilt'], body_pose=m1['body_pose'])
    report['calibration'] = calibration
    report['without_calibration'] = dict(
        note='method 2 with the rest-pose match instead of the calibration (what overlay_m2 drew before)',
        m1_vs_m2=comparison(body, m1, m2_rest, student.npz['betas']),
        roundtrip=comparison(body, k, roundtrip_rest, coach.npz['betas']))
    (out / 'compare.json').write_text(json.dumps(report, indent=1))
    print(f'compare.json: m1 vs m2 rotation {report["m1_vs_m2"]["rotation_deg"]["all"]}, '
          f'round trip {report["roundtrip"]["rotation_deg"]["all"]}, mirror check {report["mirror_check_max_deg"]}')

    coaches = {m: C.mirror(p) if mirrored else p for m, p in (('m1', m1), ('m2', m2))}
    run_overlay(body, student, coaches, pb, mirrored, out, opt.anchor, opt.alpha)
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
