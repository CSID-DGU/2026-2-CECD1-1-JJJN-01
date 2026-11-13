"""Pipeline outputs -> frontend `AnalysisResult` (frontend/src/types/analysis.ts).

Inputs: a coach_to_user output directory (coach_on_user_pan.bvh, user.bvh). Joint positions come
from the official BVH FK (pipeline.bvh_to_npz); motion_scoring runs in-process so that the frame
alignment (Result.debug['path']) is available -- score.json does not contain it.

Mapping
  skeleton   SMPL-22 (bvh_to_npz) -> the 15 frontend joint names, cm -> m, Y up, floor y = 0,
             each sequence's first-frame pelvis moved to x = z = 0 (overlay view)
  segments   motion_scoring units: structure -> verdict (substitution -> substitute),
             student_t / coach_t -> ranges, timing as motion_scoring gives it (label slow/fast,
             ratio relative to the global tempo; no label -> ok), pose.worst -> deviations
  label      squat only: from the pelvis height over the segment (see squat_label)
  alignment  DTW path [(kind, coach i, student j)] without drops -> [[student j, coach i]]
  frameMap   per user frame: the coach frame to draw (fractional, smoothed) and the step kind (frame_map.py)
  mirrored   if the scorer matched the mirrored user, the coach is shown mirrored (x -> -x,
             left/right joints swapped) and deviation bones are named on the user's own side
  pose       pose_metrics.py: deviations get direction / components (signed, body frame), and
             poseMetrics holds joint angles at the squat bottoms (per repetition) and standing
  score      placeholder total (config.DEFAULT_SCORE_TOTAL); feedback is added by the llm stage
"""
import sys

import numpy as np

from .config import DEFAULT_SCORE_TOTAL, PIPELINE_REPO
from .pose_metrics import Body, pose_metrics, signed_deviations
from .frame_map import frame_map

if str(PIPELINE_REPO) not in sys.path:
    sys.path.insert(0, str(PIPELINE_REPO))

# frontend joint name -> SMPL-22 name (pipeline/bvh_to_npz.py; BigVegas has no nose: Head joint)
FRONT_JOINTS = {
    'nose': 'head', 'neck': 'neck',
    'left_shoulder': 'left_shoulder', 'right_shoulder': 'right_shoulder',
    'left_elbow': 'left_elbow', 'right_elbow': 'right_elbow',
    'left_wrist': 'left_wrist', 'right_wrist': 'right_wrist',
    'pelvis': 'pelvis', 'left_hip': 'left_hip', 'right_hip': 'right_hip',
    'left_knee': 'left_knee', 'right_knee': 'right_knee',
    'left_ankle': 'left_ankle', 'right_ankle': 'right_ankle',
}
# motion_scoring bone -> frontend bone (frontend/src/utils/skeleton.ts BONES)
FRONT_BONES = {
    'l_thigh': 'left_thigh', 'r_thigh': 'right_thigh', 'l_shank': 'left_shin', 'r_shank': 'right_shin',
    'trunk': 'torso', 'head': 'head', 'l_upper_arm': 'left_upper_arm', 'r_upper_arm': 'right_upper_arm',
    'l_forearm': 'left_forearm', 'r_forearm': 'right_forearm',
}
VERDICT = {'match': 'match', 'missing': 'missing', 'extra': 'extra', 'substitution': 'substitute'}


def swap_side(name):
    for a, b in (('left_', 'right_'), ('right_', 'left_')):
        if name.startswith(a):
            return b + name[len(a):]
    return name


def skeleton_sequence(joints_cm, smpl_names, fps, mirror=False):
    """(T, 22, 3) cm in SMPL-22 order -> SkeletonSequence dict."""
    idx = [smpl_names.index(s) for s in FRONT_JOINTS.values()]
    p = joints_cm[:, idx].astype(np.float64) / 100.
    pelvis = list(FRONT_JOINTS).index('pelvis')
    p[..., [0, 2]] -= p[0, pelvis, [0, 2]]
    names = list(FRONT_JOINTS)
    if mirror:
        p[..., 0] *= -1
        names = [swap_side(n) for n in names]
    return dict(jointNames=names, fps=round(float(fps), 3), frames=np.round(p, 4).tolist())


def squat_label(y, t0, t1, fps, first, last):
    """Korean label of a squat segment from the pelvis height y (m, per frame) over [t0, t1] s."""
    n = len(y)
    f0, f1 = (min(max(int(round(t * fps)), 0), n - 1) for t in (t0, t1))
    R = max(float(y.max() - y.min()), 1e-6)
    seg = y[f0:f1 + 1]
    dy = float(y[f1] - y[f0])
    if seg.min() < min(y[f0], y[f1]) - 0.25 * R and abs(dy) < 0.25 * R:
        return '하강 · 상승'
    if dy < -0.25 * R:
        return '하강'
    if dy > 0.25 * R:
        return '상승'
    if float(seg.mean() - y.min()) < 0.35 * R:
        return '바닥 유지'
    if first:
        return '시작 자세'
    if last:
        return '마무리 자세'
    return '선 자세 유지'


def build_segments(units, mirrored, label_fn):
    out = []
    for k, u in enumerate(units):
        st, ct = u.get('student_t'), u.get('coach_t')
        seg = dict(id=f'seg-{k}', label=label_fn(u, k == 0, k == len(units) - 1),
                   verdict=VERDICT[u['structure']], deviations=[])
        if st:                     # missing: zero-length [t, t] at the user's time (timeline position)
            seg['studentRange'] = [float(st[0]), float(st[1])]
        if ct:
            seg['coachRange'] = [float(ct[0]), float(ct[1])]
        if u['structure'] == 'match':
            t = u.get('timing') or {}
            seg['timing'] = dict(label=t.get('label') or 'ok', ratio=float(t.get('ratio', 1.0)))
        if u.get('hold'):          # coach pause vs the student's slow-down there (motion_scoring short holds)
            h = u['hold']
            seg['hold'] = dict(coachSec=float(h['coach_s']), userSec=float(h['student_s']), ratio=float(h['ratio']))
        for w in (u.get('pose') or {}).get('worst', []):
            bone = FRONT_BONES.get(w['bone'], w['bone'])
            seg['deviations'].append(dict(bone=swap_side(bone) if mirrored else bone,
                                          angleDiffDeg=float(w['diff_deg'])))
        out.append(seg)
    return out


def build_result(job_id, exercise, retarget_dir, with_score=False):
    """-> AnalysisResult dict (and the motion_scoring to_dict() when with_score, for the LLM stage)."""
    from pipeline.bvh_to_npz import bvh_joints
    from pipeline.score_user import SCORING_DIR
    if str(SCORING_DIR) not in sys.path:
        sys.path.insert(0, str(SCORING_DIR))
    from dataclasses import replace
    from motion_scoring import Config, score, track_from_joints

    coach, names, fps_c = bvh_joints(retarget_dir / 'coach_on_user_pan.bvh')
    user, _, fps_u = bvh_joints(retarget_dir / 'user.bvh')
    cfg = replace(Config(), auto_tau=True, pose_trim=0.3)               # as pipeline.score_user
    track = lambda j, fps: track_from_joints(j, np.arange(len(j)) / fps, 'y', None, names)
    res = score(track(coach, fps_c), track(user, fps_u), cfg, try_mirror=True)
    d = res.to_dict()

    pel = names.index('pelvis')
    y_user, y_coach = user[:, pel, 1] / 100., coach[:, pel, 1] / 100.

    def label(u, first, last):
        st, ct = u.get('student_t'), u.get('coach_t')
        if st and st[1] > st[0]:
            return squat_label(y_user, st[0], st[1], fps_u, first, last)
        if ct:
            return squat_label(y_coach, ct[0], ct[1], fps_c, first, last)
        return '구간'

    alignment = sorted({(int(j), int(i)) for kind, i, j in res.debug['path'] if kind not in ('drop_c', 'drop_s')})
    segments = build_segments(d['segments'], d['mirrored'], label)
    body_u, body_c = Body(user, names, fps_u), Body(coach, names, fps_c)
    signed_deviations(segments, body_u, body_c, alignment, d['mirrored'])
    result = dict(
        jobId=job_id, exercise=exercise,
        score=dict(total=DEFAULT_SCORE_TOTAL),
        segments=segments,
        student=skeleton_sequence(user, names, fps_u),
        coachRetargeted=skeleton_sequence(coach, names, fps_c, mirror=d['mirrored']),
        alignment=[list(p) for p in alignment],
        frameMap=frame_map(res.debug['path'], len(user), len(coach)),
        poseMetrics=pose_metrics(body_u, body_c, alignment, segments, d['mirrored']),
    )
    return (result, d) if with_score else result
