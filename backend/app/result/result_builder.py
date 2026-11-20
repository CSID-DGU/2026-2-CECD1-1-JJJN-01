"""Pipeline outputs -> frontend `AnalysisResult` (frontend/src/types/analysis.ts).

Inputs: a coach_to_user output directory (coach_on_user_pan.bvh, user.bvh). Joint positions come
from the official BVH FK (pipeline.bvh_to_npz); motion_scoring runs in-process so that its frame
pairs (Result.debug['path']) and its playback order (Result.playback()) are available -- score.json
does not contain them.

Mapping
  skeleton   SMPL-22 (bvh_to_npz) -> the 15 frontend joint names, cm -> m, Y up, floor y = 0,
             each sequence's first-frame pelvis moved to x = z = 0 (overlay view)
  segments   motion_scoring units: structure -> verdict (substitution -> substitute),
             student_t / coach_t -> ranges, timing as motion_scoring gives it (label slow/fast,
             ratio relative to the global tempo; no label -> ok), bones -> deviations (see deviations)
  label      squat only: from the pelvis height over the segment (see squat_label)
  playback   both skeletons are sent in the scorer's playback order, so frame k of one belongs with
             frame k of the other: frameMap = the coach frame to draw and who moves at k, and a
             segment's studentRange = its time on that timeline. See frame_map.py. The scorer's own
             frame pairs are only used here, to pair the repetitions of poseMetrics: the response has no
             `alignment` (and no `score`).
  mirrored   if the scorer matched the mirrored user, the coach is shown mirrored (x -> -x,
             left/right joints swapped) and deviation bones are named on the user's own side
  pose       pose_metrics.py: poseMetrics holds joint angles at the squat bottoms (per repetition) and
             standing
  feedback   not here: added by the llm stage. There is no total score (the formula is not decided).
"""
import sys

import numpy as np

from ..constants import PIPELINE_REPO, POSE_SAME_DEG
from .pose_metrics import Body, pose_metrics
from .frame_map import in_playback_order

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
# the bones an exercise is judged by (motion_scoring names): a segment's deviations name these first
MAIN_BONES = {'squat': ('trunk', 'l_thigh', 'r_thigh', 'l_shank', 'r_shank')}
MAX_DEVIATIONS = 2                       # per segment
# deviations[].direction: the largest component of the difference and its sign
DIRECTION = {('up', True): 'up', ('up', False): 'down', ('forward', True): 'forward', ('forward', False): 'backward',
             ('outward', True): 'outward', ('outward', False): 'inward', ('right', True): 'right', ('right', False): 'left'}


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


def deviations(bones, exercise):
    """The bones of a match unit (motion_scoring `bones`: per bone the largest difference in the unit and which
    way it goes) -> the segment's deviations: the bones that differ by at least POSE_SAME_DEG, the exercise's
    main bones before the others, the largest first, at most MAX_DEVIATIONS. A bone is named on the user's own
    side (student_bone: the other side if the scorer matched the mirrored user).
    angleDiffDeg is the angle between the two bones (never negative); components splits it into up / forward /
    outward (away from the body's midline; for the torso and the head: right = the user's right), and
    direction names the largest of them."""
    main = MAIN_BONES.get(exercise, ())
    out = []
    for b in sorted((b for b in bones if b['deg'] >= POSE_SAME_DEG), key=lambda b: (b['bone'] not in main, -b['deg'])):
        comp = {'outward' if axis == 'out' else axis: float(v) for axis, v in b['dir'].items()}
        axis, value = max(comp.items(), key=lambda kv: abs(kv[1]))
        out.append(dict(bone=FRONT_BONES.get(b['student_bone'], b['student_bone']), angleDiffDeg=float(b['deg']),
                        direction=DIRECTION[axis, value > 0], components=comp))
    return out[:MAX_DEVIATIONS]


def build_segments(units, exercise, label_fn):
    out = []
    for k, u in enumerate(units):
        st, ct = u.get('student_t'), u.get('coach_t')
        seg = dict(id=f'seg-{k}', label=label_fn(u, k == 0, k == len(units) - 1),
                   verdict=VERDICT[u['structure']], deviations=deviations(u.get('bones') or [], exercise))
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
        out.append(seg)
    return out


def build_result(job_id, exercise, retarget_dir, with_score=False, playback=True):
    """-> AnalysisResult dict (and the motion_scoring to_dict() when with_score, for the LLM stage).
    playback=False: the two skeletons as recorded, every range on the videos' own times, and the scorer's
    frame pairs as `alignment` in place of `frameMap` (for experiments that check the analysis against the
    videos; the API never sends this form)."""
    from pipeline.bvh_to_npz import bvh_joints
    from pipeline.common.paths import SCORING_DIR
    if str(SCORING_DIR) not in sys.path:
        sys.path.insert(0, str(SCORING_DIR))
    from dataclasses import replace
    from motion_scoring import Config, score, track_from_joints

    coach, names, fps_c = bvh_joints(retarget_dir / 'coach_on_user_pan.bvh')
    user, _, fps_u = bvh_joints(retarget_dir / 'user.bvh')
    cfg = replace(Config(), auto_tau=True, pose_trim=0.3)               # as pipeline.validation.score_user
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
    segments = build_segments(d['segments'], exercise, label)
    body_u, body_c = Body(user, names, fps_u), Body(coach, names, fps_c)
    metrics = pose_metrics(body_u, body_c, alignment, segments, d['mirrored'])     # all on the videos' own times
    student = skeleton_sequence(user, names, fps_u)
    shown_coach = skeleton_sequence(coach, names, fps_c, mirror=d['mirrored'])
    view = dict(alignment=[list(p) for p in alignment])    # playback=False only: the scorer's frame pairs
    if playback:                                           # what the viewer draws: frame k of both (frame_map.py)
        view = dict(frameMap=in_playback_order(res.playback(), student, shown_coach, segments))
    result = dict(
        jobId=job_id, exercise=exercise,
        segments=segments,
        student=student,
        coachRetargeted=shown_coach,
        **view,
        poseMetrics=metrics,
    )
    return (result, d) if with_score else result
