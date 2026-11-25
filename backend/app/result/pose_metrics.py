"""Joint angles at the key poses of an exercise, from the two skeletons.

motion_scoring compares bone directions only (the unsigned angle between the user's and the coach's bone,
no joint angles): it was designed for an LLM that also watches the video. The app sends JSON only, so the
pose itself is measured here, from the same SMPL-22 joints the scorer gets (cm, Y up). The coach is already
retargeted onto the user's body, so heights in cm compare directly.

Body frame (per frame, per person): x = the person's right (hip axis on the ground plane), y = up,
z = forward (up x right; checked: the way the toes point).

Joint metrics (per frame)
  kneeFlexDeg     angle between thigh and shin, 0 = straight leg (left / right)
  hipFlexDeg      180 - angle between trunk (pelvis -> neck) and thigh, 0 = upright (left / right)
  trunkLeanDeg    trunk angle from vertical, + = leaning forward
  hipKneeCm       mean hip-joint height - mean knee height; < 0 = hips below the knees
  pelvisDropCm    standing pelvis height (95th percentile) - pelvis height
  kneeWidthRatio  knee distance / ankle distance; < 1 = knees closer together than the ankles

Key poses (values are medians over +- 0.1 s around the frame). Which frames they are is the exercise's to say
(app/result/exercises/<name>.py); their names in the response are those of the squat, the first exercise.
  reps[].bottom  the key pose of every repetition (key_frames; squat: the pelvis-height minima)
  pairing   consistent with the motion_scoring segments (see pose_metrics()): extra / missing repetitions
            (one-to-one); unpaired key poses stay with the other side null (extra / missing repetition)
  standing  the pose the exercise rests in (rest_frames; squat: standing), median over those frames
  thresholds  differences the feedback ignores: constants.POSE_SAME_DEG, POSE_SAME_LEG x leg length (cm),
            POSE_SAME_RATIO; returned with legLengthCm so the frontend / LLM use the same numbers
  If the scorer matched the mirrored user, coach left / right are swapped onto the user's sides.

The bone differences of a segment (deviations) are not measured here: motion_scoring gives them with their
direction (result_builder.deviations).
"""
import numpy as np

from .. import constants

SIDES = ('left', 'right')
UP = np.array([0., 1., 0.])


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _angle(a, b):
    return np.degrees(np.arccos(np.clip((_unit(a) * _unit(b)).sum(-1), -1., 1.)))


class Body:
    """Per-frame joint metrics and body-frame bones of one skeleton (T, 22, 3) cm."""

    def __init__(self, joints, names, fps):
        self.J, self.N, self.fps = np.asarray(joints, float), {n: i for i, n in enumerate(names)}, float(fps)
        j = lambda n: self.J[:, self.N[n]]
        right = j('right_hip') - j('left_hip')
        self.right = _unit(right - (right @ UP)[:, None] * UP)
        self.fwd = np.cross(UP, self.right)
        trunk = j('neck') - j('pelvis')
        m = {}
        for s in SIDES:
            thigh, shin = j(f'{s}_knee') - j(f'{s}_hip'), j(f'{s}_ankle') - j(f'{s}_knee')
            m[f'knee_{s}'] = _angle(thigh, shin)
            m[f'hip_{s}'] = 180. - _angle(trunk, thigh)
        tu = _unit(trunk)
        m['lean'] = _angle(trunk, UP) * np.where((tu * self.fwd).sum(1) >= 0, 1., -1.)
        m['hipKnee'] = (j('left_hip')[:, 1] + j('right_hip')[:, 1] - j('left_knee')[:, 1] - j('right_knee')[:, 1]) / 2
        self.leg = float(np.median(sum(np.linalg.norm(j(f'{s}_knee') - j(f'{s}_hip'), axis=1)
                                       + np.linalg.norm(j(f'{s}_ankle') - j(f'{s}_knee'), axis=1) for s in SIDES) / 2))
        self.pelvis = j('pelvis')[:, 1]
        self.stand_h = float(np.percentile(self.pelvis, 95))
        m['drop'] = self.stand_h - self.pelvis
        m['kneeWidth'] = (np.linalg.norm(j('left_knee') - j('right_knee'), axis=1)
                          / np.maximum(np.linalg.norm(j('left_ankle') - j('right_ankle'), axis=1), 1e-6))
        self.m = m

    def pose(self, frames, swap=False):
        """Joint metrics at a frame (int) or the median over frames (array); swap = mirror left/right."""
        v = {k: float(np.median(x[frames])) for k, x in self.m.items()}
        lr = lambda k: {'left': round(v[f'{k}_right' if swap else f'{k}_left']),
                        'right': round(v[f'{k}_left' if swap else f'{k}_right'])}
        return dict(kneeFlexDeg=lr('knee'), hipFlexDeg=lr('hip'), trunkLeanDeg=round(v['lean']),
                    hipKneeCm=round(v['hipKnee'], 1), pelvisDropCm=round(v['drop'], 1),
                    kneeWidthRatio=round(v['kneeWidth'], 2))

    def window(self, f, sec=0.1):
        """Frames within +- sec of f: key-pose values are medians over this window (robust to jitter)."""
        h = max(1, int(round(sec * self.fps)))
        return np.arange(max(f - h, 0), min(f + h, len(self.pelvis) - 1) + 1)


def _diff(u, c):
    """user - coach for the comparable metrics (both pose() dicts)."""
    out = {}
    for k in ('kneeFlexDeg', 'hipFlexDeg'):
        out[k] = {s: u[k][s] - c[k][s] for s in SIDES}
    for k, nd in (('trunkLeanDeg', 0), ('hipKneeCm', 1), ('pelvisDropCm', 1), ('kneeWidthRatio', 2)):
        out[k] = round(u[k] - c[k], nd) if nd else u[k] - c[k]
    return out


def _segment_at(segments, t, key, tol=0.05):
    """id of the segment whose range (key) contains t (+- tol s), preferring non-empty ranges."""
    best = None
    for s in segments:
        r = s.get(key)
        if r and r[0] - tol <= t <= r[1] + tol:
            if r[1] > r[0]:
                return s['id']
            best = best or s['id']
    return best


def _segment_near(segments, t, key, verdicts, tol=0.1):
    """Segment of one of `verdicts` whose non-empty range (key) contains t (+- tol), nearest first."""
    best, dist = None, None
    for s in segments:
        r = s.get(key)
        if s['verdict'] in verdicts and r and r[1] > r[0] and r[0] - tol <= t <= r[1] + tol:
            dd = max(r[0] - t, t - r[1], 0.)
            if best is None or dd < dist:
                best, dist = s, dd
    return best


def pose_metrics(user, coach, alignment, segments, mirrored, ex):
    """-> AnalysisResult.poseMetrics. user / coach: Body; alignment: [[user j, coach i]]; ex: the module of
    the exercise (app/result/exercises), which says where the key poses are.

    Pairing follows motion_scoring's segments so that poseMetrics and the segments tell the same story:
    a key pose (a squat's bottom) inside an extra (user) / missing (coach) segment that holds a whole movement
    is unpaired; otherwise the user's is mapped to coach time through its match / substitute segment (linear
    within the segment), else through the frame alignment, and taken by the nearest free one of the coach."""
    al = np.asarray(alignment, int).reshape(-1, 2)
    ub, cb = ex.key_frames(user), ex.key_frames(coach)

    def extra(j):
        s = _segment_near(segments, j / user.fps, 'studentRange', ('extra',), tol=0.)
        return s is not None and ex.moves(user, s['studentRange'], j)

    def missing(i):
        s = _segment_near(segments, i / coach.fps, 'coachRange', ('missing',), tol=0.)
        return s is not None and ex.moves(coach, s['coachRange'], i)

    def to_coach(j):
        t = j / user.fps
        s = _segment_near(segments, t, 'studentRange', ('match', 'substitute'))
        if s and s.get('coachRange'):
            (a, b), (c, d) = s['studentRange'], s['coachRange']
            return (c + (min(max(t, a), b) - a) / (b - a) * (d - c)) * coach.fps
        if len(al):
            return float(al[np.argmin(np.abs(al[:, 0] - j)), 1])
        return None

    free = [i for i in cb if not missing(i)]
    pairs, used = [], set()
    for j in ub:
        i = None
        ci = None if extra(j) else to_coach(j)
        if ci is not None:
            cand = [b for b in free if b not in used and abs(b - ci) <= 1.5 * coach.fps]
            if cand:
                i = min(cand, key=lambda b: abs(b - ci))
                used.add(i)
        pairs.append((j, i))
    pairs += [(None, i) for i in cb if i not in used]
    # timeline order: user time, coach-only reps placed by their aligned user frame (else coach time)
    def order(p):
        j, i = p
        if j is not None:
            return j / user.fps
        k = np.argmin(np.abs(al[:, 1] - i)) if len(al) else None
        return al[k, 0] / user.fps if k is not None else i / coach.fps
    reps = []
    for n, (j, i) in enumerate(sorted(pairs, key=order), 1):
        u = user.pose(user.window(j)) if j is not None else None
        c = coach.pose(coach.window(i), swap=mirrored) if i is not None else None
        rep = dict(index=n, userSec=round(j / user.fps, 2) if j is not None else None,
                   coachSec=round(i / coach.fps, 2) if i is not None else None,
                   segmentId=(_segment_at(segments, j / user.fps, 'studentRange') if j is not None
                              else _segment_at(segments, i / coach.fps, 'coachRange')),
                   bottom=dict(user=u, coach=c))
        if u and c:
            rep['bottom']['diff'] = _diff(u, c)
        reps.append(rep)
    su, sc = user.pose(ex.rest_frames(user)), coach.pose(ex.rest_frames(coach), swap=mirrored)
    th = dict(sameAngleDeg=constants.POSE_SAME_DEG, sameHeightCm=round(constants.POSE_SAME_LEG * user.leg, 1),
              sameKneeWidthRatio=constants.POSE_SAME_RATIO)
    return dict(legLengthCm=round(user.leg, 1), thresholds=th, repCount=dict(user=len(ub), coach=len(cb)),
                standing=dict(user=su, coach=sc, diff=_diff(su, sc)), reps=reps,
                bottomSummary=_summary([r['bottom'] for r in reps if r['bottom'].get('diff')]))


def _summary(bottoms):
    """[min, max] over the paired repetitions of each bottom metric (knee / hip: mean of left and right)."""
    if not bottoms:
        return None
    val = lambda p, k: (p[k]['left'] + p[k]['right']) / 2 if isinstance(p[k], dict) else p[k]
    out = {}
    for k, nd in (('kneeFlexDeg', 0), ('hipFlexDeg', 0), ('trunkLeanDeg', 0), ('hipKneeCm', 1),
                  ('pelvisDropCm', 1), ('kneeWidthRatio', 2)):
        out[k] = {}
        for who in ('user', 'coach', 'diff'):
            v = [val(b[who], k) for b in bottoms]
            lo, hi = round(min(v), nd), round(max(v), nd)
            out[k][who] = [int(lo), int(hi)] if nd == 0 else [lo, hi]
    return dict(reps=len(bottoms), **out)
