"""Joint angles, squat key poses and signed bone differences from the two skeletons.

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

Key poses (squat; values are medians over +- 0.1 s around the frame)
  bottoms   pelvis-height minima: prominence >= 35 % of the pelvis range (and >= 7 % of the leg), >= 0.6 s apart
  pairing   consistent with the motion_scoring segments (see pose_metrics()): extra / missing repetitions
            (one-to-one); unpaired bottoms stay with the other side null (extra / missing repetition)
  standing  median over the frames whose pelvis is within 3.5 % of the leg of the standing height
  thresholds  differences the feedback ignores: config.POSE_SAME_DEG, POSE_SAME_LEG x leg length (cm),
            POSE_SAME_RATIO; returned with legLengthCm so the frontend / LLM use the same numbers
  If the scorer matched the mirrored user, coach left / right are swapped onto the user's sides.

Signed bone differences (segment deviations)
  over the aligned frame pairs of the segment: mean of (user bone - coach bone), unit vectors in each
  person's own body frame, split into up / forward / outward (left/right for torso and head) and scaled
  so the components add up in quadrature to the deviation angle motion_scoring reports.
"""
import numpy as np
from scipy.signal import find_peaks

from . import config

SIDES = ('left', 'right')
BONE_JOINTS = {                                  # frontend bone -> (from, to) SMPL-22 joints, as motion_scoring
    'left_thigh': ('left_hip', 'left_knee'), 'right_thigh': ('right_hip', 'right_knee'),
    'left_shin': ('left_knee', 'left_ankle'), 'right_shin': ('right_knee', 'right_ankle'),
    'torso': ('pelvis', 'neck'), 'head': ('neck', 'head'),
    'left_upper_arm': ('left_shoulder', 'left_elbow'), 'right_upper_arm': ('right_shoulder', 'right_elbow'),
    'left_forearm': ('left_elbow', 'left_wrist'), 'right_forearm': ('right_elbow', 'right_wrist'),
}
UP = np.array([0., 1., 0.])
# lengths are shares of the leg length (hip -> knee -> ankle), so a child and an adult are judged alike;
# the cm in the comments are for an 85 cm leg
MIN_DEPTH_LEG = 0.07      # a squat lowers the pelvis by at least this (6 cm)
STAND_BAND_LEG = 0.035    # standing = pelvis within this of its standing height (3 cm)


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _angle(a, b):
    return np.degrees(np.arccos(np.clip((_unit(a) * _unit(b)).sum(-1), -1., 1.)))


def _swap(name):
    for a, b in (('left', 'right'), ('right', 'left')):
        if name.startswith(a):
            return b + name[len(a):]
    return name


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

    def bone(self, name):
        """Unit bone vectors in the body frame [T, 3] (x right, y up, z forward)."""
        a, b = BONE_JOINTS[name]
        v = _unit(self.J[:, self.N[b]] - self.J[:, self.N[a]])
        return np.stack([(v * self.right).sum(1), v[:, 1], (v * self.fwd).sum(1)], 1)

    def bottoms(self):
        rng = float(np.percentile(self.pelvis, 95) - np.percentile(self.pelvis, 5))
        if rng < MIN_DEPTH_LEG * self.leg:
            return []
        pk, _ = find_peaks(-self.pelvis, prominence=max(0.35 * rng, MIN_DEPTH_LEG * self.leg),
                           distance=max(1, int(0.6 * self.fps)))
        return [int(p) for p in pk]

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

    def standing(self, swap=False):
        return self.pose(np.where(self.pelvis >= self.stand_h - STAND_BAND_LEG * self.leg)[0], swap)


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


def _moves(body, r):
    """The range r (s) holds a repetition's worth of pelvis movement (not just a hold)."""
    f0, f1 = (min(max(int(round(t * body.fps)), 0), len(body.pelvis) - 1) for t in r)
    bot = body.bottoms()
    depth = body.stand_h - float(np.median(body.pelvis[bot])) if bot else 0.
    return depth > 0 and float(np.ptp(body.pelvis[f0:f1 + 1])) >= 0.5 * depth


def pose_metrics(user, coach, alignment, segments, mirrored):
    """-> AnalysisResult.poseMetrics. user / coach: Body; alignment: [[user j, coach i]].

    Pairing follows motion_scoring's segments so that poseMetrics and the segments tell the same story:
    a bottom inside an extra (user) / missing (coach) segment that holds a whole movement is unpaired;
    otherwise the user bottom is mapped to coach time through its match / substitute segment (linear
    within the segment), else through the frame alignment, and taken by the nearest free coach bottom."""
    al = np.asarray(alignment, int).reshape(-1, 2)
    ub, cb = user.bottoms(), coach.bottoms()

    def extra(j):
        s = _segment_near(segments, j / user.fps, 'studentRange', ('extra',), tol=0.)
        return s is not None and _moves(user, s['studentRange'])

    def missing(i):
        s = _segment_near(segments, i / coach.fps, 'coachRange', ('missing',), tol=0.)
        return s is not None and _moves(coach, s['coachRange'])

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
    su, sc = user.standing(), coach.standing(swap=mirrored)
    th = dict(sameAngleDeg=config.POSE_SAME_DEG, sameHeightCm=round(config.POSE_SAME_LEG * user.leg, 1),
              sameKneeWidthRatio=config.POSE_SAME_RATIO)
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


def signed_deviations(segments, user, coach, alignment, mirrored):
    """Add direction / components to each segment deviation (in place). Bones are named on the user's side."""
    al = np.asarray(alignment, int).reshape(-1, 2)
    cache = {}
    def bones(body, name, key):
        if (key, name) not in cache:
            cache[key, name] = body.bone(name)
        return cache[key, name]
    for s in segments:
        if not s['deviations'] or 'coachRange' not in s or not len(al):
            continue
        i0, i1 = (int(round(t * coach.fps)) for t in s['coachRange'])
        sel = al[(al[:, 1] >= i0) & (al[:, 1] <= i1)]
        if not len(sel):
            continue
        for d in s['deviations']:
            name = d['bone']
            if name not in BONE_JOINTS:
                continue
            ub = bones(user, name, 'u')[sel[:, 0]]
            cbn = _swap(name) if mirrored else name
            cbv = bones(coach, cbn, 'c')[sel[:, 1]].copy()
            if mirrored:
                cbv[:, 0] *= -1
            m = (ub - cbv).mean(0)
            norm = float(np.linalg.norm(m))
            if norm < 1e-6:
                continue
            comp = m / norm * float(d['angleDiffDeg'])
            lat = comp[0] if name.startswith('right') else -comp[0]      # + = away from the body midline
            c = dict(up=round(float(comp[1]), 1), forward=round(float(comp[2]), 1))
            if name.startswith(('left', 'right')):
                c['outward'] = round(float(lat), 1)
            else:
                c['right'] = round(float(comp[0]), 1)
            axis, val = max(c.items(), key=lambda kv: abs(kv[1]))
            d['direction'] = {('up', True): 'up', ('up', False): 'down', ('forward', True): 'forward',
                              ('forward', False): 'backward', ('outward', True): 'outward',
                              ('outward', False): 'inward', ('right', True): 'right',
                              ('right', False): 'left'}[axis, val > 0]
            d['components'] = c
