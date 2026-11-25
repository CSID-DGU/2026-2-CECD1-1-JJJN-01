"""Bench press: what the app knows about this exercise (exercises/__init__.py says what a module holds).

The lifter lies on the back, so what a squat and a deadlift are read from does not move: the pelvis stays
where it is and the trunk does not lean. A bench press is read from the height of the hands above the
shoulders (cm, mean of left and right):
  the chest pose   the hands lowest, the bar at the chest: where a repetition turns (its key pose)
  the lockout      the arms straight, the hands highest: where a repetition starts and ends (the rest pose)
Two more things follow from lying down.
  directions   The scorer's "forward" is the way the toes of a standing person point; lying on the back that
               is towards the feet, and "backward" towards the head. On the result screen "뒤로" would be
               read wrongly, so the response leaves those two out (HIDDEN_DIRECTIONS: the screen then says
               how large the difference is, not which way), and the LLM is told every direction in the
               words of someone lying down (AXES_KO, DIR_KO).
  values       poseMetrics of the response holds the knees, the hips and the trunk, which say nothing about
               a bench press. The response keeps that shape (the frontend's type); the LLM gets the arms
               instead (llm_metrics).
Lengths are shares of the arm length (shoulder -> elbow -> wrist); the cm in the comments are for a 54 cm arm.
"""
import numpy as np
from scipy.signal import find_peaks

from ..pose_metrics import Body

# the bones a bench press is judged by (motion_scoring names)
MAIN_BONES = ('l_upper_arm', 'r_upper_arm', 'l_forearm', 'r_forearm')

MIN_PRESS_ARM = 0.2        # a press moves the hands by at least this (11 cm)
LOCKOUT_BAND_ARM = 0.06    # the lockout = the hands within this of their highest (3 cm)
CHEST_BAND_ARM = 0.04      # still at the chest = the hands within this (or 5 % of their range) of their lowest there
CHEST_HOLD_S = 0.4         # the bar is held at the chest if the hands stay there this long; a press that only
                           # touches the chest stays about 0.3 s within that band of its turning point
BEFORE_PRESS_S = 0.1       # the key pose of a held chest pose: this long before the hands leave it

HIDDEN_DIRECTIONS = ('forward', 'backward')         # not named in deviations[].direction (see above)
# the components of a deviation, and a direction, in the words of someone lying on the back
AXES_KO = (('up', '천장 쪽으로', '바닥 쪽으로'), ('forward', '발 쪽으로', '머리 쪽으로'),
           ('outward', '바깥쪽으로', '안쪽으로'), ('right', '오른쪽으로', '왼쪽으로'))
DIR_KO = dict(up='천장 쪽으로', down='바닥 쪽으로', forward='발 쪽으로', backward='머리 쪽으로')

SIDES = ('left', 'right')
UP = np.array([0., 1., 0.])


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _angle(a, b):
    return np.degrees(np.arccos(np.clip((_unit(a) * _unit(b)).sum(-1), -1., 1.)))


def arms(body):
    """Per-frame values of the arms of a pose_metrics.Body (joints cm, Y up):
      elbow_<side>   angle between the upper arm and the forearm, 0 = straight arm
      flare_<side>   seen from above, the angle between the upper arm and the line from the head to the feet:
                     0 = the elbow at the side of the body, 90 = straight out at shoulder height. nan while
                     the upper arm stands upright (less than 30 % of it shows from above)
      tilt_<side>    angle of the forearm from the vertical, 0 = the wrist straight above the elbow
      hand_<side>    height of the wrist above its shoulder (cm); hand = the mean of the two
      arm            arm length (cm), one number for the clip"""
    if getattr(body, '_arms', None) is None:
        j = lambda n: body.J[:, body.N[n]]
        feet = j('pelvis') - j('neck')
        feet = feet - (feet @ UP)[:, None] * UP                   # the long axis of the body, seen from above
        a, length = {}, []
        for s in SIDES:
            upper, fore = j(f'{s}_elbow') - j(f'{s}_shoulder'), j(f'{s}_wrist') - j(f'{s}_elbow')
            flat = upper - (upper @ UP)[:, None] * UP
            shows = np.linalg.norm(flat, axis=1) > 0.3 * np.linalg.norm(upper, axis=1)
            a[f'elbow_{s}'] = _angle(upper, fore)
            a[f'flare_{s}'] = np.where(shows, _angle(flat, feet), np.nan)
            a[f'tilt_{s}'] = _angle(fore, UP)
            a[f'hand_{s}'] = j(f'{s}_wrist')[:, 1] - j(f'{s}_shoulder')[:, 1]
            length.append(np.linalg.norm(upper, axis=1) + np.linalg.norm(fore, axis=1))
        a['hand'] = (a['hand_left'] + a['hand_right']) / 2
        a['arm'] = float(np.median(sum(length) / 2))
        body._arms = a
    return body._arms


# ---------------------------------------------------------------- segment names
def labeller(joints, names, fps):
    """-> label(t0, t1, first, last): the Korean name of the segment [t0, t1] s of this clip (joints (T, 22, 3)
    cm, Y up), from the height of the hands over it; first / last = the first / last segment of the result."""
    hand = arms(Body(joints, names, fps))['hand']
    return lambda t0, t1, first, last: label(hand, t0, t1, fps, first, last)


def label(hand, t0, t1, fps, first, last):
    """Korean label of a bench-press segment from the hand height (cm, per frame) over [t0, t1] s."""
    n = len(hand)
    f0, f1 = (min(max(int(round(t * fps)), 0), n - 1) for t in (t0, t1))
    R = max(float(hand.max() - hand.min()), 1e-6)
    seg = hand[f0:f1 + 1]
    d = float(hand[f1] - hand[f0])
    if seg.min() < min(hand[f0], hand[f1]) - 0.25 * R and abs(d) < 0.25 * R:
        return '하강 · 프레스'
    if d < -0.25 * R:
        return '하강'
    if d > 0.25 * R:
        return '프레스'
    if float(seg.mean() - hand.min()) < 0.35 * R:
        return '가슴 터치'
    if first:
        return '시작 자세'
    if last:
        return '마무리 자세'
    return '락아웃'


# ---------------------------------------------------------------- key poses (body: pose_metrics.Body)
def key_frames(body):
    """The frame of every repetition's key pose, the bar at the chest: minima of the hand height with a
    prominence of at least 35 % of its range (and 20 % of the arm), at least 0.6 s apart. Where the bar is
    held at the chest, it is the end of the hold, where the press starts."""
    a = arms(body)
    hand = a['hand']
    rng = float(np.percentile(hand, 95) - np.percentile(hand, 5))
    if rng < MIN_PRESS_ARM * a['arm']:
        return []
    pk, _ = find_peaks(-hand, prominence=max(0.35 * rng, MIN_PRESS_ARM * a['arm']), distance=max(1, int(0.6 * body.fps)))
    band = max(CHEST_BAND_ARM * a['arm'], 0.05 * rng)
    out = []
    for p in pk:
        here = hand <= hand[p] + band
        lo = hi = int(p)                                                   # how long the hands stay at the chest
        while lo > 0 and here[lo - 1]:
            lo -= 1
        while hi + 1 < len(hand) and here[hi + 1]:
            hi += 1
        held = (hi - lo) / body.fps >= CHEST_HOLD_S
        out.append(max(lo, hi - int(round(BEFORE_PRESS_S * body.fps))) if held else int(p))
    return out


def rest_frames(body):
    """The frames of the pose a bench press rests in, the lockout: the hands within 6 % of the arm of their highest."""
    a = arms(body)
    return np.where(a['hand'] >= np.percentile(a['hand'], 95) - LOCKOUT_BAND_ARM * a['arm'])[0]


def moves(body, r, key=None):
    """The range r (s) holds a repetition's worth of hand movement (not just a hold). The chest pose (frame
    `key`) is in the middle of a repetition, so the movement on either side of it counts."""
    hand = arms(body)['hand']
    f0, f1 = (min(max(int(round(t * body.fps)), 0), len(hand) - 1) for t in r)
    keys = key_frames(body)
    depth = float(np.percentile(hand, 95) - np.median(hand[keys])) if keys else 0.
    return depth > 0 and float(np.ptp(hand[f0:f1 + 1])) >= 0.5 * depth


# ---------------------------------------------------------------- the values the LLM gets (not in the response)
ARM_VALUES = (('elbowFlexDeg', 0), ('elbowFlareDeg', 0), ('forearmTiltDeg', 0), ('handHeightCm', 1), ('handLevelCm', 1))


def _arm_pose(body, frames, swap=False):
    """The arm values at a frame or over frames (medians); swap = mirror left / right. A value that most of
    the frames do not have is None (the flare of the elbows while the arms are upright)."""
    a = arms(body)
    v = {k: np.atleast_1d(x[frames]) for k, x in a.items() if k != 'arm'}
    med = lambda x: None if np.isnan(x).mean() > 0.5 else float(np.nanmedian(x))
    side = lambda k, s: med(v[f'{k}_{SIDES[1 - SIDES.index(s)] if swap else s}'])
    lr = lambda k: {s: None if side(k, s) is None else round(side(k, s)) for s in SIDES}
    level = med(v['hand_left'] - v['hand_right'])
    return dict(elbowFlexDeg=lr('elbow'), elbowFlareDeg=lr('flare'), forearmTiltDeg=lr('tilt'),
                handHeightCm=round(med(v['hand']), 1), handLevelCm=round(-level if swap else level, 1))


def _arm_diff(u, c):
    """user - coach for every arm value (None where one of the two could not be measured)."""
    minus = lambda x, y, nd: None if x is None or y is None else (round(x - y, nd) if nd else x - y)
    return {k: {s: minus(u[k][s], c[k][s], nd) for s in SIDES} if isinstance(u[k], dict) else minus(u[k], c[k], nd)
            for k, nd in ARM_VALUES}


def llm_metrics(user, coach, pm, mirrored):
    """What the LLM is given as poseMetrics for a bench press: the same repetitions as the response's
    poseMetrics pm (their order, their pairing, the times of their key poses) with the values of the arms.
      lockout        the arms straight (rest_frames)
      reps[].chest   the bar at the chest (key_frames): user, coach, diff = user - coach
      chestSummary   [min, max] over the paired repetitions (elbow values: mean of left and right)
    If the scorer matched the mirrored user, the coach's left / right are swapped onto the user's sides."""
    at = lambda body, sec, swap: None if sec is None else _arm_pose(body, body.window(int(round(sec * body.fps))), swap)
    reps = []
    for r in pm['reps']:
        u, c = at(user, r['userSec'], False), at(coach, r['coachSec'], mirrored)
        rep = dict(index=r['index'], userSec=r['userSec'], coachSec=r['coachSec'], segmentId=r['segmentId'],
                   chest=dict(user=u, coach=c))
        if u and c:
            rep['chest']['diff'] = _arm_diff(u, c)
        reps.append(rep)
    lu, lc = _arm_pose(user, rest_frames(user)), _arm_pose(coach, rest_frames(coach), mirrored)
    th = pm['thresholds']
    return dict(armLengthCm=round(arms(user)['arm'], 1),
                thresholds=dict(sameAngleDeg=th['sameAngleDeg'], sameHeightCm=th['sameHeightCm']),
                repCount=pm['repCount'], lockout=dict(user=lu, coach=lc, diff=_arm_diff(lu, lc)), reps=reps,
                chestSummary=_arm_summary([r['chest'] for r in reps if r['chest'].get('diff')]))


def _arm_summary(chests):
    """[min, max] over the paired repetitions of every arm value (left / right values: their mean)."""
    if not chests:
        return None
    def val(p, k):
        if not isinstance(p[k], dict):
            return p[k]
        both = [x for x in p[k].values() if x is not None]
        return sum(both) / len(both) if both else None
    out = {}
    for k, nd in ARM_VALUES:
        out[k] = {}
        for who in ('user', 'coach', 'diff'):
            v = [x for x in (val(c[who], k) for c in chests) if x is not None]
            out[k][who] = None if not v else [int(round(min(v))), int(round(max(v)))] if nd == 0 else [round(min(v), nd), round(max(v), nd)]
    return dict(reps=len(chests), **out)


# ---------------------------------------------------------------- what the LLM is told (llm/feedback.py system_prompt)
PROMPT = dict(
    missing_example='반복 사이에 팔을 편 채 쉬기',
    extra_example='가슴에서 멈추기',
    rest_where='팔을 편 채(또는 가슴에서)',
    rest_labels='락아웃, 가슴 터치',
    still='가만히 멈춰 있던',
    deviations='하강 · 프레스 구간이면 보통 바를 가장 낮게 내린 순간의 값이다. 종목에서 중요한 부위(벤치프레스: 위팔, 아래팔)가 먼저 나온다. 누워서 하는 운동이라 방향은 천장 쪽 / 바닥 쪽, 머리 쪽 / 발 쪽, 바깥쪽 / 안쪽으로 나온다.',
    pose_metrics="""- poseMetrics: 팔 자세 측정. 누워서 하는 운동이라 다리와 상체 값은 없다.
  - armLengthCm: 사용자 팔 길이(어깨에서 손목까지, cm). thresholds: 이보다 작은 차이는 비슷한 것으로 보는 기준.
  - repCount: 바를 가슴까지 내렸다 올린 횟수(user, coach).
  - chestSummary: 코치와 짝이 맞은 반복들의 바를 가장 낮게 내린 순간 값의 범위 [최소, 최대] (user, coach, diff). 좌우가 있는 값은 좌우 평균. 여러 반복을 묶어 말할 때는 이 범위를 그대로 쓴다.
  - lockout: 팔을 다 편 자세의 값. reps: 반복마다 바를 가장 낮게 내린 순간(chest)의 값. user, coach, diff(= 사용자 − 코치). user 나 coach 가 null 이면 그 반복은 한쪽에만 있다. segmentId 는 그 순간이 속한 구간.
  - elbowFlexDeg: 팔꿈치 굽힘. 0 이면 팔을 편 상태, 클수록 많이 굽혔다. left / right.
  - elbowFlareDeg: 위에서 봤을 때 위팔이 몸통에서 벌어진 각도. 0 이면 팔꿈치가 옆구리 쪽, 90 이면 어깨 높이에서 옆으로 완전히 벌린 것이다. left / right. 팔을 세우고 있을 때는 잴 수 없어 null 이다.
  - forearmTiltDeg: 아래팔이 수직에서 기운 각도. 0 이면 손목이 팔꿈치 바로 위에 있다. left / right.
  - handHeightCm: 손 높이 − 어깨 높이(cm, 좌우 평균). 작을수록 바를 깊이 내렸다.
  - handLevelCm: 왼손 높이 − 오른손 높이(cm). 0 에서 멀수록 바가 한쪽으로 기울었다.""",
    rules="""- 벤치프레스는 바를 내린 깊이(handHeightCm, elbowFlexDeg), 팔꿈치를 벌린 정도(elbowFlareDeg), 아래팔이 수직인지(forearmTiltDeg), 좌우가 고른지(handLevelCm, 좌우 값의 차이), 속도와 멈춤을 먼저 본다. 다리나 머리 위치 차이는 마지막에 한 번만 짧게 말한다.
- poseMetrics.thresholds 보다 작은 차이(각도 sameAngleDeg 도, 높이 sameHeightCm cm)는 비슷한 것으로 보고 말하지 않는다.
- 자세 비교는 reps(바를 가장 낮게 내린 순간)를 먼저 본다. lockout(팔을 편 자세)은 차이가 클 때만 말하고, 팔을 편 자세라고 밝힌다.
- 깊이는 handHeightCm 과 elbowFlexDeg 로만 판단한다(handHeightCm 이 작을수록, elbowFlexDeg 가 클수록 깊다). 팔꿈치를 벌린 정도를 깊이로 바꿔 말하지 않는다.
- 각 항목에 근거 수치를 넣는다. 예: "바를 가장 낮게 내렸을 때 팔꿈치를 72~78도 벌렸고 코치는 50~55도예요", "내리는 데 0.80초 걸렸고 코치는 1.20초였어요". 각도는 정수, cm 는 소수 첫째 자리까지 말한다.""",
)
