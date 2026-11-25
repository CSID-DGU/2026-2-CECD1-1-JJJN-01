"""Deadlift: what the app knows about this exercise (exercises/__init__.py says what a module holds).

A deadlift is read from the forward lean of the trunk (pose_metrics.Body: trunkLeanDeg, 0 = upright), not from
the pelvis height as a squat is: every way of doing it leans the trunk a long way, while the pelvis hardly
comes down in some (stiff legs). Two poses matter:
  the floor pose   the trunk leaned furthest, hands at the bar: where a lift starts
  the lockout      standing upright with the bar: where a lift ends
A clip may start standing (walk up, bend down, lift) or already in the floor pose, and may hold the bar at the
floor between repetitions or only touch it. A repetition is counted by its lift, so its key pose is the floor
pose the lift starts from: three repetitions have three of them in either kind of clip, while the floor pose
a clip only ends in starts no lift and is not one.
"""
import numpy as np
from scipy.signal import find_peaks

from ..pose_metrics import Body

# the bones a deadlift is judged by (motion_scoring names)
MAIN_BONES = ('trunk', 'l_thigh', 'r_thigh', 'l_shank', 'r_shank')

MIN_LEAN_DEG = 20.        # a deadlift leans the trunk at least this much further than standing does
UPRIGHT_BAND_DEG = 5.     # the lockout = the trunk within this of its most upright
FLOOR_BAND_DEG = 3.       # still in the floor pose = the trunk and the knees within this (or 5 % of their range)
                          # of where they are when the trunk is lowest
FLOOR_HOLD_S = 0.35       # the floor pose is held (the bar rests) if the body stays in it this long; a lift
                          # that only touches the floor stays about 0.25 s within that band of its turning point
BEFORE_LIFT_S = 0.1       # the key pose of a held floor pose: this long before the body leaves it (the values
                          # of a pose are medians over +- 0.1 s, which then all lie in the hold)
EDGE_HOLD_S = 0.25        # a clip starts in the floor pose if it holds it this long (else it is cut mid-lift)


# ---------------------------------------------------------------- segment names
def labeller(joints, names, fps):
    """-> label(t0, t1, first, last): the Korean name of the segment [t0, t1] s of this clip (joints (T, 22, 3)
    cm, Y up), from the lean of the trunk over it; first / last = the first / last segment of the result."""
    lean = Body(joints, names, fps).m['lean']
    return lambda t0, t1, first, last: label(lean, t0, t1, fps, first, last)


def label(lean, t0, t1, fps, first, last):
    """Korean label of a deadlift segment from the trunk lean (deg, per frame) over [t0, t1] s."""
    n = len(lean)
    f0, f1 = (min(max(int(round(t * fps)), 0), n - 1) for t in (t0, t1))
    R = max(float(lean.max() - lean.min()), 1e-6)
    seg = lean[f0:f1 + 1]
    d = float(lean[f1] - lean[f0])                     # + = leaning further = going down
    if abs(d) < 0.25 * R:                              # down and up again (or up and down) within the segment
        if seg.max() > max(lean[f0], lean[f1]) + 0.25 * R:
            return '내리기 · 리프트'
        if seg.min() < min(lean[f0], lean[f1]) - 0.25 * R:
            return '리프트 · 내리기'
    if d > 0.25 * R:                                   # going down: to the bar the first time, with it after a lift
        return '내리기' if float(lean[:f0 + 1].max() - lean.min()) > 0.65 * R else '셋업'
    if d < -0.25 * R:
        return '리프트'
    if float(lean.max() - seg.mean()) < 0.35 * R:      # held in the floor pose
        return '마무리 자세' if last else '셋업'
    if first:
        return '시작 자세'
    if last:
        return '마무리 자세'
    return '락아웃'


# ---------------------------------------------------------------- key poses (body: pose_metrics.Body)
def key_frames(body):
    """The frame of every repetition's key pose, the floor pose a lift starts from: maxima of the trunk lean
    with a prominence of at least 35 % of its range (and 20 deg), at least 0.6 s apart. Where the floor pose
    is held (the bar rests on the floor), it is the end of the hold, where the lift starts: the knees
    straighten before the trunk comes up, so the hold ends when either leaves the pose. A clip that starts in
    the floor pose and holds it has one there; the pose a clip ends in has no lift after it."""
    lean = body.m['lean']
    rng = float(np.percentile(lean, 95) - np.percentile(lean, 5))
    if rng < MIN_LEAN_DEG:
        return []
    prominence = max(0.35 * rng, MIN_LEAN_DEG)
    # a maximum at the first frame is no peak for find_peaks: one lower sample in front of the clip makes it one
    pk, _ = find_peaks(np.r_[lean.min() - prominence, lean], prominence=prominence, distance=max(1, int(0.6 * body.fps)))
    hold = max(1, int(round(EDGE_HOLD_S * body.fps)))
    knee = (body.m['knee_left'] + body.m['knee_right']) / 2
    band = max(FLOOR_BAND_DEG, 0.05 * rng)
    band_knee = max(FLOOR_BAND_DEG, 0.05 * float(np.percentile(knee, 95) - np.percentile(knee, 5)))
    out = []
    for p in pk - 1:
        if p < hold and float(np.ptp(lean[:hold + 1])) > 0.15 * rng:      # the clip starts while already moving
            continue
        here = (lean >= lean[p] - band) & (np.abs(knee - knee[p]) <= band_knee)
        a = b = int(p)                                                    # how long the body stays in this pose
        while a > 0 and here[a - 1]:
            a -= 1
        while b + 1 < len(lean) and here[b + 1]:
            b += 1
        held = (b - a) / body.fps >= FLOOR_HOLD_S
        out.append(max(a, b - int(round(BEFORE_LIFT_S * body.fps))) if held else int(p))
    return out


def rest_frames(body):
    """The frames of the pose a deadlift rests in, the lockout: the trunk within 5 deg of its most upright."""
    lean = body.m['lean']
    return np.where(lean <= np.percentile(lean, 5) + UPRIGHT_BAND_DEG)[0]


def moves(body, r, key):
    """The range r (s) holds the lift that starts at the frame `key`: from there the trunk comes up by at least
    half of a repetition's depth before the range ends. Not so for the floor pose alone, nor for the way down
    to it, which ends at the key pose of the lift that follows."""
    lean = body.m['lean']
    f0, f1 = (min(max(int(round(t * body.fps)), 0), len(lean) - 1) for t in r)
    keys = key_frames(body)
    depth = float(np.median(lean[keys]) - np.percentile(lean, 5)) if keys else 0.
    start = min(max(int(key), f0), f1)
    return depth > 0 and float(lean[start] - lean[start:f1 + 1].min()) >= 0.5 * depth


# ---------------------------------------------------------------- what the LLM is told (llm/feedback.py system_prompt)
PROMPT = dict(
    missing_example='반복 사이에 바닥에 내려놓고 쉬기',
    extra_example='다 일어선 자세에서 오래 멈추기',
    rest_where='다 일어서서(또는 바닥에 내려놓고)',
    rest_labels='락아웃, 셋업',
    still='가만히 멈춰 있던',
    deviations='리프트 · 내리기 구간이면 보통 상체를 가장 깊이 숙인 순간의 값이다. 종목에서 중요한 부위(데드리프트: 몸통, 허벅지, 정강이)가 먼저 나온다.',
    pose_metrics="""- poseMetrics: 관절 각도 측정. 필드 이름은 스쿼트 기준이라 bottom 은 바닥 자세, standing 은 다 일어선 자세(락아웃)를 뜻한다.
  - legLengthCm: 사용자 다리 길이(cm). thresholds: 이보다 작은 차이는 비슷한 것으로 보는 기준.
  - repCount: 바닥에서 들어 올린 횟수(user, coach).
  - bottomSummary: 코치와 짝이 맞은 반복들의 바닥 자세(들어 올리기 시작하는 자세) 값의 범위 [최소, 최대] (user, coach, diff). 무릎과 엉덩이는 좌우 평균. 여러 반복을 묶어 말할 때는 이 범위를 그대로 쓴다.
  - standing: 다 일어선 자세의 값. reps: 반복마다 바닥 자세의 값. user, coach, diff(= 사용자 − 코치). user 나 coach 가 null 이면 그 반복은 한쪽에만 있다. segmentId 는 그 순간이 속한 구간.
  - kneeFlexDeg: 무릎 굽힘. 0 이면 다리를 편 상태, 클수록 많이 굽혔다. left / right.
  - hipFlexDeg: 엉덩이 굽힘(상체와 허벅지가 접힌 정도). 0 이면 똑바로 선 상태. left / right.
  - trunkLeanDeg: 상체가 앞으로 숙여진 각도. 0 이면 수직.
  - hipKneeCm: 엉덩이 높이 − 무릎 높이(cm). 작을수록 엉덩이를 낮춰 앉듯이 잡았다.
  - pelvisDropCm: 다 일어선 자세보다 골반이 내려간 거리(cm).
  - kneeWidthRatio: 무릎 사이 거리 ÷ 발목 사이 거리. 1보다 많이 작으면 무릎이 안쪽으로 모였다.""",
    rules="""- 데드리프트는 바닥 자세(상체 숙임, 엉덩이 높이, 무릎 굽힘), 다 일어섰을 때 엉덩이와 무릎을 끝까지 폈는지(standing 의 hipFlexDeg, kneeFlexDeg), 속도와 멈춤을 먼저 본다. 팔이나 머리 위치 차이는 마지막에 한 번만 짧게 말한다.
- poseMetrics.thresholds 보다 작은 차이(각도 sameAngleDeg 도, 높이 sameHeightCm cm, kneeWidthRatio sameKneeWidthRatio)는 비슷한 것으로 보고 말하지 않는다. 높이 기준은 사용자 다리 길이(legLengthCm)에 맞춘 값이다.
- 자세 비교는 reps(바닥 자세)를 먼저 본다. standing(다 일어선 자세)은 차이가 클 때만 말하고, 다 일어선 자세라고 밝힌다.
- 엉덩이 높이는 hipKneeCm 과 pelvisDropCm 으로만 판단한다(hipKneeCm 이 작을수록, pelvisDropCm 이 클수록 엉덩이가 낮다). 상체 숙임을 엉덩이 높이로 바꿔 말하지 않는다.
- 각 항목에 근거 수치를 넣는다. 예: "바닥에서 잡았을 때 상체를 52~58도 숙였고 코치는 44~47도예요", "들어 올리는 데 1.40초 걸렸고 코치는 1.05초였어요". 각도는 정수, cm 는 소수 첫째 자리까지 말한다.""",
)
