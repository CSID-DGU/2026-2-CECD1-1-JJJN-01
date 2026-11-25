"""Squat: what the app knows about this exercise (exercises/__init__.py says what a module holds).

Lengths are shares of the leg length (hip -> knee -> ankle), so a child and an adult are judged alike;
the cm in the comments are for an 85 cm leg.
"""
import numpy as np
from scipy.signal import find_peaks

# the bones a squat is judged by (motion_scoring names)
MAIN_BONES = ('trunk', 'l_thigh', 'r_thigh', 'l_shank', 'r_shank')

MIN_DEPTH_LEG = 0.07      # a squat lowers the pelvis by at least this (6 cm)
STAND_BAND_LEG = 0.035    # standing = pelvis within this of its standing height (3 cm)


# ---------------------------------------------------------------- segment names
def labeller(joints, names, fps):
    """-> label(t0, t1, first, last): the Korean name of the segment [t0, t1] s of this clip (joints (T, 22, 3)
    cm, Y up), from the pelvis height over it; first / last = the first / last segment of the result."""
    y = joints[:, names.index('pelvis'), 1] / 100.
    return lambda t0, t1, first, last: label(y, t0, t1, fps, first, last)


def label(y, t0, t1, fps, first, last):
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


# ---------------------------------------------------------------- key poses (body: pose_metrics.Body)
def key_frames(body):
    """The frame of every repetition's key pose, the bottom: pelvis-height minima with a prominence of at least
    35 % of the pelvis range (and 7 % of the leg), at least 0.6 s apart."""
    rng = float(np.percentile(body.pelvis, 95) - np.percentile(body.pelvis, 5))
    if rng < MIN_DEPTH_LEG * body.leg:
        return []
    pk, _ = find_peaks(-body.pelvis, prominence=max(0.35 * rng, MIN_DEPTH_LEG * body.leg),
                       distance=max(1, int(0.6 * body.fps)))
    return [int(p) for p in pk]


def rest_frames(body):
    """The frames of the pose a squat rests in, standing: the pelvis within 3.5 % of the leg of its standing height."""
    return np.where(body.pelvis >= body.stand_h - STAND_BAND_LEG * body.leg)[0]


def moves(body, r, key=None):
    """The range r (s) holds a repetition's worth of pelvis movement (not just a hold). The bottom (frame `key`)
    is in the middle of a squat, so the movement on either side of it counts."""
    f0, f1 = (min(max(int(round(t * body.fps)), 0), len(body.pelvis) - 1) for t in r)
    bot = key_frames(body)
    depth = body.stand_h - float(np.median(body.pelvis[bot])) if bot else 0.
    return depth > 0 and float(np.ptp(body.pelvis[f0:f1 + 1])) >= 0.5 * depth


# ---------------------------------------------------------------- what the LLM is told (llm/feedback.py system_prompt)
PROMPT = dict(
    missing_example='반복 사이에 서서 쉬기',
    extra_example='바닥에서 멈추기',
    rest_where='서서(또는 앉아서)',
    rest_labels='선 자세 유지, 바닥 유지',
    still='가만히 서 있던',
    deviations='하강 · 상승 구간이면 보통 가장 낮게 앉은 순간의 값이다. 종목에서 중요한 부위(스쿼트: 몸통, 허벅지, 정강이)가 먼저 나온다.',
    pose_metrics="""- poseMetrics: 관절 각도 측정.
  - legLengthCm: 사용자 다리 길이(cm). thresholds: 이보다 작은 차이는 비슷한 것으로 보는 기준.
  - repCount: 앉았다 일어난 횟수(user, coach).
  - bottomSummary: 코치와 짝이 맞은 반복들의 가장 낮게 앉은 순간 값의 범위 [최소, 최대] (user, coach, diff). 무릎과 엉덩이는 좌우 평균. 여러 반복을 묶어 말할 때는 이 범위를 그대로 쓴다.
  - standing: 선 자세의 값. reps: 반복마다 가장 낮게 앉은 순간의 값. user, coach, diff(= 사용자 − 코치). user 나 coach 가 null 이면 그 반복은 한쪽에만 있다. segmentId 는 그 순간이 속한 구간.
  - kneeFlexDeg: 무릎 굽힘. 0 이면 다리를 편 상태, 클수록 많이 굽혔다. left / right.
  - hipFlexDeg: 엉덩이 굽힘(상체와 허벅지가 접힌 정도). 0 이면 똑바로 선 상태. left / right.
  - trunkLeanDeg: 상체가 앞으로 숙여진 각도. 0 이면 수직.
  - hipKneeCm: 엉덩이 높이 − 무릎 높이(cm). 작을수록 깊게 앉았고, 0 이하면 허벅지가 수평보다 아래로 내려갔다.
  - pelvisDropCm: 선 자세보다 골반이 내려간 거리(cm).
  - kneeWidthRatio: 무릎 사이 거리 ÷ 발목 사이 거리. 1보다 많이 작으면 무릎이 안쪽으로 모였다.""",
    rules="""- 스쿼트는 깊이(hipKneeCm, pelvisDropCm, kneeFlexDeg), 상체 숙임, 엉덩이 굽힘, 무릎 간격, 속도와 멈춤을 먼저 본다. 팔이나 손 위치 차이는 마지막에 한 번만 짧게 말한다.
- poseMetrics.thresholds 보다 작은 차이(각도 sameAngleDeg 도, 높이 sameHeightCm cm, kneeWidthRatio sameKneeWidthRatio)는 비슷한 것으로 보고 말하지 않는다. 높이 기준은 사용자 다리 길이(legLengthCm)에 맞춘 값이다.
- 자세 비교는 reps(가장 낮게 앉은 순간)를 먼저 본다. standing(선 자세)은 차이가 클 때만 말하고, 선 자세라고 밝힌다.
- 깊이는 hipKneeCm 과 pelvisDropCm 으로만 판단한다(값이 작을수록, 내려간 거리가 클수록 깊다). 상체 숙임이나 엉덩이 굽힘을 깊이로 바꿔 말하지 않는다.
- 각 항목에 근거 수치를 넣는다. 예: "가장 낮게 앉았을 때 상체를 31~38도 숙였고 코치는 41~45도예요", "내려가는 데 0.93초 걸렸고 코치는 1.23초였어요". 각도는 정수, cm 는 소수 첫째 자리까지 말한다.""",
)
