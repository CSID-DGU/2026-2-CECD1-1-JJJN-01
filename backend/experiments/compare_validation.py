"""Validation of the comparison stage (motion_scoring + pose_metrics) with known, injected differences.

Base: a real skeleton from the pipeline (the retargeted coach of a job, SMPL-22, cm, 30 fps). The "user"
is a copy with ONE exactly known change: a time warp, an inserted hold, a removed rest, an extra or a
missing repetition, or a rigid joint rotation (bone lengths kept). The truth is what we injected, so it is
exact. The app's own result_builder.build_result runs on (coach = base, user = changed copy); only the BVH
reader is swapped for the arrays, no app code is changed.

Each case checks
  found     the injected difference is reported at the right place with about the right size
  clean     nothing is reported anywhere else (no false structure / timing / pose labels, no pose diffs
            over the feedback thresholds config.POSE_SAME_*)

Run from backend/:  ../.venv/bin/python -m experiments.compare_validation [--job e80464dd6496]
Outputs: data/experiments/validation/report.md and results.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from app import config
from app import result_builder as rb
import pipeline.bvh_to_npz as b2n                     # importable once result_builder put the repo on sys.path

OUT = config.DATA_DIR / 'experiments' / 'validation'
UP = np.array([0., 1., 0.])
TH_DEG, TH_RATIO = config.POSE_SAME_DEG, config.POSE_SAME_RATIO
TH_CM = None          # config.POSE_SAME_LEG x leg length: read from the first result (poseMetrics.thresholds)
UPPER = ('pelvis', 'left_hip', 'right_hip', 'spine1', 'spine2', 'spine3', 'neck', 'head', 'left_collar',
         'right_collar', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist')
TRUNK_UP = tuple(n for n in UPPER if n not in ('pelvis', 'left_hip', 'right_hip'))


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def rot(p, c, k, th):
    """Rotate points p [n, 3] about the axis k through c by th (rad), Rodrigues."""
    k = k / np.linalg.norm(k)
    v = p - c
    return c + v * np.cos(th) + np.cross(k, v) * np.sin(th) + np.outer(v @ k, k) * (1 - np.cos(th))


class Base:
    def __init__(self, X, names, fps):
        self.X, self.names, self.fps = X.astype(float), list(names), float(fps)
        self.N = {n: i for i, n in enumerate(names)}
        self.y = self.X[:, self.N['pelvis'], 1]
        self.stand = float(np.percentile(self.y, 95))
        self.reps = []
        for f in find_peaks(-self.y, prominence=15)[0]:
            s, e = f, f
            while s > 0 and self.y[s] < self.stand - 2:
                s -= 1
            while e < len(self.y) - 1 and self.y[e] < self.stand - 2:
                e += 1
            self.reps.append(dict(start=int(s), bottom=int(f), end=int(e)))

    def ix(self, *names):
        return [self.N[n] for n in names]

    def right(self, X):
        r = X[:, self.N['right_hip']] - X[:, self.N['left_hip']]
        return unit(r - np.outer(r @ UP, UP))

    def weight(self, rep):
        """0 outside the repetition, depth fraction (1 at the bottom) inside it."""
        r = self.reps[rep]
        w = np.zeros(len(self.y))
        sl = slice(r['start'], r['end'] + 1)
        w[sl] = np.clip((self.stand - self.y[sl]) / (self.stand - self.y[r['bottom']]), 0, 1)
        return w

    def t(self, f):
        return f / self.fps


# ---------------------------------------------------------------- independent truth metrics (simple formulas)
def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1, 1))))


def metrics(B, X, f):
    j = lambda n: X[f, B.N[n]]
    trunk = j('neck') - j('pelvis')
    r = B.right(X[f:f + 1])[0]
    fwd = np.cross(UP, r)
    lean = angle(trunk, UP) * (1 if trunk @ fwd >= 0 else -1)
    out = dict(trunkLeanDeg=lean,
               hipKneeCm=(j('left_hip')[1] + j('right_hip')[1] - j('left_knee')[1] - j('right_knee')[1]) / 2,
               pelvisDropCm=B.stand - j('pelvis')[1],
               kneeWidthRatio=np.linalg.norm(j('left_knee') - j('right_knee')) /
               np.linalg.norm(j('left_ankle') - j('right_ankle')))
    for s in ('left', 'right'):
        thigh, shin = j(f'{s}_knee') - j(f'{s}_hip'), j(f'{s}_ankle') - j(f'{s}_knee')
        out[f'knee_{s}'], out[f'hip_{s}'] = angle(thigh, shin), 180 - angle(trunk, thigh)
    return out


def truth_diff(B, U, fu, fc):
    a, b = metrics(B, U, fu), metrics(B, B.X, fc)
    d = {k: a[k] - b[k] for k in a}
    return dict(kneeFlexDeg=(d['knee_left'] + d['knee_right']) / 2, hipFlexDeg=(d['hip_left'] + d['hip_right']) / 2,
                trunkLeanDeg=d['trunkLeanDeg'], hipKneeCm=d['hipKneeCm'], pelvisDropCm=d['pelvisDropCm'],
                kneeWidthRatio=d['kneeWidthRatio'])


# ---------------------------------------------------------------- injected changes -> (user skeleton, truth)
def resample(X, n):
    src = np.linspace(0, len(X) - 1, n)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, len(X) - 1)
    a = (src - i0)[:, None, None]
    return X[i0] * (1 - a) + X[i1] * a


def splice_xz(parts):
    """Concatenate pieces, shifting each in x / z so that it starts where the previous one ended."""
    out = [parts[0]]
    for p in parts[1:]:
        p = p.copy()
        d = out[-1][-1, 0] - p[0, 0]                     # pelvis of the last frame vs the first frame
        p[..., 0] += d[0]
        p[..., 2] += d[2]
        out.append(p)
    return np.concatenate(out)


def case_control(B):
    return B.X.copy(), dict(kind='control')


def case_noise(B, sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return B.X + rng.normal(0, sigma, B.X.shape), dict(kind='control', note=f'{sigma} cm joint noise')


def case_slow_descent(B, rep=1, factor=1.5):
    r = B.reps[rep]
    a, b = r['start'], r['bottom']
    n = int(round((b - a) * factor)) + 1
    U = np.concatenate([B.X[:a], resample(B.X[a:b + 1], n), B.X[b + 1:]])
    return U, dict(kind='timing', label='slow', ratio=factor, user=(B.t(a), B.t(a + n - 1)), coach=(B.t(a), B.t(b)))


def case_tempo(B, factor=0.8):
    return resample(B.X, int(round(len(B.X) * factor))), dict(kind='tempo', tempo=factor)


def case_hold(B, rep=2, sec=1.0):
    f = B.reps[rep]['bottom']
    n = int(round(sec * B.fps))
    U = np.concatenate([B.X[:f + 1], np.repeat(B.X[f:f + 1], n, 0), B.X[f + 1:]])
    return U, dict(kind='extra_hold', sec=sec, user=(B.t(f), B.t(f + n)))


def case_no_rest(B, rep=1):
    a, b = B.reps[rep]['end'] + 2, B.reps[rep + 1]['start'] - 2       # standing rest between rep and rep+1
    U = splice_xz([B.X[:a], B.X[b:]])
    return U, dict(kind='missing_rest', coach=(B.t(a), B.t(b)), user_at=B.t(a), sec=B.t(b - a))


def coach_pauses(B):
    """Standing pauses of the base as motion_scoring itself detects them: [(i0, i1)] frames (clip edges excluded)."""
    from pipeline.score_user import SCORING_DIR
    if str(SCORING_DIR) not in sys.path:
        sys.path.insert(0, str(SCORING_DIR))
    from dataclasses import replace
    from motion_scoring import Config, track_from_joints
    from motion_scoring.step1_preprocess import Clip
    from motion_scoring.step2_checkpoints import detect_events
    cfg = replace(Config(), auto_tau=True, pose_trim=0.3)
    X = Clip(track_from_joints(B.X, np.arange(len(B.X)) / B.fps, 'y', None, B.names), cfg)
    return [(e.i0, e.i1) for e in detect_events(X, cfg)
            if e.type == 'pause' and B.y[(e.i0 + e.i1) // 2] > B.stand - 8]


def case_short_rest(B, keep, which=2):
    """Keep only `keep` of one standing pause (frames cut from its middle, the body is still there)."""
    i0, i1 = coach_pauses(B)[which]
    n = i1 - i0
    m = int(round(n * (1 - keep)))
    a = i0 + (n - m) // 2
    U = splice_xz([B.X[:a], B.X[a + m:]])
    return U, dict(kind='short_rest', keep=keep, coach=(B.t(i0), B.t(i1)), sec=B.t(n),
                   expect='missing' if keep < config_ratio() else 'held')


def config_ratio():
    from motion_scoring import Config
    return Config().hold_missing_ratio


def case_extra_rep(B, rep=2, at_end=False):
    r, nxt = B.reps[rep], (B.reps[rep + 1]['start'] if rep + 1 < len(B.reps) else len(B.X))
    piece = B.X[r['start']:nxt]                                          # the repetition + its rest
    if at_end:
        r = B.reps[-1]
        U = splice_xz([B.X, B.X[r['start']:]])
        u0 = len(B.X)
        end = len(U) - 1
    else:
        U = splice_xz([B.X[:nxt], piece, B.X[nxt:]])
        u0 = nxt
        end = u0 + len(piece)
    ub = u0 + r['bottom'] - r['start']
    # the original and its copy are the same movement back to back: either one is a correct "extra"
    return U, dict(kind='extra_rep', user=(B.t(r['start']), B.t(end)),
                   user_bottoms=(B.t(r['bottom']), B.t(ub)), reps=(len(B.reps) + 1, len(B.reps)))


def case_missing_rep(B, rep=3):
    a = B.reps[rep]['start']
    b = B.reps[rep + 1]['start'] if rep + 1 < len(B.reps) else len(B.X)
    U = splice_xz([B.X[:a], B.X[b:]])
    return U, dict(kind='missing_rep', coach=(B.t(a), B.t(b)), coach_bottom=B.t(B.reps[rep]['bottom']),
                   reps=(len(B.reps) - 1, len(B.reps)))


def case_trunk(B, rep=2, deg=10.):
    U, w, R = B.X.copy(), B.weight(rep), B.right(B.X)
    idx, p = B.ix(*TRUNK_UP), B.N['pelvis']
    for f in np.nonzero(w)[0]:
        th = np.radians(deg * w[f])
        cand = [rot(U[f, idx], U[f, p], R[f], s * th) for s in (1, -1)]
        fwd = np.cross(UP, R[f])
        nk = TRUNK_UP.index('neck')
        U[f, idx] = max(cand, key=lambda c: (c[nk] - U[f, p]) @ fwd)          # the one leaning forward
    fb = B.reps[rep]['bottom']
    return U, dict(kind='pose', rep=rep, user_bottom=B.t(fb), coach_bottom=B.t(fb),
                   injected=f'trunk +{deg:g} deg forward', truth=truth_diff(B, U, fb, fb))


def case_shallow(B, rep=2, cm=8.):
    """Raise pelvis + upper body by cm at the bottom (ankles / feet fixed), knees re-solved (2-link IK)."""
    U, w = B.X.copy(), B.weight(rep)
    up_idx = B.ix(*UPPER)
    for f in np.nonzero(w)[0]:
        U[f, up_idx, 1] += cm * w[f]
        for s in ('left', 'right'):
            h, k, a = B.N[f'{s}_hip'], B.N[f'{s}_knee'], B.N[f'{s}_ankle']
            L1 = np.linalg.norm(B.X[f, k] - B.X[f, h])
            L2 = np.linalg.norm(B.X[f, a] - B.X[f, k])
            H, A, K = U[f, h], U[f, a], B.X[f, k]
            d = min(np.linalg.norm(A - H), L1 + L2 - 1e-3)
            u = unit(A - H)
            v = unit((K - H) - ((K - H) @ u) * u)
            ca = np.clip((L1 ** 2 + d ** 2 - L2 ** 2) / (2 * L1 * d), -1, 1)
            U[f, k] = H + L1 * (ca * u + np.sqrt(1 - ca ** 2) * v)
    fb = B.reps[rep]['bottom']
    return U, dict(kind='pose', rep=rep, user_bottom=B.t(fb), coach_bottom=B.t(fb),
                   injected=f'pelvis {cm:g} cm higher (shallower)', truth=truth_diff(B, U, fb, fb))


def case_knees_in(B, rep=2, deg=15.):
    U, w, R = B.X.copy(), B.weight(rep), B.right(B.X)
    p = B.N['pelvis']
    for f in np.nonzero(w)[0]:
        for s in ('left', 'right'):
            h, k, a = B.N[f'{s}_hip'], B.N[f'{s}_knee'], B.N[f'{s}_ankle']
            cand = [rot(U[f, [k]], U[f, h], U[f, a] - U[f, h], sg * np.radians(deg * w[f]))[0] for sg in (1, -1)]
            U[f, k] = min(cand, key=lambda c: abs((c - U[f, p]) @ R[f]))      # closer to the midline
    fb = B.reps[rep]['bottom']
    return U, dict(kind='pose', rep=rep, user_bottom=B.t(fb), coach_bottom=B.t(fb),
                   injected=f'knees {deg:g} deg inward', truth=truth_diff(B, U, fb, fb))


def case_arm_up(B, deg=30.):
    U, R = B.X.copy(), B.right(B.X)
    e, wr = B.N['right_elbow'], B.N['right_wrist']
    for f in range(len(U)):
        cand = [rot(U[f, [wr]], U[f, e], R[f], s * np.radians(deg))[0] for s in (1, -1)]
        U[f, wr] = max(cand, key=lambda c: c[1])                               # the one that lifts the wrist
    return U, dict(kind='arm', bone='right_forearm', deg=deg, direction='up')


def case_mirror(B):
    U, R = B.X.copy(), B.right(B.X)
    c = B.X[:, B.N['pelvis']]
    U = U - 2 * np.einsum('tj,t->tj', np.einsum('tjk,tk->tj', U - c[:, None], R), np.ones(len(U)))[..., None] * R[:, None]
    swap = [B.N[rb.swap_side(n)] if rb.swap_side(n) in B.N else i for i, n in enumerate(B.names)]
    return U[:, swap], dict(kind='mirror')


CASES = [
    ('control', '그대로 (대조군)', case_control),
    ('noise', '관절마다 1 cm 노이즈 (대조군)', case_noise),
    ('slow_descent', '2번째 하강만 1.5배 느리게', case_slow_descent),
    ('tempo', '전체 0.8배 시간 (빠르게)', case_tempo),
    ('hold', '3번째 최저점에서 1초 멈춤 추가', case_hold),
    ('no_rest', '2-3번째 사이 선 자세 쉼 제거', case_no_rest),
    ('rest_short70', '2-3번째 사이 쉼(0.40초)을 70%로 줄임', lambda B: case_short_rest(B, 0.7)),
    ('rest_short30', '2-3번째 사이 쉼(0.40초)을 30%로 줄임', lambda B: case_short_rest(B, 0.3)),
    ('extra_rep_mid', '3번째 반복을 한 번 더 (중간)', lambda B: case_extra_rep(B, 2)),
    ('extra_rep_end', '마지막 반복을 한 번 더 (끝)', lambda B: case_extra_rep(B, 4, at_end=True)),
    ('missing_rep', '4번째 반복 삭제', case_missing_rep),
    ('trunk', '3번째 최저점에서 상체 10° 더 숙임', case_trunk),
    ('shallow', '3번째 최저점에서 골반 8 cm 덜 내려감', case_shallow),
    ('knees_in', '3번째 최저점에서 무릎 15° 안쪽으로', case_knees_in),
    ('arm_up', '오른쪽 아래팔 항상 30° 위로', case_arm_up),
    ('mirror', '좌우 반전 (거울처럼 따라 함)', case_mirror),
]


# ---------------------------------------------------------------- evaluation
def overlaps(rg, zone, tol=0.1):
    return rg is not None and zone is not None and rg[0] <= zone[1] + tol and rg[1] >= zone[0] - tol


def over_threshold(d):
    out = []
    for k in ('kneeFlexDeg', 'hipFlexDeg'):
        for s in ('left', 'right'):
            if abs(d[k][s]) >= TH_DEG:
                out.append(f'{k}.{s} {d[k][s]:+}')
    if abs(d['trunkLeanDeg']) >= TH_DEG:
        out.append(f"trunkLeanDeg {d['trunkLeanDeg']:+}")
    for k in ('hipKneeCm', 'pelvisDropCm'):
        if abs(d[k]) >= TH_CM:
            out.append(f'{k} {d[k]:+}')
    if abs(d['kneeWidthRatio']) >= TH_RATIO:
        out.append(f"kneeWidthRatio {d['kneeWidthRatio']:+}")
    return out


def false_positives(r, zone_u=None, zone_c=None, skip_rep=None):
    fp = []
    for s in r['segments']:
        if overlaps(s.get('studentRange'), zone_u) or overlaps(s.get('coachRange'), zone_c):
            continue
        t = s.get('timing') or {}
        if s['verdict'] != 'match':
            fp.append(f"{s['id']} {s['label']} {s['verdict']}")
        elif t.get('label', 'ok') != 'ok':
            fp.append(f"{s['id']} {s['label']} {t['label']} {t['ratio']}")
        elif s['deviations']:
            fp.append(f"{s['id']} {s['label']} pose " + ', '.join(f"{d['bone']} {d['angleDiffDeg']:.0f}°" for d in s['deviations']))
    for rep in r['poseMetrics']['reps']:
        if skip_rep is not None and skip_rep(rep):
            continue
        d = rep['bottom'].get('diff')
        if d is None:
            fp.append(f"rep{rep['index']} unpaired (user {rep['userSec']}, coach {rep['coachSec']})")
        elif over_threshold(d):
            fp.append(f"rep{rep['index']} " + ', '.join(over_threshold(d)))
    return fp


def near(a, b, tol):
    return a is not None and b is not None and abs(a - b) <= tol


def mean_lr(v):
    return (v['left'] + v['right']) / 2 if isinstance(v, dict) else v


def evaluate(B, truth, r, s):
    k = truth['kind']
    found, info = [], []
    pm = r['poseMetrics']
    if k == 'control':
        fp = false_positives(r)
        info.append(f"tempo {s['global_tempo']}, mirrored {s['mirrored']}")
    elif k == 'tempo':
        found.append((f"globalTempo {s['global_tempo']} (정답 {truth['tempo']})", near(s['global_tempo'], truth['tempo'], 0.05)))
        fp = false_positives(r)
    elif k == 'timing':
        hit = [x for x in r['segments'] if x['verdict'] == 'match' and overlaps(x.get('studentRange'), truth['user'], 0)
               and (x.get('timing') or {}).get('label') == truth['label']]
        txt = ', '.join(f"{x['id']} {x['label']} {x['timing']['label']} {x['timing']['ratio']}" for x in hit) or '없음'
        found.append((f"해당 구간 {truth['label']}: {txt} (정답 비율 {truth['ratio']})",
                      bool(hit) and any(near(x['timing']['ratio'], truth['ratio'], 0.2) for x in hit)))
        fp = false_positives(r, zone_u=truth['user'])
    elif k == 'extra_hold':
        hit = [x for x in r['segments'] if x['verdict'] == 'extra' and overlaps(x.get('studentRange'), truth['user'])]
        dur = [round(x['studentRange'][1] - x['studentRange'][0], 2) for x in hit]
        found.append((f"추가 구간 {', '.join(x['id'] + ' ' + x['label'] for x in hit) or '없음'}, 길이 {dur} s (정답 {truth['sec']} s)",
                      bool(hit) and any(near(v, truth['sec'], 0.3) for v in dur)))
        fp = false_positives(r, zone_u=truth['user'])
    elif k == 'missing_rest':
        hit = [x for x in r['segments'] if x['verdict'] == 'missing' and overlaps(x.get('coachRange'), truth['coach'])]
        found.append((f"누락 구간 {', '.join(x['id'] + ' ' + x['label'] for x in hit) or '없음'} (정답: 코치 {truth['sec']:.2f} s 쉼)", bool(hit)))
        fp = false_positives(r, zone_c=truth['coach'])
    elif k == 'short_rest':
        segs = [x for x in r['segments'] if overlaps(x.get('coachRange'), truth['coach'], 0.05)
                and x['label'] in ('선 자세 유지', '시작 자세', '마무리 자세')]
        txt = ', '.join(f"{x['id']} {x['verdict']} {(x.get('timing') or {}).get('label', '')} hold={x.get('hold')}" for x in segs) or '없음'
        want = 'missing' if truth['expect'] == 'missing' else 'match'
        ok = any(x['verdict'] == want and (x.get('hold') is None or near(x['hold']['ratio'], truth['keep'], 0.2)) for x in segs)
        found.append((f"코치 쉼 {truth['sec']:.2f}s 를 {truth['keep']:.0%} 로: {txt} (정답 {want}, 비율 약 {truth['keep']})", ok))
        fp = false_positives(r, zone_c=truth['coach'])
    elif k == 'extra_rep':
        hit = [x for x in r['segments'] if x['verdict'] in ('extra', 'substitute') and overlaps(x.get('studentRange'), truth['user'])]
        rc = pm['repCount']
        is_dup = lambda x: x['bottom']['coach'] is None and any(near(x['userSec'], t, 0.3) for t in truth['user_bottoms'])
        un = [x for x in pm['reps'] if is_dup(x)]
        found.append((f"구간: {', '.join(x['id'] + ' ' + x['label'] + ' ' + x['verdict'] for x in hit) or '없음'}", bool(hit)))
        found.append((f"반복 수 {rc['user']} vs {rc['coach']} (정답 {truth['reps'][0]} vs {truth['reps'][1]}), "
                      f"짝 없는 사용자 반복 {[x['userSec'] for x in un]} "
                      f"(정답: 같은 두 반복 {truth['user_bottoms'][0]:.2f} s / {truth['user_bottoms'][1]:.2f} s 중 하나)",
                      (rc['user'], rc['coach']) == truth['reps'] and len(un) == 1))
        fp = false_positives(r, zone_u=truth['user'], skip_rep=is_dup)
    elif k == 'missing_rep':
        hit = [x for x in r['segments'] if x['verdict'] in ('missing', 'substitute') and overlaps(x.get('coachRange'), truth['coach'])]
        rc = pm['repCount']
        un = [x for x in pm['reps'] if x['bottom']['user'] is None and near(x['coachSec'], truth['coach_bottom'], 0.3)]
        found.append((f"구간: {', '.join(x['id'] + ' ' + x['label'] + ' ' + x['verdict'] for x in hit) or '없음'}", bool(hit)))
        found.append((f"반복 수 {rc['user']} vs {rc['coach']} (정답 {truth['reps'][0]} vs {truth['reps'][1]}), "
                      f"짝 없는 코치 반복 {[x['coachSec'] for x in un]} (정답 {truth['coach_bottom']:.2f} s)",
                      (rc['user'], rc['coach']) == truth['reps'] and bool(un)))
        fp = false_positives(r, zone_c=truth['coach'], skip_rep=lambda x: x['bottom']['user'] is None
                             and near(x['coachSec'], truth['coach_bottom'], 0.3))
    elif k == 'pose':
        rep = [x for x in pm['reps'] if near(x['coachSec'], truth['coach_bottom'], 0.2)]
        d = rep[0]['bottom'].get('diff') if rep else None
        tol = dict(kneeFlexDeg=2., hipFlexDeg=2., trunkLeanDeg=2., hipKneeCm=1., pelvisDropCm=1., kneeWidthRatio=0.03)
        rows, ok = [], d is not None
        for m, tv in truth['truth'].items():
            ov = mean_lr(d[m]) if d else None
            good = ov is not None and abs(ov - tv) <= tol[m]
            ok &= good
            if abs(tv) >= tol[m] or not good:
                rows.append(f"{m} {ov if ov is None else round(ov, 2)} (정답 {tv:+.2f})" + ('' if good else ' ✗'))
        found.append(('해당 반복 최저점: ' + ('; '.join(rows) or '변화 없음'), ok))
        segs = [x for x in r['segments'] if overlaps(x.get('studentRange'), (truth['user_bottom'] - 1.2, truth['user_bottom'] + 1.2))]
        devs = [f"{x['id']} {dv['bone']} {dv['angleDiffDeg']:.0f}° {dv.get('direction')}" for x in segs for dv in x['deviations']]
        info.append('구간 차이(참고): ' + (', '.join(devs) or '없음'))
        zone = (truth['user_bottom'] - 1.2, truth['user_bottom'] + 1.2)
        fp = false_positives(r, zone_u=zone, skip_rep=lambda x: near(x['coachSec'], truth['coach_bottom'], 0.2))
    elif k == 'arm':
        match = [x for x in r['segments'] if x['verdict'] == 'match']
        withb = [x for x in match for dv in x['deviations'] if dv['bone'] == truth['bone']]
        right = [x for x in match for dv in x['deviations'] if dv['bone'] == truth['bone'] and dv.get('direction') == truth['direction']]
        mags = [round(dv['angleDiffDeg']) for x in match for dv in x['deviations'] if dv['bone'] == truth['bone']]
        found.append((f"오른쪽 아래팔이 차이로 나온 구간 {len(withb)}/{len(match)}, 방향 up {len(right)}/{len(withb)}, "
                      f"크기 {sorted(set(mags))} (정답 {truth['deg']:g}°)",
                      len(withb) == len(match) and len(right) == len(withb) and all(abs(m - truth['deg']) <= 5 for m in mags)))
        fp = []
        for x in r['segments']:
            t = x.get('timing') or {}
            if x['verdict'] != 'match' or t.get('label', 'ok') != 'ok':
                fp.append(f"{x['id']} {x['label']} {x['verdict']} {t.get('label', '')}")
            other = [dv['bone'] for dv in x['deviations'] if dv['bone'] != truth['bone']]
            if other:
                fp.append(f"{x['id']} other bones {other}")
        fp += [f"rep{x['index']} " + ', '.join(over_threshold(x['bottom']['diff'])) for x in pm['reps']
               if x['bottom'].get('diff') and over_threshold(x['bottom']['diff'])]
    elif k == 'mirror':
        found.append((f"mirrored = {s['mirrored']}", bool(s['mirrored'])))
        fp = false_positives(r)
    return found, fp, info


def run_case(B, U):
    arrays = {'coach_on_user_pan.bvh': (B.X, B.names, B.fps), 'user.bvh': (U, B.names, B.fps)}
    real = b2n.bvh_joints
    b2n.bvh_joints = lambda p: arrays[Path(p).name]
    try:
        return rb.build_result('validation', 'squat', Path('/nonexistent'), with_score=True)
    finally:
        b2n.bvh_joints = real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', default='e80464dd6496', help='base skeleton: this job\'s retargeted coach')
    a = ap.parse_args()
    X, names, fps = b2n.bvh_joints(config.JOBS_DIR / a.job / 'retarget' / 'coach_on_user_pan.bvh')
    B = Base(X, names, fps)
    print(f'base: {len(X)} frames, {len(B.reps)} reps, bottoms {[r["bottom"] for r in B.reps]}')
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    global TH_CM
    n_ctrl = None
    for key, title, fn in CASES:
        U, truth = fn(B)
        r, s = run_case(B, U)
        if TH_CM is None:
            TH_CM = r['poseMetrics']['thresholds']['sameHeightCm']
        found, fp, info = evaluate(B, truth, r, s)
        # sanity for every case: no zero-length matched segment (a timed movement that takes 0 s)
        fp += [f"{x['id']} {x['label']} 0초 구간 (timing {x.get('timing')})" for x in r['segments']
               if x['verdict'] == 'match' and any(x.get(k) and x[k][1] - x[k][0] < 0.05 for k in ('studentRange', 'coachRange'))
               and (x.get('timing') or {}).get('label', 'ok') != 'ok']
        # unchanged structure must keep the control's segments (an empty result is not a pass)
        if key == 'control':
            n_ctrl = len(r['segments'])
        elif truth['kind'] in ('control', 'mirror', 'arm', 'tempo') and len(r['segments']) != n_ctrl:
            fp.append(f"구간 수 {len(r['segments'])} (대조군 {n_ctrl})")
        ok_found = all(ok for _, ok in found)
        results.append(dict(case=key, title=title, found=found, false_positives=fp, info=info,
                            passed=ok_found and not fp, truth={k: v for k, v in truth.items()}))
        (OUT / f'{key}.json').write_text(json.dumps(dict(truth=truth, result=r, score_segments=s['segments'],
                                                         global_tempo=s['global_tempo'], mirrored=s['mirrored']),
                                                    ensure_ascii=False, default=float))
        mark = 'PASS' if ok_found and not fp else ('FOUND+FP' if ok_found else 'MISS')
        print(f'{mark:8s} {key:14s} {title}')
        for t, ok in found:
            print(f'           {"✓" if ok else "✗"} {t}')
        for t in fp:
            print(f'           ! {t}')
        for t in info:
            print(f'           · {t}')
    lines = ['# 비교 알고리즘 검증 (정답을 넣은 합성 사용자)', '',
             f'기준 골격: job {a.job} 의 리타겟팅된 코치 ({len(X)} 프레임, {len(B.reps)}회). '
             f'임계값: {TH_DEG:g}°, {TH_CM:g} cm (다리 {config.POSE_SAME_LEG:g} 배), 비율 {TH_RATIO:g}.', '',
             '| 경우 | 결과 | 찾았나 | 다른 곳 오탐 |', '|---|---|---|---|']
    for x in results:
        res = '통과' if x['passed'] else ('찾음, 오탐 있음' if all(ok for _, ok in x['found']) else '놓침')
        lines.append(f"| {x['title']} | {res} | {'<br>'.join(('✓ ' if ok else '✗ ') + t for t, ok in x['found']) or '-'} "
                     f"| {'<br>'.join(x['false_positives']) or '없음'} |")
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n')
    (OUT / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=1, default=float))
    print('passed', sum(x['passed'] for x in results), '/', len(results), '->', OUT / 'report.md')


if __name__ == '__main__':
    main()
