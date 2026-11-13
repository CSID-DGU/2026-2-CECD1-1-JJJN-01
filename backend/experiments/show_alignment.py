"""Alignment report of one job: CP-DTW's own result -> what the backend sends -> what the frontend shows.

  1. CP-DTW (motion_scoring), re-run on the job's retarget outputs (deterministic, same input as the API):
     checkpoints of both clips, the checkpoint pairing (ops) and anchors, the frame path with its step kinds
  2. backend: result['alignment'] built by result_builder from that path (checked to be the same pairs)
  3. frontend: the user frame -> coach frame table that utils/skeleton.ts buildAlignmentLookup derives

Run from backend/:  ../.venv/bin/python -m experiments.show_alignment <jobId>
Writes data/jobs/<jobId>/alignment_report.md and cpdtw_alignment.json (raw path with kinds, events, ops).
"""
import json
import sys
from collections import Counter
from dataclasses import replace

import numpy as np

from app import config
from app.result_builder import build_result
from pipeline.bvh_to_npz import bvh_joints
from pipeline.score_user import SCORING_DIR

if str(SCORING_DIR) not in sys.path:
    sys.path.insert(0, str(SCORING_DIR))
from motion_scoring import Config, score, track_from_joints  # noqa: E402

KIND = {'match': '둘 다 한 프레임씩 진행', 'warp_c': '코치만 진행 (사용자는 그 프레임에 머묾 = 사용자가 빨리 지나감)',
        'warp_s': '사용자만 진행 (코치는 그 프레임에 머묾 = 사용자가 느리거나 멈춤)',
        'drop_c': '코치 프레임 버림 (짝 없음)', 'drop_s': '사용자 프레임 버림 (짝 없음)'}
EV = {'rest': '시작·끝 정지', 'pause': '멈춤', 'reversal': '방향 전환'}


def frontend_lookup(alignment, n):
    """Same rule as frontend/src/utils/skeleton.ts buildAlignmentLookup."""
    lk = [-1] * n
    for s, c in alignment:
        if 0 <= s < n and lk[s] == -1:
            lk[s] = c
    last = 0
    for i in range(n):
        if lk[i] == -1:
            lk[i] = last
        else:
            last = lk[i]
    return lk


def runs(lookup, covered):
    """Compress user frame -> coach frame into readable runs."""
    out, i, n = [], 0, len(lookup)
    while i < n:
        j = i
        if lookup[i] == (lookup[i - 1] if i else None) or i not in covered:      # frozen / not covered
            pass
        while j + 1 < n and ((lookup[j + 1] == lookup[j] and ((j + 1) in covered) == (i in covered))
                             or (lookup[j + 1] == lookup[j] + 1 and (j + 1) in covered and i in covered
                                 and lookup[i + 1 if i + 1 < n else i] == lookup[i] + 1)):
            j += 1
        a, b = lookup[i], lookup[j]
        if i not in covered:
            note = '짝 없음 → 직전 코치 프레임 유지' if i else '짝 없음 → 기본값 0'
        elif a == b and j > i:
            note = '코치 멈춤'
        elif b - a == j - i:
            note = '함께 진행'
        else:
            note = ''
        out.append((i, j, a, b, note))
        i = j + 1
    return out


def main():
    job = sys.argv[1]
    d = config.JOBS_DIR / job
    rd = d / 'retarget'
    coach, names, fc = bvh_joints(rd / 'coach_on_user_pan.bvh')
    user, _, fu = bvh_joints(rd / 'user.bvh')
    cfg = replace(Config(), auto_tau=True, pose_trim=0.3)               # as result_builder / pipeline.score_user
    tr = lambda j, f: track_from_joints(j, np.arange(len(j)) / f, 'y', None, names)
    res = score(tr(coach, fc), tr(user, fu), cfg, try_mirror=True)
    D = res.debug
    result = build_result(job, 'squat', rd)
    al = result['alignment']
    mine = sorted({(int(j), int(i)) for k, i, j in D['path'] if k not in ('drop_c', 'drop_s')})
    same = [list(p) for p in mine] == al
    nU, nC = len(user), len(coach)
    kinds = Counter(k for k, i, j in D['path'])
    pu = [j for k, i, j in D['path']]
    pc = [i for k, i, j in D['path']]
    lk = frontend_lookup(al, nU)
    covered = {s for s, c in al}
    skipped = sorted(set(range(min(pc), max(pc) + 1)) - {c for c in lk})

    (d / 'cpdtw_alignment.json').write_text(json.dumps(dict(
        path=[dict(kind=k, coach=int(i), user=int(j)) for k, i, j in D['path']],
        anchors=[dict(coach=int(a), user=int(b), coachEvent=ce, userEvent=ue) for a, b, ce, ue in D['anchors']],
        ops=[{k: v for k, v in o.items()} for o in D['ops']],
        coachEvents=[e.as_dict() for e in D['eC']], userEvents=[e.as_dict() for e in D['eS']],
        mirrored=res.mirrored if hasattr(res, 'mirrored') else res.to_dict()['mirrored']), ensure_ascii=False, default=float))

    L = [f'# 정렬 보고서: job {job}', '',
         f'사용자 {nU}프레임, 코치 {nC}프레임, 30 fps. 좌우 반전 채점: {res.to_dict()["mirrored"]}.', '',
         '## 1. CP-DTW 결과 (motion_scoring 원본)', '', '### 1-1. 마디 (체크포인트)', '']
    for who, ev, fps in (('코치', D['eC'], fc), ('사용자', D['eS'], fu)):
        L.append(f'**{who}** {len(ev)}개: ' + ', '.join(
            f"{k}:{EV.get(e.type, e.type)} {e.t:.2f}s({e.dwell:.2f}s)" for k, e in enumerate(ev)))
        L.append('')
    L += ['### 1-2. 마디 짝짓기 (ops) 와 기준점 (anchors)', '',
          '| # | 판정 | 코치 마디 | 사용자 마디 | 비용 | 여유(margin) |', '|---|---|---|---|---|---|']
    for n, o in enumerate(D['ops']):
        ce = f"{o['i']} ({D['eC'][o['i']].t:.2f}s)" if o.get('i') is not None else '-'
        ue = f"{o['j']} ({D['eS'][o['j']].t:.2f}s)" if o.get('j') is not None else '-'
        L.append(f"| {n} | {o['kind']} | {ce} | {ue} | {o.get('cost', 0):.2f} | "
                 f"{'-' if o.get('margin') is None else round(o['margin'], 2)} |")
    L += ['', '기준점(확실한 짝, 프레임 정렬의 고정점) [코치 프레임, 사용자 프레임]: ' +
          ', '.join(f'[{a}, {b}]' for a, b, _, _ in D['anchors']), '',
          '### 1-3. 프레임 정렬 경로 (path)', '',
          f'총 {len(D["path"])}단계. 코치 프레임 {min(pc)}~{max(pc)}, 사용자 프레임 {min(pu)}~{max(pu)} 를 덮는다 '
          f'(첫 기준점 앞, 마지막 기준점 뒤는 경로 밖).', '']
    L += [f'- `{k}` {kinds.get(k, 0)}: {KIND[k]}' for k in KIND]
    L += ['', '처음 15단계 (종류, 코치, 사용자): ' + ', '.join(f'({k}, {i}, {j})' for k, i, j in D['path'][:15]), '',
          '## 2. 백엔드가 프론트에 보내는 정렬 (result.alignment)', '',
          '- 변환: 버림(drop_c, drop_s) 제외 → [사용자 프레임, 코치 프레임] 으로 순서 바꿈 → 중복 제거 · 정렬. 값은 그대로.',
          f'- 결과 {len(al)}쌍. CP-DTW 경로를 같은 규칙으로 바꾼 것과 **{"완전히 같음" if same else "다름 (확인 필요)"}**.',
          f'- 짝이 있는 사용자 프레임 {len(covered)}/{nU}개. 짝이 없는 사용자 프레임: '
          + (', '.join(f'{a}~{b}' if a != b else f'{a}' for a, b in _ranges(sorted(set(range(nU)) - covered))) or '없음'),
          f'- 처음 12쌍: {al[:12]}', '',
          '함께 보내는 다른 정보: segments(구간·판정·속도·차이·쉼), student / coachRetargeted(골격 프레임), '
          'poseMetrics(관절 각도), feedback(LLM). 정렬과 관련된 것은 alignment 하나뿐이다.', '',
          '## 3. 프론트가 실제로 쓰는 표 (buildAlignmentLookup)', '',
          '- 규칙: 사용자 프레임마다 첫 번째 짝을 쓰고, 짝이 없으면 직전 코치 프레임을 그대로 쓴다.',
          f'- 한 번도 보이지 않는 코치 프레임: {len(skipped)}개 (경로 범위 {min(pc)}~{max(pc)} 안에서)', '',
          '| 사용자 프레임 | 코치 프레임 | 비고 |', '|---|---|---|']
    for i, j, a, b, note in runs(lk, covered)[:60]:
        L.append(f"| {i}{'' if i == j else f'~{j}'} | {a}{'' if a == b else f'~{b}'} | {note} |")
    # 0.5 s view of the CP-DTW pairs on the user timeline, with the step kinds and the user's segment
    by_u = {}
    for k, i, j in D['path']:
        by_u.setdefault(j, []).append((i, k))
    step_ko = {'match': '함께', 'warp_c': '코치만 진행', 'warp_s': '코치 대기', 'drop_c': '코치 버림', 'drop_s': '사용자 버림'}

    def seg_at(t):
        for s in result['segments']:
            r = s.get('studentRange')
            if r and r[1] > r[0] and r[0] <= t <= r[1]:
                return s['label'] + ('' if s['verdict'] == 'match' else f" ({s['verdict']})")
        return '-'
    L += ['', '## 4. 0.5초 간격 매칭표 (CP-DTW 경로, 사용자 시간 기준)', '',
          '| 사용자 시각 (프레임) | 코치 프레임 (시각) | 걸음 종류 | 사용자 구간 |', '|---|---|---|---|']
    step = int(round(0.5 * fu))
    for u in range(0, nU, step):
        m = by_u.get(u)
        if not m:
            L.append(f'| {u / fu:.1f}초 ({u}) | 짝 없음 (정렬 범위 밖) | - | {seg_at(u / fu)} |')
            continue
        cs = [c for c, _ in m]
        a, b = min(cs), max(cs)
        L.append(f"| {u / fu:.1f}초 ({u}) | {a if a == b else f'{a}~{b}'} ({a / fc:.2f}초{'' if a == b else f'~{b / fc:.2f}초'}) | "
                 f"{', '.join(sorted({step_ko.get(k, k) for _, k in m}))} | {seg_at(u / fu)} |")
    (d / 'alignment_report.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L[:12]))
    print(f'... wrote {d / "alignment_report.md"} and cpdtw_alignment.json | backend alignment == CP-DTW path: {same}')


def _ranges(xs):
    out = []
    for x in xs:
        if out and x == out[-1][1] + 1:
            out[-1][1] = x
        else:
            out.append([x, x])
    return out


if __name__ == '__main__':
    main()
