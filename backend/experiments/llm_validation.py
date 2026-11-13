"""LLM stage validation: does the feedback say what the JSON says, without inventing or reversing anything?

Inputs
  synthetic  the comparison-validation cases (data/experiments/validation/<case>.json, from
             experiments.compare_validation): the injected difference is known, so the feedback must mention
             it (need), must not claim the opposite (avoid), and control-like cases must have no items
  real       real pairs (job e80464dd6496, demo student, child): need / avoid from what their JSON shows,
             plus a human read
Models: the production prompt and payload (app.feedback) on each model in --models, with retries.

Checks (all automatic)
  grounded   every number in the text is in the payload (or a simple derivation: |user - coach|, ranges,
             tempo as %, rep-count difference); ordinals ("3번째") are skipped
  need       each group of patterns is found together in one item (or the summary)
  avoid      none of the patterns (claims in the wrong direction / of something absent) appear
  no_items   for inputs without any real difference: no items

Run from backend/:  ../.venv/bin/python -m experiments.llm_validation
  --only case1,case2   subset; --models m1,m2; --score-only re-scores saved outputs (e.g. hand-written ones)
Outputs: data/experiments/llm_validation/<model>/<input>.json and report.md
"""
import argparse
import json
import re
import time
from pathlib import Path

from app import config, feedback
from app.result_builder import build_result

OUT = config.DATA_DIR / 'experiments' / 'llm_validation'
SYN = config.DATA_DIR / 'experiments' / 'validation'

REST = r'쉬|쉼|멈|휴식'
SPEC = {
    'control': dict(no_items=True),
    'noise': dict(no_items=True),
    'slow_descent': dict(need=[(r'내려|하강', r'느리|느렸|천천|오래')]),
    'tempo': dict(need=[(r'빠르|빨리|빨랐|빠른',)]),
    'hold': dict(need=[(r'바닥|가장 낮|최저|앉은', r'멈|머무|머물|버티|유지')]),
    'no_rest': dict(need=[(REST,)]),
    'rest_short70': dict(avoid=[r'쉬지 않', r'쉼이 없', r'쉬는 시간이 없', r'쉼.{0,6}(생략|빠졌|빠뜨)']),
    'rest_short30': dict(need=[(REST,)]),
    'extra_rep_mid': dict(need=[(r'6\s*(회|번)|한 번 더|1회 더|1번 더',)]),
    'extra_rep_end': dict(need=[(r'6\s*(회|번)|한 번 더|1회 더|1번 더',)]),
    'missing_rep': dict(need=[(r'4\s*(회|번)|한 번 (덜|적)|1회 (덜|적)|1번 (덜|적)',)]),
    'trunk': dict(need=[(r'상체|몸통', r'숙')], avoid=[r'(상체|몸통)[^.]{0,12}덜 숙(였|여|인)', r'덜 숙였']),
    'shallow': dict(need=[(r'얕|덜 내려|덜 앉|높게|높았|높아',)],
                    avoid=[r'더 (깊게|깊이) (앉았|내려갔|앉으셨)', r'더 낮게 (앉았|내려갔)']),
    'knees_in': dict(need=[(r'무릎', r'안쪽|모였|모이|모아|좁')], avoid=[r'무릎[^.]{0,12}(더 벌어|바깥쪽으로 벌)']),
    'arm_up': dict(no_items=True),        # the JSON has no arm difference (scorer property A): nothing to say
    'mirror': dict(no_items=True),
}
REAL = {
    'job_e80464dd6496': dict(job='e80464dd6496', need=[(r'6\s*(회|번)|1회 더|한 번 더|1번 더',),
                                                       (r'바닥|가장 낮|앉은', r'멈|머무|머물'),
                                                       (r'상체', r'숙'), (REST,)],
                             avoid=[r'얕게 앉았', r'덜 (앉았|내려갔)', r'상체[^.]{0,12}더 숙(였|여)']),
    'demo_student': dict(job='demo', need=[(r'4(\.0)?\s*초',), (r'상체', r'숙'), (r'무릎', r'안쪽|모였|모이|좁')],
                         avoid=[r'상체[^.]{0,12}덜 숙(였|여)', r'더 (깊게|깊이) (앉았|내려갔)']),
    'child': dict(job='child', need=[(r'얕|덜 내려|덜 앉|높게|높았|높아',), (r'상체', r'숙')],
                  avoid=[r'더 (깊게|깊이) (앉았|내려갔)', r'상체[^.]{0,12}덜 숙(였|여)']),
}
REAL_DIRS = {'demo': config.PIPELINE_REPO / 'runs/coach_user/coach_to_student',
             'child': config.PIPELINE_REPO / 'runs/coach_user/coach_to_child'}


# ---------------------------------------------------------------- inputs
def synthetic_payload(case):
    d = json.loads((SYN / f'{case}.json').read_text())
    score = dict(segments=d['score_segments'], global_tempo=d['global_tempo'], mirrored=d['mirrored'])
    return feedback.llm_input(d['result'], score), {s['id'] for s in d['result']['segments']}


def real_payload(job):
    rd = REAL_DIRS.get(job, config.JOBS_DIR / job / 'retarget')
    r, s = build_result(job, 'squat', rd, with_score=True)
    return feedback.llm_input(r, s), {x['id'] for x in r['segments']}


# ---------------------------------------------------------------- checks
NUM_RE = re.compile(r'(?<![\d.\-])(\d+(?:\.\d+)?)(?![\d])')
UNIT = [('deg', r'\s*(도|°)'), ('cm', r'\s*(cm|센티)'), ('sec', r'\s*초'), ('count', r'\s*(회|번)(?!\s*째)'),
        ('pct', r'\s*(%|퍼센트)')]


def _cat(key, cat):
    if key in ('up', 'forward', 'outward', 'right', 'deg') or 'Deg' in key:
        return 'deg'
    if 'Cm' in key:
        return 'cm'
    if 'Sec' in key or key in ('coach_s', 'student_s'):
        return 'sec'
    if key == 'repCount':
        return 'count'
    if 'atio' in key or key in ('globalTempo', 'ratio'):
        return 'ratio'
    return cat


def allowed_numbers(payload):
    """{category: [values]} from the payload, by key name, plus simple derivations."""
    pools = {c: [] for c in ('deg', 'cm', 'sec', 'count', 'pct', 'ratio', 'other')}

    def walk(x, cat):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            pools[cat].append(abs(float(x)))
        elif isinstance(x, str):
            if cat == 'deg':                                    # deviation detail: "위로 33°, 뒤로 15°"
                pools['deg'] += [float(n) for n in re.findall(r'\d+(?:\.\d+)?', x)]
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(v, _cat(k, cat) if k != 'detail' else 'deg')
        elif isinstance(x, list):
            for v in x:
                walk(v, cat)
    walk(payload, 'other')
    for s in payload.get('segments', []):
        if 'userSec' in s and 'coachSec' in s:
            pools['sec'].append(abs(s['userSec'] - s['coachSec']))
        if 'hold' in s:
            pools['sec'].append(abs(s['hold']['coachSec'] - s['hold']['userSec']))
        if 'timing' in s:
            pools['pct'].append(round(abs(1 - s['timing']['ratio']) * 100))
    t = payload.get('globalTempo')
    if t:
        pools['pct'] += [round(abs(1 - t) * 100), round(abs(1 - 1 / t) * 100)]
    pm = payload.get('poseMetrics') or {}
    rc = pm.get('repCount')
    if rc:
        pools['count'].append(abs(rc['user'] - rc['coach']))
    for rep in pm.get('reps', []):          # left / right means, as the summary uses them
        for who in ('user', 'coach', 'diff'):
            p = rep['bottom'].get(who)
            if p:
                pools['deg'] += [abs(p[k]['left'] + p[k]['right']) / 2 for k in ('kneeFlexDeg', 'hipFlexDeg')]
                pools['sec'].append(rep['userSec'] or 0); pools['sec'].append(rep['coachSec'] or 0)
    return pools


def ungrounded(text, pools):
    bad = []
    everything = [v for vs in pools.values() for v in vs]
    for m in NUM_RE.finditer(text):
        tok, rest, before = m.group(1), text[m.end():m.end() + 6], text[max(0, m.start() - 4):m.start()]
        if re.match(r'\s*(번\s*째|회차)', rest) or before.endswith('seg-'):
            continue
        # a range "31~36도": the first number takes the unit of the second
        rng = re.match(r'\s*(~|∼|–|-|에서)\s*\d+(?:\.\d+)?', rest + text[m.end() + 6:m.end() + 12])
        tail = text[m.end():m.end() + 16] if rng else rest
        if rng:
            tail = re.sub(r'^\s*(~|∼|–|-|에서)\s*\d+(?:\.\d+)?', '', tail)
        cat = next((c for c, pat in UNIT if re.match(pat, tail)), None)
        pool = pools[cat] if cat else everything
        v = float(tok)
        tol = 0.051 if '.' in tok else 0.6
        if not any(abs(v - a) <= tol for a in pool):
            bad.append(tok + (dict(deg='도', cm='cm', sec='초', count='회', pct='%').get(cat, '')))
    return bad


def units_of(fb):
    return [fb['summary']] + [it['text'] for it in fb['items']]


def check(fb, payload, spec):
    out = {}
    pools = allowed_numbers(payload)
    text = ' '.join(units_of(fb))
    out['grounded'] = ungrounded(text, pools)
    if spec.get('no_items'):
        out['no_items'] = len(fb['items']) == 0
    if spec.get('need'):
        miss = [g for g in spec['need'] if not any(all(re.search(p, u) for p in g) for u in units_of(fb))]
        out['need_missing'] = [' & '.join(g) for g in miss]
    if spec.get('avoid'):
        out['avoid_hit'] = [p for p in spec['avoid'] if re.search(p, text)]
    out['passed'] = (not out['grounded'] and out.get('no_items', True)
                     and not out.get('need_missing') and not out.get('avoid_hit'))
    return out


# ---------------------------------------------------------------- run
def call(payload, model, log, tries=5):
    for k in range(tries):
        t0 = time.time()
        try:
            fb, raw = feedback.call_gemini(payload, model, 90)
            return dict(summary=fb.summary, items=[i.model_dump(exclude_none=True) for i in fb.items]), round(time.time() - t0, 1)
        except Exception as e:                                  # noqa: BLE001
            log(f'      try {k + 1}: {type(e).__name__} {str(e)[:70]}')
            time.sleep(min(20 * (k + 1), 60))
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='gemini-3.5-flash,gemini-3.1-flash-lite')
    ap.add_argument('--only', default='')
    ap.add_argument('--score-only', action='store_true', help='re-score saved outputs without calling the API')
    a = ap.parse_args()
    only = set(filter(None, a.only.split(',')))
    inputs = [(c, 'synthetic') for c in SPEC] + [(k, 'real') for k in REAL]
    inputs = [x for x in inputs if not only or x[0] in only]
    models = a.models.split(',')
    payloads = {}
    for name, kind in inputs:
        payloads[name] = synthetic_payload(name) if kind == 'synthetic' else real_payload(REAL[name]['job'])
    rows = []
    for model in models:
        mdir = OUT / model
        mdir.mkdir(parents=True, exist_ok=True)
        for name, kind in inputs:
            payload, seg_ids = payloads[name]
            spec = SPEC.get(name) or REAL[name]
            f = mdir / f'{name}.json'
            if a.score_only or (f.exists() and json.loads(f.read_text()).get('feedback')):
                saved = json.loads(f.read_text()) if f.exists() else {}
                fb, sec = saved.get('feedback'), saved.get('seconds')
            else:
                print(f'  {model} {name}')
                fb, sec = call(payload, model, print)
            res = dict(model=model, input=name, kind=kind, seconds=sec, feedback=fb,
                       status='ok' if fb else 'api_fail')
            if fb:
                fb = feedback.clean(feedback.Feedback.model_validate(fb), seg_ids) or dict(summary='', items=[])
                res['feedback'] = fb
                res['check'] = check(fb, payload, spec)
            f.write_text(json.dumps(res, ensure_ascii=False, indent=1))
            rows.append(res)
            c = res.get('check', {})
            print(f"{model:24s} {name:18s} {res['status']:8s} {'PASS' if c.get('passed') else 'FAIL' if c else '-':4s} "
                  f"{'' if not c else {k: v for k, v in c.items() if k != 'passed' and v not in ([], True)}}")
    report(rows, models)


def report(rows, models):
    lines = ['# LLM 피드백 검증', '', '| 입력 | ' + ' | '.join(models) + ' |', '|---|' + '---|' * len(models)]
    for name in dict.fromkeys(r['input'] for r in rows):
        cells = []
        for m in models:
            r = next((x for x in rows if x['input'] == name and x['model'] == m), None)
            if not r or r['status'] != 'ok':
                cells.append('API 실패')
                continue
            c = r['check']
            probs = ([f"근거 없는 수치 {c['grounded']}"] if c['grounded'] else []) + \
                    (['항목이 있으면 안 됨'] if c.get('no_items') is False else []) + \
                    [f'빠짐: {g}' for g in c.get('need_missing', [])] + [f'반대/금지 표현: {p}' for p in c.get('avoid_hit', [])]
            cells.append('통과' if c['passed'] else '<br>'.join(probs))
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    for m in models:
        ok = [r for r in rows if r['model'] == m and r['status'] == 'ok']
        lines.append(f"\n**{m}**: {sum(r['check']['passed'] for r in ok)}/{len(ok)} 통과 (API 실패 "
                     f"{sum(r['model'] == m and r['status'] != 'ok' for r in rows)})")
    lines += ['', '## 답변 원문', '']
    for r in rows:
        if r['status'] == 'ok':
            lines += [f"### {r['model']} | {r['input']}", '', f"**summary** {r['feedback']['summary']}", '']
            lines += [f"- `{i.get('segmentId', '-')}` {i['text']}" for i in r['feedback']['items']] + ['']
    (OUT / 'report.md').write_text('\n'.join(lines))
    print('wrote', OUT / 'report.md')


if __name__ == '__main__':
    main()
