"""llm stage: analysis result -> Korean feedback { summary, items[{ segmentId?, text }] }.

Input to the model is a compact summary built from the frontend result (segment ids, labels,
verdicts, timing, deviations with their direction, poseMetrics: joint angles at the exercise's key poses,
for a squat each bottom and standing) plus a few motion_scoring fields (global tempo, mirrored, coach / user
motion of missing / extra segments). No skeleton frames, no video, no score (there is no total score yet).
The instruction that goes with it is system_prompt(exercise): one text for every exercise, with the parts
that are about the exercise written by its module (app/result/exercises).

Provider: Google Gemini (google-genai SDK) with a JSON response schema; models are
constants.GEMINI_MODELS (tried in order on overload / rate limit / timeout), the key GEMINI_API_KEY (backend/.env). LLM_PROVIDER=none disables the stage.
Any failure (no key, timeout, API error, invalid output) returns None: the job still finishes,
the result simply has no `feedback` (the frontend shows its placeholder).
"""
import json
import time
import traceback
from pathlib import Path

from pydantic import BaseModel

from ... import constants
from .. import exercises
from ..result_builder import FRONT_BONES, swap_side

# frontend/src/utils/skeleton.ts BONE_LABELS
BONE_KO = {
    'torso': '몸통', 'head': '머리', 'left_clavicle': '왼쪽 어깨선', 'right_clavicle': '오른쪽 어깨선',
    'left_upper_arm': '왼쪽 위팔', 'right_upper_arm': '오른쪽 위팔', 'left_forearm': '왼쪽 아래팔',
    'right_forearm': '오른쪽 아래팔', 'left_pelvis': '왼쪽 골반', 'right_pelvis': '오른쪽 골반',
    'left_thigh': '왼쪽 허벅지', 'right_thigh': '오른쪽 허벅지', 'left_shin': '왼쪽 정강이', 'right_shin': '오른쪽 정강이',
}
EXERCISE_KO = {'squat': '스쿼트', 'deadlift': '데드리프트', 'bench_press': '벤치프레스'}

def system_prompt(exercise):
    """The instruction the model gets with the analysis of an exercise: the text below is the same for every
    exercise, and the exercise's module (app/result/exercises) writes the parts that are about the exercise --
    what a segment's deviations are in it, what poseMetrics holds, what to look at first, and a few examples."""
    p = exercises.get(exercise).PROMPT
    return f"""너는 운동 동작을 봐 주는 코치다. 사용자의 동작을 코치 시범 동작과 비교한 분석 결과(JSON)를 받아, 앱 화면에 보여 줄 짧은 피드백을 한국어로 쓴다. 코치 골격은 사용자 체형에 맞춰 바꾼 것이라 cm 값을 그대로 비교해도 된다.

입력 설명
- exercise: 종목.
- globalTempo: 사용자 전체 동작 시간 ÷ 코치 전체 동작 시간. 1보다 크면 전체적으로 코치보다 천천히, 작으면 빠르게 했다.
- mirrored: true 면 사용자가 거울처럼 좌우를 바꿔 따라 했다고 보고 비교했다. 좌우는 모두 사용자 기준이다.
- segments: 사용자 영상을 시간 순서로 나눈 구간. id, label(구간 이름), verdict(판정).
  - match: 코치와 같은 동작을 했다.
  - missing: 코치에게 있는 동작(예: {p['missing_example']})이 사용자에게는 없다.
  - extra: 코치에게 없는 동작(예: {p['extra_example']})을 사용자가 했다.
  - substitute: 같은 자리에서 코치와 다른 동작을 했다.
- userSec / coachSec: 그 구간에 사용자와 코치가 쓴 시간(초). 없으면 해당 사람에게 그 구간이 없다.
- timing: match 구간에만 있다. label 이 ok 면 속도 문제가 없다. slow / fast 면 사용자의 전체 속도를 감안해도 이 구간이 상대적으로 느리거나 빠르다는 뜻이다. ratio 는 (사용자 시간 ÷ 코치 시간)을 globalTempo 로 나눈 상대값이다.
- hold: 코치가 {p['rest_where']} 쉰 자리에서 사용자가 실제로 느려진 시간. coachSec / userSec(초), ratio = userSec ÷ (coachSec × globalTempo). ratio 가 0.5 미만이면 missing(사실상 쉬지 않음), 0.5 이상이면 쉼이 있는 match 구간이다. 쉼 구간({p['rest_labels']})의 timing 이 fast 면 쉼이 짧았다는 뜻이다.
- deviations: 그 구간에서 코치와 가장 크게 벌어진 순간의 신체 부위별 각도 차이(도). {p['deviations']} direction 은 사용자 쪽 부위가 코치보다 더 향한 쪽, detail 은 방향별 성분이다.
- coachMotion / userMotion: 코치에게만 / 사용자에게만 있는 구간에서 그 사람이 한 움직임(부위, 각도, 방향). 각도가 모두 5도 미만이면 사실상 가만히 있었다는 뜻이다.
{p['pose_metrics']}

규칙
- 입력에 있는 사실과 수치만 쓴다. 방향은 direction, detail, diff 의 부호로만 말하고, 그 밖의 방향이나 원인을 추측하지 않는다.
{p['rules']}
- 같은 차이가 여러 반복에서 되풀이되면 한 항목으로 묶어 범위로 말하고, 대표 구간 id 하나를 segmentId 에 넣는다.
- 반복 횟수(repCount)가 다르면 그 사실을 말한다.
- 코치가 {p['still']} missing 구간들(반복 사이 쉼)은 하나의 항목으로 묶는다. hold 가 있으면 "쉬지 않았다"가 아니라 "코치는 0.4~0.5초 쉬었고 사용자는 0.1~0.2초로 거의 쉬지 않았다"처럼 두 시간을 말한다.
- timing.ratio 는 상대값이므로 "1.5배 느려요" 처럼 배수로 말하지 않는다. userSec / coachSec 같은 절대 시간은 그대로 말해도 된다.
- 기준은 코치 동작이다. 코치와 다른 점과 코치에 가깝게 하는 연습 방법을 말하되, 의학적으로 옳고 그름을 단정하지 않는다.
- 말투는 "~해요" 체. 친절하고 짧게, 전문 용어 대신 쉬운 말을 쓴다(예: 엉덩이 굽힘 → 엉덩이를 뒤로 빼며 접는 정도).
- summary: 전체 평가 1~2문장.
- items: 최대 6개, 중요한 것부터. 각 2문장 이내로 무엇이 달랐는지와 어떻게 연습하면 좋은지를 쓴다. 문제없는 것은 쓰지 않는다.
- 점수, 의학적 진단, 부상 위험 단정은 말하지 않는다.
"""


# the instruction for a squat under its old name: the experiments under verification/ read the prompt from here
SYSTEM_PROMPT = system_prompt('squat')


class FeedbackItem(BaseModel):
    segmentId: str | None = None
    text: str


class Feedback(BaseModel):
    summary: str
    items: list[FeedbackItem]


def bone_ko(name):
    return BONE_KO.get(FRONT_BONES.get(name, name), name)


DIR_KO = {'up': '위로', 'down': '아래로', 'forward': '앞으로', 'backward': '뒤로', 'outward': '바깥쪽으로',
          'inward': '안쪽으로', 'left': '왼쪽으로', 'right': '오른쪽으로'}
AXES_KO = (('up', '위로', '아래로'), ('forward', '앞으로', '뒤로'), ('outward', '바깥쪽으로', '안쪽으로'),
           ('right', '오른쪽으로', '왼쪽으로'))


def detail_ko(components, min_deg=5., axes=AXES_KO):
    """{'up': 33.1, 'forward': -15.4, ...} -> '위로 33°, 뒤로 15°' (components >= min_deg)."""
    parts = [(abs(v), f'{pos if v > 0 else neg} {round(abs(v))}°')
             for axis, pos, neg in axes if abs(v := components.get(axis, 0.)) >= min_deg]
    return ', '.join(t for _, t in sorted(parts, reverse=True))


def motion_ko(motions, mirrored=False, dir_ko=DIR_KO):
    return [dict(part=bone_ko(swap_side(FRONT_BONES.get(m['bone'], m['bone'])) if mirrored else m['bone']),
                 deg=round(m['deg']), direction=dir_ko.get(m.get('dir'), m.get('dir'))) for m in motions]


def llm_input(result, score):
    """Compact JSON-able summary for the model (result: AnalysisResult dict, score: motion_scoring to_dict()).
    An exercise may have its own words for directions (a lifter lying down) and its own values in place of
    poseMetrics (score['exerciseMetrics'], see result_builder.build_result): app/result/exercises."""
    ex = exercises.MODULES.get(result['exercise'])
    axes, dir_ko = getattr(ex, 'AXES_KO', AXES_KO), {**DIR_KO, **getattr(ex, 'DIR_KO', {})}
    segs = []
    units = score.get('segments', [])
    mirrored = bool(score.get('mirrored'))
    for k, s in enumerate(result['segments']):
        u = units[k] if k < len(units) else {}
        item = dict(id=s['id'], label=s['label'], verdict=s['verdict'])
        st = u.get('student_t')                         # on the user's video (studentRange is the viewer's timeline)
        if st and st[1] > st[0]:
            item['userSec'] = round(st[1] - st[0], 2)
        if 'coachRange' in s:
            item['coachSec'] = round(s['coachRange'][1] - s['coachRange'][0], 2)
        if 'timing' in s:
            item['timing'] = s['timing']
        if 'hold' in s:
            item['hold'] = s['hold']
        if s['deviations']:
            item['deviations'] = []
            for d in s['deviations']:
                dv = dict(part=BONE_KO.get(d['bone'], d['bone']), angleDiffDeg=round(d['angleDiffDeg']))
                if d.get('direction'):
                    dv['direction'] = dir_ko[d['direction']]
                    dv['detail'] = detail_ko(d['components'], axes=axes)
                elif d.get('components'):               # a direction the response does not name: the largest component
                    axis, value = max(d['components'].items(), key=lambda kv: abs(kv[1]))
                    dv['direction'] = next(pos if value > 0 else neg for a, pos, neg in axes if a == axis)
                    dv['detail'] = detail_ko(d['components'], axes=axes)
                item['deviations'].append(dv)
        if u.get('coach_motion'):                       # coach's own sides
            item['coachMotion'] = motion_ko(u['coach_motion'], dir_ko=dir_ko)
        if u.get('student_motion'):                     # scored on the mirrored user when mirrored
            item['userMotion'] = motion_ko(u['student_motion'], mirrored, dir_ko)
        segs.append(item)
    out = dict(exercise=EXERCISE_KO.get(result['exercise'], result['exercise']),
               globalTempo=score.get('global_tempo'), mirrored=mirrored, segments=segs)
    metrics = score.get('exerciseMetrics') or result.get('poseMetrics')
    if metrics:
        out['poseMetrics'] = metrics
    return out


def clean(fb: Feedback, segment_ids):
    """Drop unknown segment ids and empty texts; None if nothing usable."""
    items = []
    for it in fb.items[:10]:
        text = it.text.strip()
        if not text:
            continue
        sid = it.segmentId if it.segmentId in segment_ids else None
        items.append({'segmentId': sid, 'text': text} if sid else {'text': text})
    summary = fb.summary.strip()
    return dict(summary=summary, items=items) if summary else None


def call_gemini(payload, model, timeout_sec, prompt=None):
    """prompt: the system instruction (system_prompt of the exercise); None = that of a squat"""
    import os
    from google import genai
    from google.genai import types
    key = os.environ.get('GEMINI_API_KEY')
    if not key:
        raise RuntimeError('GEMINI_API_KEY is not set (backend/.env)')
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(timeout_sec * 1000)))
    resp = client.models.generate_content(
        model=model,
        contents=json.dumps(payload, ensure_ascii=False),
        config=types.GenerateContentConfig(system_instruction=prompt or SYSTEM_PROMPT, response_mime_type='application/json',
                                           response_schema=Feedback, temperature=0.3),
    )
    fb = resp.parsed if isinstance(resp.parsed, Feedback) else Feedback.model_validate_json(resp.text)
    return fb, resp.text


def call_with_fallback(payload, log, prompt=None):
    """Try constants.GEMINI_MODELS in order within constants.LLM_DEADLINE_SEC; -> (Feedback, raw, model)."""
    start, last = time.time(), None
    for model in constants.GEMINI_MODELS:
        left = constants.LLM_DEADLINE_SEC - (time.time() - start)
        if left < 5:
            break
        t0 = time.time()
        try:
            fb, raw = call_gemini(payload, model, min(constants.LLM_TIMEOUT_SEC, left), prompt)
            log.append(f'{model}: ok ({time.time() - t0:.1f}s)')
            return fb, raw, model
        except Exception as e:                           # noqa: BLE001 -- next model
            last = e
            log.append(f'{model}: {type(e).__name__} ({time.time() - t0:.1f}s) {str(e)[:300]}')
    raise RuntimeError(f'all Gemini models failed, last: {last!r}')


def generate_feedback(result, score, log_dir=None):
    """-> feedback dict for result['feedback'], or None (stage skipped / failed; see logs/llm.log)."""
    if constants.LLM_PROVIDER == 'none':
        return None
    payload = llm_input(result, score)
    log = [f'provider {constants.LLM_PROVIDER} models {constants.GEMINI_MODELS}',
           'input ' + json.dumps(payload, ensure_ascii=False, indent=1)]
    t0 = time.time()
    out = None
    try:
        if constants.LLM_PROVIDER != 'gemini':
            raise RuntimeError(f'unknown LLM_PROVIDER {constants.LLM_PROVIDER}')
        fb, raw, model = call_with_fallback(payload, log, system_prompt(result['exercise']))
        log.append(f'raw {raw}')
        out = clean(fb, {s['id'] for s in result['segments']})
        log.append('feedback ' + json.dumps(out, ensure_ascii=False, indent=1))
    except Exception:                                    # noqa: BLE001 -- feedback is optional
        log.append(f'FAILED ({time.time() - t0:.1f}s)\n' + traceback.format_exc())
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        (Path(log_dir) / 'llm.log').write_text('\n'.join(log) + '\n')
    return out
