"""2x2 ablation: does the feedback change because of the new JSON or because of the new system prompt?

  prompt  old = production SYSTEM_PROMPT before 2026-11-12 (verbatim below)   new = app.feedback.SYSTEM_PROMPT
  json    old = llm_input without the fields added on 2026-11-12 (poseMetrics, deviation direction / detail,
                userMotion)                                                    new = app.feedback.llm_input

Run from backend/:
  ../.venv/bin/python -m experiments.prompt_json_ablation --pair job:2f770147a336 --pair demo
Outputs: data/experiments/ablation/<pair>/<prompt>_<json>.json and data/experiments/ablation/report.md
"""
import argparse
import copy
import json
import time

from app import config, feedback
from app.result_builder import build_result
from experiments.llm_compare import numbers_cited, pair_spec

OUT = config.DATA_DIR / 'experiments' / 'ablation'

OLD_PROMPT = """너는 운동 동작을 봐 주는 코치다. 사용자의 동작을 코치 시범 동작과 비교한 분석 결과(JSON)를 받아, 앱 화면에 보여 줄 짧은 피드백을 한국어로 쓴다.

입력 설명
- exercise: 종목.
- segments: 사용자 영상을 시간 순서로 나눈 구간. id, label(구간 이름), verdict(판정).
  - match: 코치와 같은 동작을 했다.
  - missing: 코치에게 있는 동작(예: 바닥에서 잠시 멈춤)이 사용자에게는 없다.
  - extra: 코치에게 없는 동작을 사용자가 했다.
  - substitute: 같은 자리에서 코치와 다른 동작을 했다.
- userSec / coachSec: 그 구간에 사용자와 코치가 쓴 시간(초). 없으면 해당 사람에게 그 구간이 없다.
- timing: match 구간에만 있다. label 이 ok 면 속도 문제가 없다. slow / fast 면 사용자의 전체 속도를 감안해도 이 구간이 상대적으로 느리거나 빠르다는 뜻이다. ratio 는 (사용자 시간 ÷ 코치 시간)을 globalTempo 로 나눈 상대값이므로 "1.5배 느려요" 처럼 절대 배수로 말하지 않는다.
- globalTempo: 사용자 전체 동작 시간 ÷ 코치 전체 동작 시간. 1보다 크면 전체적으로 코치보다 천천히 했다.
- deviations: 그 구간에서 사용자와 코치의 신체 부위 방향이 가장 많이 달랐던 부위와 각도 차이(도). 어느 쪽으로 달랐는지(더 숙였는지, 더 벌렸는지 등)는 주어지지 않는다.
- coachMotion: missing / substitute 구간에서 코치가 한 움직임(부위, 각도, 방향).
- mirrored: true 면 사용자가 거울처럼 좌우를 바꿔 따라 했다고 보고 비교했다.

규칙
- 입력에 있는 사실과 수치만 쓴다. 방향이나 원인(무릎이 안으로 모인다, 허리가 굽었다 등)을 추측하지 않는다. 방향을 알 수 없으면 "3D 뷰어에서 코치와 비교해 확인해 보세요" 처럼 확인을 안내한다.
- 말투는 "~해요" 체. 친절하고 짧게, 전문 용어는 줄인다.
- summary: 전체 평가 1~2문장.
- items: 고칠 점이 있는 구간마다 하나씩, segmentId 에 그 구간 id 를 넣는다. 각 2문장 이내로, 무엇이 달랐는지와 어떻게 연습하면 좋은지를 쓴다. 문제없는 구간은 쓰지 않는다. 전체에 대한 조언이 필요하면 segmentId 없이 하나까지 쓸 수 있다.
- 각도는 정수로 반올림해 말한다. 15도 미만 차이는 "조금" 다르다고 표현한다.
- 점수, 의학적 진단, 부상 위험 단정은 말하지 않는다.
"""


def old_json(payload):
    """The payload as production sent it before 2026-11-12: no poseMetrics, no deviation direction, no userMotion."""
    p = copy.deepcopy(payload)
    p.pop('poseMetrics', None)
    for s in p['segments']:
        s.pop('userMotion', None)
        for d in s.get('deviations', []):
            d.pop('direction', None)
            d.pop('detail', None)
    return p


def call(payload, prompt, model, log):
    from google.genai import types
    from google import genai
    import os
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
    for attempt in range(6):
        t0 = time.time()
        try:
            resp = client.models.generate_content(
                model=model, contents=json.dumps(payload, ensure_ascii=False),
                config=types.GenerateContentConfig(system_instruction=prompt, response_mime_type='application/json',
                                                   response_schema=feedback.Feedback, temperature=0.3,
                                                   http_options=types.HttpOptions(timeout=90_000)))
            fb = resp.parsed if isinstance(resp.parsed, feedback.Feedback) else \
                feedback.Feedback.model_validate_json(resp.text)
            return fb, round(time.time() - t0, 1), resp.usage_metadata.prompt_token_count
        except Exception as e:                                  # noqa: BLE001
            log(f'    try {attempt + 1}: {type(e).__name__} {str(e)[:80]}')
            time.sleep(20 * (attempt + 1))
    raise RuntimeError('no answer')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', action='append', required=True)
    ap.add_argument('--model', default='gemini-3.5-flash')
    a = ap.parse_args()
    rows = []
    for pair in a.pair:
        spec = pair_spec(pair)
        r, s = build_result(spec['job'], spec['exercise'], spec['retarget'], with_score=True)
        new = feedback.llm_input(r, s)
        jsons = dict(old=old_json(new), new=new)
        prompts = dict(old=OLD_PROMPT, new=feedback.SYSTEM_PROMPT)
        out = OUT / pair.replace(':', '_')
        out.mkdir(parents=True, exist_ok=True)
        for pk in ('old', 'new'):
            for jk in ('old', 'new'):
                print(f'{pair} prompt={pk} json={jk}')
                fb, sec, tok = call(jsons[jk], prompts[pk], a.model, print)
                fbd = feedback.clean(fb, {x['id'] for x in r['segments']})
                text = fbd['summary'] + ' ' + ' '.join(i['text'] for i in fbd['items'])
                row = dict(pair=pair, prompt=pk, json=jk, model=a.model, seconds=sec, input_tokens=tok,
                           raw_items=len(fb.items), feedback=fbd, numbers=numbers_cited(text))
                (out / f'{pk}_{jk}.json').write_text(json.dumps(row, ensure_ascii=False, indent=1))
                print(f'  -> {sec}s, {tok} input tokens, {len(fb.items)} items, {len(row["numbers"])} numbers')
                rows.append(row)
    lines = ['# prompt x JSON ablation', '']
    for r_ in rows:
        lines += [f'## {r_["pair"]} | prompt {r_["prompt"]} | json {r_["json"]} '
                  f'({r_["seconds"]} s, {r_["input_tokens"]} tokens, model items {r_["raw_items"]})', '',
                  f'**summary** {r_["feedback"]["summary"]}', '']
        lines += [f'- `{i.get("segmentId", "-")}` {i["text"]}' for i in r_['feedback']['items']]
        lines += ['', f'numbers: {", ".join(r_["numbers"]) or "none"}', '']
    (OUT / 'report.md').write_text('\n'.join(lines))
    print('wrote', OUT / 'report.md')


if __name__ == '__main__':
    main()
