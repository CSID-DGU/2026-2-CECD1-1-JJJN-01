"""LLM feedback experiment: what does the algorithm JSON add over just showing the model the videos?

Conditions (same model, same output schema as production `Feedback {summary, items[{segmentId?, text}]}`):
  A  video          coach video + user video only
  B  video_json     both videos + the algorithm JSON (the same llm_input payload production sends)
  C  json_prod      JSON only with the production SYSTEM_PROMPT (= what the app shows today)
  D  json_numbers   JSON only, production prompt + rules that require citing the measured numbers

Pairs:
  job:<jobId>   backend/data/jobs/<jobId>: coach.mp4 (uploaded coach) or the library coach video, user.mp4,
                retarget/ (JSON is rebuilt with build_result, same code path as the API)
  demo          pipeline/data/squat_diag.mp4 -> student_squat_h264.mp4 (runs/coach_user/coach_to_student)

Run from backend/:
  ../.venv/bin/python -m experiments.llm_compare --pair job:e80464dd6496 --pair demo
Outputs (gitignored): data/experiments/<pair>/{input.json, <cond>[_<k>].json, upload_*.mp4} and data/experiments/report.md

A and B upload both videos to the Gemini Files API (deleted right after the run; free-tier content may be
used by Google to improve its products).
"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from app import config, feedback
from app.result_builder import build_result

OUT = config.DATA_DIR / 'experiments'
FIELD_DOC = feedback.SYSTEM_PROMPT.split('입력 설명\n')[1].split('\n규칙')[0].strip()

COMMON_RULES = """- 말투는 "~해요" 체. 친절하고 짧게, 전문 용어는 줄인다.
- summary: 전체 평가 1~2문장.
- items: 고칠 점마다 하나씩, 각 2문장 이내로 무엇이 달랐는지와 어떻게 연습하면 좋은지를 쓴다. 최대 6개.
- 점수, 의학적 진단, 부상 위험 단정은 말하지 않는다."""

PROMPT_A = f"""너는 운동 동작을 봐 주는 코치다. 첫 번째 영상은 코치의 시범, 두 번째 영상은 사용자가 같은 운동을 따라 한 영상이다. 두 영상을 비교해 앱 화면에 보여 줄 피드백을 한국어로 쓴다.

규칙
- 영상에서 실제로 보이는 것만 쓴다. 잘 보이지 않으면 단정하지 않는다.
- 가능하면 구체적으로 쓴다: 어느 부위가 어떻게 다른지(각도, 방향), 언제인지(사용자 영상 몇 초 무렵, 몇 번째 반복), 속도가 코치보다 빠른지 느린지.
- segmentId 는 비워 둔다.
{COMMON_RULES}
"""

PROMPT_B = f"""너는 운동 동작을 봐 주는 코치다. 첫 번째 영상은 코치의 시범, 두 번째 영상은 사용자가 같은 운동을 따라 한 영상이다. 함께 주는 JSON 은 두 영상에서 3D 골격을 뽑고, 코치 동작을 사용자 체형에 맞춘 뒤 사용자와 비교한 측정 결과다. 영상과 JSON 을 함께 보고 앱 화면에 보여 줄 피드백을 한국어로 쓴다.

JSON 설명
{FIELD_DOC}

규칙
- 수치(각도 차이, 구간 시간, 속도)는 JSON 을 따른다. 영상을 보고 새 수치를 만들지 않는다.
- JSON 에는 차이의 방향이 없다. 방향(더 굽혔다, 팔이 더 높다 등)은 영상에서 분명히 보일 때만 쓰고, 애매하면 쓰지 않는다.
- 측정값이 영상과 맞지 않아 보이면(예: 영상에서는 비슷한데 각도 차이가 크다) 단정하지 말고 그렇게 보인다고 짧게 덧붙인다.
- 각 항목에 근거 수치를 넣는다. 예: "내려갈 때 오른쪽 아래팔 각도가 코치와 40도 달라요", "올라오는 데 0.87초 걸렸고 코치는 1.3초였어요".
- 같은 차이가 여러 반복에서 되풀이되면 한 항목으로 묶고 대표 구간 id 하나를 segmentId 에 넣는다.
{COMMON_RULES}
"""

PROMPT_D = feedback.SYSTEM_PROMPT + """
추가 규칙 (위 규칙과 겹치면 이쪽을 따른다)
- 각 항목에 근거 수치를 넣는다. 예: "내려갈 때 오른쪽 아래팔 각도가 코치와 40도 달라요", "올라오는 데 0.87초 걸렸고 코치는 1.3초였어요". userSec / coachSec 같은 절대 시간은 그대로 말해도 된다.
- 같은 차이가 여러 반복에서 되풀이되면 한 항목으로 묶어 범위로 말하고(예: 38~46도), 대표 구간 id 하나를 segmentId 에 넣는다.
- items 는 최대 6개.
"""

CONDS = {
    'A': dict(name='video', prompt=PROMPT_A, video=True, json=False),
    'B': dict(name='video_json', prompt=PROMPT_B, video=True, json=True),
    'C': dict(name='json_prod', prompt=feedback.SYSTEM_PROMPT, video=False, json=True),
    'D': dict(name='json_numbers', prompt=PROMPT_D, video=False, json=True),
}
RUN_OFFSET = 0                                                # --run-offset: number repeat runs after earlier ones
RETRY = ('503', '504', '429', 'UNAVAILABLE', 'DEADLINE', 'RESOURCE_EXHAUSTED', 'timed out', 'Timeout')


def pair_spec(pair):
    if pair == 'demo':
        data = config.PIPELINE_REPO / 'pipeline/data'
        return dict(coach=data / 'squat_diag.mp4', user=data / 'student_squat_h264.mp4',
                    retarget=config.DEMO_RETARGET_DIR, exercise='squat', job='demo')
    job = pair.split(':', 1)[1]
    d = config.JOBS_DIR / job
    meta = json.loads((d / 'job.json').read_text())
    coach = d / 'coach.mp4'
    if not coach.exists():                                   # library coach: video recorded in its run report
        run = config.COACHES[meta['coachId']]['run']
        coach = Path(json.loads((run / 'report.json').read_text())['video_original'])
    return dict(coach=coach, user=d / 'user.mp4', retarget=d / 'retarget', exercise=meta['exercise'], job=job)


def small_video(src, dst):
    """720 px long side, 30 fps, no audio: faster upload, same content."""
    if not dst.exists():
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(src), '-vf',
                        "scale='if(gt(iw,ih),720,-2)':'if(gt(iw,ih),-2,720)',fps=30", '-c:v', 'libx264',
                        '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p', '-an', str(dst)], check=True)
    return dst


def upload(client, path):
    f = client.files.upload(file=str(path), config=dict(mime_type='video/mp4', display_name=path.name))
    while f.state.name == 'PROCESSING':
        time.sleep(2)
        f = client.files.get(name=f.name)
    if f.state.name != 'ACTIVE':
        raise RuntimeError(f'upload {path.name}: {f.state}')
    return f


def call(client, model, prompt, contents, timeout_sec):
    from google.genai import types
    t0 = time.time()
    resp = client.models.generate_content(
        model=model, contents=contents,
        config=types.GenerateContentConfig(system_instruction=prompt, response_mime_type='application/json',
                                           response_schema=feedback.Feedback, temperature=0.3,
                                           http_options=types.HttpOptions(timeout=int(timeout_sec * 1000))))
    fb = resp.parsed if isinstance(resp.parsed, feedback.Feedback) else feedback.Feedback.model_validate_json(resp.text)
    u = resp.usage_metadata
    return fb, dict(seconds=round(time.time() - t0, 1), prompt_tokens=u.prompt_token_count,
                    output_tokens=u.candidates_token_count)


def call_retry(client, models, prompt, contents, timeout_sec, log):
    for model in models:
        for attempt in range(6):
            try:
                fb, meta = call(client, model, prompt, contents, timeout_sec)
                return fb, dict(model=model, **meta)
            except Exception as e:                         # noqa: BLE001
                msg = f'{type(e).__name__}: {str(e)[:200]}'
                log(f'    {model} try {attempt + 1}: {msg}')
                if not any(k in msg for k in RETRY):
                    break
                time.sleep(15 * (attempt + 1))
    raise RuntimeError('all models failed')


def numbers_cited(text):
    return re.findall(r'\d+(?:\.\d+)?\s*(?:~\s*\d+(?:\.\d+)?\s*)?(?:도|초|배|%)', text)


def run_pair(client, pair, conds, models, fps, repeat, log):
    from google.genai import types
    spec = pair_spec(pair)
    out = OUT / pair.replace(':', '_')
    out.mkdir(parents=True, exist_ok=True)
    result, score = build_result(spec['job'], spec['exercise'], spec['retarget'], with_score=True)
    payload = feedback.llm_input(result, score)
    (out / 'input.json').write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    seg_ids = {s['id'] for s in result['segments']}
    files = []
    try:
        if any(CONDS[c]['video'] for c in conds):
            vids = [small_video(spec['coach'], out / 'upload_coach.mp4'), small_video(spec['user'], out / 'upload_user.mp4')]
            files = [upload(client, v) for v in vids]
            vparts = [types.Part(file_data=types.FileData(file_uri=f.uri, mime_type=f.mime_type),
                                 video_metadata=types.VideoMetadata(fps=fps)) for f in files]
        rows = []
        for c in conds:
            cd = CONDS[c]
            data = json.dumps(payload, ensure_ascii=False)
            if cd['video']:
                contents = [types.Part(text='[코치 시범 영상]'), vparts[0], types.Part(text='[사용자 영상]'), vparts[1]]
                if cd['json']:
                    contents.append(types.Part(text='[측정 결과 JSON]\n' + data))
            else:
                contents = data                              # exactly what production sends
            for k in range(repeat):
                log(f'  {pair} {c}:{cd["name"]} run {k + 1 + RUN_OFFSET}')
                fb, meta = call_retry(client, models, cd['prompt'], contents, 180 if cd['video'] else 60, log)
                fbd = feedback.clean(fb, seg_ids)
                text = fbd['summary'] + ' ' + ' '.join(i['text'] for i in fbd['items'])
                row = dict(pair=pair, cond=c, name=cd['name'], run=k + 1 + RUN_OFFSET, **meta, feedback=fbd,
                           numbers=numbers_cited(text), n_items=len(fbd['items']))
                (out / f'{c}_{cd["name"]}{"_" + str(k + 1 + RUN_OFFSET) if repeat > 1 or RUN_OFFSET else ""}.json').write_text(
                    json.dumps(row, ensure_ascii=False, indent=1))
                log(f'    -> {meta["model"]} {meta["seconds"]}s, tokens in {meta["prompt_tokens"]}, '
                    f'{len(fbd["items"])} items, numbers {row["numbers"]}')
                rows.append(row)
        return rows
    finally:
        for f in files:
            try:
                client.files.delete(name=f.name)
            except Exception:                              # noqa: BLE001
                pass


def report(rows):
    lines = ['# LLM feedback experiment', '']
    for pair in dict.fromkeys(r['pair'] for r in rows):
        lines += [f'## {pair}', '']
        for r in (r for r in rows if r['pair'] == pair):
            lines += [f'### {r["cond"]} {r["name"]} (run {r["run"]}) - {r["model"]}, {r["seconds"]} s, '
                      f'input {r["prompt_tokens"]} tokens', '', f'**summary** {r["feedback"]["summary"]}', '']
            lines += [f'- `{i.get("segmentId", "-")}` {i["text"]}' for i in r['feedback']['items']]
            lines += ['', f'numbers cited: {", ".join(r["numbers"]) or "none"}', '']
    return '\n'.join(lines)


def main():
    import os
    from google import genai
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', action='append', required=True, help='job:<jobId> or demo (repeatable)')
    ap.add_argument('--conds', default='ABCD')
    ap.add_argument('--models', default='gemini-3.5-flash,gemini-3.8-flash', help='first is used unless it keeps failing')
    ap.add_argument('--fps', type=float, default=5., help='video frames per second the model samples (API default 1)')
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--run-offset', type=int, default=0, help='e.g. 1 to add runs 2..n next to an earlier run 1')
    a = ap.parse_args()
    global RUN_OFFSET
    RUN_OFFSET = a.run_offset
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
    rows = []
    for pair in a.pair:
        rows += run_pair(client, pair, list(a.conds), a.models.split(','), a.fps, a.repeat, print)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob('*/[A-D]_*.json'))]   # every run so far
    (OUT / 'report.md').write_text(report(sorted(rows, key=lambda r: (r['pair'], r['cond'], r['run']))))
    print(f'wrote {OUT / "report.md"}')


if __name__ == '__main__':
    main()
