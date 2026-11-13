"""Stage 2 runner: the real analysis for one job (runs in the JobStore worker thread).

  normalize   ffmpeg -> H.264, 30 fps, rotation applied, long side <= 1920   (user.mp4 / coach.mp4)
  extracting  pipeline.run_video (WHAM -> IK) on the user video (and the uploaded coach video)
  retargeting pipeline.coach_to_user (PAN): coach motion on the user's body
  scoring     motion_scoring + conversion to the frontend result (result_builder)
  llm         Gemini feedback (feedback.py); a failure only leaves `feedback` out

Everything a job produces stays in data/jobs/<jobId>/:
  upload_user.<ext> (upload), user.mp4 (normalized), user_run/ (run_video), retarget/ (coach_to_user),
  logs/<step>.log (pipeline output), result.json, job.json, error.log on unexpected errors.
Pipeline steps run as subprocesses of the project venv (WHAM itself runs in .venv-wham).
"""
import re
import subprocess
import threading
import time
from pathlib import Path

from . import config
from .jobs import UserFacingError
from .feedback import generate_feedback
from .result_builder import build_result

MSG_NO_PERSON = '영상에서 사람을 찾지 못했어요. 전신이 3초 이상 나오게 다시 찍어 주세요.'
MSG_SHORT_TRACK = '영상에서 사람이 보이는 시간이 너무 짧아요. 처음부터 끝까지 전신이 나오게 다시 찍어 주세요.'
MSG_BAD_VIDEO = '영상 파일을 읽을 수 없어요. 다른 영상으로 다시 시도해 주세요.'
MSG_TIMEOUT = '분석 시간이 너무 오래 걸려서 중단했어요. 더 짧은 영상으로 다시 시도해 주세요.'
MSG_BUSY = '서버가 잠시 바빠요. 잠시 후 다시 시도해 주세요.'

# known pipeline failures (regex on the step log) -> message for the user
FAILURES = [
    (r'WHAM found no person|No foot joints observed|Feet never observed'
     r'|index -1 is out of bounds for axis 0 with size 0', MSG_NO_PERSON),
    (r'Person tracked in only', MSG_SHORT_TRACK),
    (r'Cannot determine facing direction', MSG_NO_PERSON),
    (r'CUDA out of memory|OutOfMemoryError', MSG_BUSY),
]

TIMEOUT = dict(normalize=300, extract=1200, retarget=600)


class StepFailed(Exception):
    def __init__(self, step, log):
        super().__init__(f'{step} failed, see {log}')
        self.step, self.log = step, log


def run_step(step, cmd, logs, on_line=None, timeout=600):
    """Run a pipeline step, stream its output to logs/<step>.log, raise UserFacingError on known failures."""
    logs.mkdir(exist_ok=True)
    log = logs / f'{step}.log'
    with log.open('w') as f:
        f.write('$ ' + ' '.join(map(str, cmd)) + '\n')
        f.flush()
        p = subprocess.Popen([str(c) for c in cmd], cwd=config.PIPELINE_REPO, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        timer = threading.Timer(timeout, p.kill)
        timer.start()
        started = time.time()
        try:
            for line in p.stdout:
                f.write(line)
                f.flush()
                if on_line:
                    on_line(line)
            p.wait()
        finally:
            timer.cancel()
    if p.returncode != 0:
        if time.time() - started >= timeout - 1:
            raise UserFacingError(MSG_TIMEOUT)
        text = log.read_text(errors='replace')
        for pattern, message in FAILURES:
            if re.search(pattern, text):
                raise UserFacingError(message)
        raise StepFailed(step, log)


def normalize(src, dst, logs):
    """H.264 30 fps copy with the rotation metadata applied (phone videos), long side <= 1920."""
    try:
        run_step(f'normalize_{dst.stem}', ['ffmpeg', '-v', 'error', '-y', '-i', src, '-vf',
                                           'fps=30,scale=1920:1920:force_original_aspect_ratio=decrease'
                                           ':force_divisible_by=2',
                                           '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
                                           '-pix_fmt', 'yuv420p', '-an', dst],
                 logs, timeout=TIMEOUT['normalize'])
    except StepFailed:
        raise UserFacingError(MSG_BAD_VIDEO)
    return dst


def extract(video, out, logs, on_progress, ground='frame'):
    """run_video (WHAM -> IK). on_progress(fraction 0..1 of this step) from the step markers."""
    marks = {'[S0] wham': 0.05, '[S1]': 0.75, '[S2]': 0.8, 'Wrote': 1.0}

    def on_line(line):
        for key, frac in marks.items():
            if line.startswith(key):
                on_progress(frac)

    run_step(f'extract_{out.name}', [config.PIPELINE_PYTHON, '-m', 'pipeline.run_video', video,
                                     '--estimator', 'wham', '--ground', ground, '--targets', '--out', out],
             logs, on_line, TIMEOUT['extract'])
    if not (out / 'source_person.bvh').exists():
        raise StepFailed('extract', logs / f'extract_{out.name}.log')
    return out


def pipeline_runner(job, set_status):
    d = Path(job.dir)
    logs = d / 'logs'
    # progress budget: extraction dominates (WHAM); an uploaded coach video doubles it
    spans = ([('coach', 0.0, 0.4), ('user', 0.4, 0.8)] if job.coachVideo else [('user', 0.0, 0.8)])

    if job.coachVideo:
        coach_run = None
    else:
        coach_run = config.COACHES[job.coachId]['run']
        if not (coach_run / 'source_person.bvh').exists():
            raise RuntimeError(f'coach run missing: {coach_run}')

    runs = {}
    for who, a, b in spans:
        set_status('extracting', a)
        src = Path(job.coachVideo if who == 'coach' else job.video)
        video = normalize(src, d / f'{who}.mp4', logs)
        runs[who] = extract(video, d / f'{who}_run', logs,
                            lambda f, a=a, b=b: set_status('extracting', a + (b - a) * f))
    coach_run = runs.get('coach', coach_run)

    set_status('retargeting', 0.8)
    retarget = d / 'retarget'
    run_step('retarget', [config.PIPELINE_PYTHON, '-m', 'pipeline.coach_to_user', '--coach', coach_run,
                          '--user', runs['user'], '--out', retarget, '--no-viz'], logs, timeout=TIMEOUT['retarget'])

    set_status('scoring', 0.9)
    result, score = build_result(job.jobId, job.exercise, retarget, with_score=True)

    set_status('llm', 0.95)
    feedback = generate_feedback(result, score, logs)          # None: no feedback (see logs/llm.log)
    if feedback:
        result['feedback'] = feedback
    return result
