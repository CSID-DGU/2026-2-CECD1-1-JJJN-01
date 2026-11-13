"""End-to-end with the real pipeline (GPU, a few minutes per video).

  cd backend && ../.venv/bin/python tests/test_pipeline_e2e.py [VIDEO ...]
Default: child_squat.mp4 (should finish) and a generated video without a person (should fail
with the user-facing message).
"""
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['ANALYSIS_RUNNER'] = 'pipeline'
from fastapi.testclient import TestClient  # noqa: E402

from app import config  # noqa: E402
from app.main import app  # noqa: E402
from tests.test_api import check_result  # noqa: E402


def run(c, video, timeout=1500):
    t0 = time.time()
    r = c.post('/analysis', data={'exercise': 'squat', 'coachId': 'squat-diag'},
               files={'video': (video.name, video.read_bytes(), 'video/mp4')})
    assert r.status_code == 202, r.text
    job_id, seen = r.json()['jobId'], []
    while time.time() - t0 < timeout:
        j = c.get(f'/analysis/{job_id}').json()
        if not seen or seen[-1][0] != j['status']:
            seen.append((j['status'], round(time.time() - t0)))
        if j['status'] in ('done', 'failed'):
            break
        time.sleep(1)
    print(f'{video.name}: ' + ' -> '.join(f'{s}@{t}s' for s, t in seen))
    return job_id, j


def main():
    videos = [Path(v) for v in sys.argv[1:]]
    blank = Path(config.DATA_DIR) / 'test_no_person.mp4'
    if not videos:
        if not blank.exists():
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc=size=1280x720:rate=30',
                            '-t', '4', '-pix_fmt', 'yuv420p', str(blank)], check=True)
        videos = [config.PIPELINE_REPO / 'pipeline/data/child_squat.mp4', blank]
    c = TestClient(app)
    for v in videos:
        job_id, j = run(c, v)
        if j['status'] == 'done':
            check_result(j['result'], job_id)
            r = j['result']
            print('  ok:', len(r['segments']), 'segments', [s['label'] + '/' + s['verdict'] for s in r['segments']],
                  len(r['student']['frames']), 'student frames')
            print('  feedback:', r.get('feedback', '(none)'))
        else:
            print('  failed:', j.get('error'))


if __name__ == '__main__':
    main()
