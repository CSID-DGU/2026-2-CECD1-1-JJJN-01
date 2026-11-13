"""API test against the frontend contract (frontend/src/types/analysis.ts), no browser needed.

  cd backend && ../.venv/bin/python tests/test_api.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os  # noqa: E402
os.environ.setdefault('ANALYSIS_RUNNER', 'demo')     # contract test: no GPU work
from fastapi.testclient import TestClient  # noqa: E402

from app import config  # noqa: E402
from app.main import app  # noqa: E402

VIDEO = config.PIPELINE_REPO / 'pipeline/data/child_squat.mp4'
VERDICTS = {'match', 'missing', 'extra', 'substitute'}
TIMING = {'ok', 'slow', 'fast'}
FRONT_BONES = {'torso', 'head', 'left_clavicle', 'right_clavicle', 'left_upper_arm', 'right_upper_arm',
               'left_forearm', 'right_forearm', 'left_pelvis', 'right_pelvis', 'left_thigh', 'right_thigh',
               'left_shin', 'right_shin'}
NEEDED_JOINTS = {'nose', 'neck', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist',
                 'right_wrist', 'pelvis', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
                 'right_ankle'}


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def check_sequence(s, name):
    assert set(s) >= {'jointNames', 'fps', 'frames'}, name
    assert NEEDED_JOINTS <= set(s['jointNames']), f'{name}: joints missing {NEEDED_JOINTS - set(s["jointNames"])}'
    assert num(s['fps']) and s['fps'] > 0
    J = len(s['jointNames'])
    for f in s['frames']:
        assert len(f) == J and all(len(p) == 3 and all(num(v) for v in p) for p in f), name


DIRECTIONS = {'up', 'down', 'forward', 'backward', 'outward', 'inward', 'left', 'right'}


def check_pose(p):
    for k in ('kneeFlexDeg', 'hipFlexDeg'):
        assert num(p[k]['left']) and num(p[k]['right']), (k, p)
    for k in ('trunkLeanDeg', 'hipKneeCm', 'pelvisDropCm', 'kneeWidthRatio'):
        assert num(p[k]), (k, p)


def check_pose_metrics(pm, seg_ids):
    assert pm['repCount']['user'] >= 0 and pm['repCount']['coach'] >= 0
    for who in ('user', 'coach', 'diff'):
        check_pose(pm['standing'][who])
    assert len(pm['reps']) >= max(pm['repCount']['user'], pm['repCount']['coach'])
    for rep in pm['reps']:
        b = rep['bottom']
        assert b['user'] or b['coach'], rep
        for who in ('user', 'coach', 'diff'):
            if b.get(who):
                check_pose(b[who])
        assert rep['segmentId'] is None or rep['segmentId'] in seg_ids, rep


def check_result(r, job_id):
    assert r['jobId'] == job_id and r['exercise'] == 'squat'
    assert num(r['score']['total']) and 0 <= r['score']['total'] <= 100
    for s in r['segments']:
        assert isinstance(s['id'], str) and isinstance(s['label'], str) and s['verdict'] in VERDICTS
        for k in ('studentRange', 'coachRange'):
            if k in s:
                a, b = s[k]
                assert num(a) and num(b) and a <= b, (k, s)
        if 'timing' in s:
            assert s['verdict'] == 'match' and s['timing']['label'] in TIMING and num(s['timing']['ratio'])
        for dv in s['deviations']:
            assert dv['bone'] in FRONT_BONES and num(dv['angleDiffDeg']), dv
            if 'direction' in dv:                         # pose_metrics.signed_deviations
                assert dv['direction'] in DIRECTIONS and all(num(v) for v in dv['components'].values()), dv
    starts = [s['studentRange'][0] for s in r['segments'] if 'studentRange' in s]
    check_pose_metrics(r['poseMetrics'], {s['id'] for s in r['segments']})
    assert starts == sorted(starts), 'segments not in time order'
    check_sequence(r['student'], 'student')
    check_sequence(r['coachRetargeted'], 'coachRetargeted')
    nS, nC = len(r['student']['frames']), len(r['coachRetargeted']['frames'])
    fm = r['frameMap']                                    # frame_map.py: one coach position per user frame
    assert len(fm['coachFrame']) == nS and len(fm['kind']) == nS
    assert all(0 <= c <= nC - 1 for c in fm['coachFrame']), 'frameMap out of range'
    assert all(b >= a - 1e-6 for a, b in zip(fm['coachFrame'], fm['coachFrame'][1:])), 'frameMap goes back'
    assert set(fm['kind']) <= {'together', 'coachAhead', 'coachWaits', 'userOnly', 'outside'}
    for s_, c_ in r['alignment']:
        assert 0 <= s_ < nS and 0 <= c_ < nC
    if 'feedback' in r:                                   # llm stage (optional)
        fb, ids = r['feedback'], {s['id'] for s in r['segments']}
        assert isinstance(fb['summary'], str) and fb['summary'].strip()
        for it in fb['items']:
            assert isinstance(it['text'], str) and it['text'].strip()
            assert 'segmentId' not in it or it['segmentId'] in ids, it


def main():
    c = TestClient(app)
    # rejections
    bad = c.post('/analysis', data={'exercise': 'deadlift', 'coachId': 'squat-diag'},
                 files={'video': ('a.mp4', VIDEO.read_bytes(), 'video/mp4')})
    assert bad.status_code == 400 and 'message' in bad.json(), bad.text
    bad = c.post('/analysis', data={'exercise': 'squat', 'coachId': 'nope'},
                 files={'video': ('a.mp4', VIDEO.read_bytes(), 'video/mp4')})
    assert bad.status_code == 400, bad.text
    bad = c.post('/analysis', data={'exercise': 'squat', 'coachId': 'squat-diag'},
                 files={'video': ('a.txt', b'not a video', 'text/plain')})
    assert bad.status_code == 400, bad.text
    bad = c.post('/analysis', data={'exercise': 'squat', 'coachId': 'squat-diag'},
                 files={'video': ('a.mp4', b'not a video', 'video/mp4')})
    assert bad.status_code == 400 and '읽을 수 없' in bad.json()['message'], bad.text
    assert c.get('/analysis/unknown').status_code == 404
    print('rejections ok:', bad.json()['message'])

    # happy path
    r = c.post('/analysis', data={'exercise': 'squat', 'coachId': 'squat-diag'},
               files={'video': ('child.mp4', VIDEO.read_bytes(), 'video/mp4')},
               headers={'Origin': 'http://localhost:5173'})
    assert r.status_code == 202, r.text
    assert r.headers.get('access-control-allow-origin') == '*', 'CORS header missing'
    job_id = r.json()['jobId']
    seen = []
    for _ in range(100):
        j = c.get(f'/analysis/{job_id}').json()
        if not seen or seen[-1] != j['status']:
            seen.append(j['status'])
        if j['status'] in ('done', 'failed'):
            break
        time.sleep(0.3)
    print('statuses:', ' -> '.join(seen))
    assert j['status'] == 'done', j
    check_result(j['result'], job_id)
    print('result ok:', len(j['result']['segments']), 'segments,', len(j['result']['student']['frames']),
          'student frames,', len(j['result']['alignment']), 'alignment pairs')


if __name__ == '__main__':
    main()
