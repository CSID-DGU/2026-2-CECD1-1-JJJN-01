"""API traffic log: what the frontend sent and what the backend answered.

  data/api.log                         every request, one JSON line each (rejected ones included)
  data/jobs/<jobId>/logs/api.log       the lines of that job
  data/jobs/<jobId>/api_response.json  the exact GET response the frontend received when the job was done

POST: the form fields and the uploaded files' metadata (file name, type, bytes, duration); never the video.
GET:  the frontend polls every second or so, so a job is logged once per status, and when it is done the line
      summarises what the result holds (segments, skeleton frames, alignment pairs, poseMetrics, feedback).
Each line is also echoed, shortened, on the server console (uvicorn log).
"""
import json
import logging
import threading
import time
from collections import Counter
from pathlib import Path

from . import config

console = logging.getLogger('uvicorn.error')
_lock = threading.Lock()
_last_status: dict[str, str] = {}


def _write(entry, job_dir=None):
    entry = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), **entry}
    line = json.dumps(entry, ensure_ascii=False)
    with _lock:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        with (config.DATA_DIR / 'api.log').open('a') as f:
            f.write(line + '\n')
        if job_dir:
            logs = Path(job_dir) / 'logs'
            logs.mkdir(parents=True, exist_ok=True)
            with (logs / 'api.log').open('a') as f:
                f.write(line + '\n')


def file_info(upload):
    return None if upload is None else {'filename': upload.filename, 'contentType': upload.content_type}


def post(sent, response, status_code, job_dir=None, client=None):
    _write(dict(event='POST /analysis', client=client, request=sent, status=status_code, response=response), job_dir)
    v = sent.get('video') or {}
    coach = sent.get('coachId') or ((sent.get('coachVideo') or {}).get('filename'))
    size = f"{v['bytes'] / 1e6:.1f}MB, {v['durationSec']:.1f}s" if 'bytes' in v and 'durationSec' in v else '-'
    console.info(f"api POST {status_code} {response} | exercise={sent.get('exercise')} video={v.get('filename')} "
                 f"({size}) coach={coach}")


def rejected(path, message, status_code, client=None):
    _write(dict(event=path, client=client, status=status_code, response={'message': message}))
    console.info(f'api {path} {status_code} {message}')


def summarize(result):
    """What a done result holds, without the bulky arrays."""
    segs = result.get('segments', [])
    pm = result.get('poseMetrics') or {}
    fb = result.get('feedback')
    al = result.get('alignment', [])
    seq = lambda s: dict(frames=len(s['frames']), joints=len(s['jointNames']), fps=s['fps']) if s else None
    return dict(
        bytes=len(json.dumps(result, ensure_ascii=False).encode()),
        fields=list(result.keys()),
        score=result.get('score'),
        segments=dict(
            count=len(segs),
            verdicts=dict(Counter(s['verdict'] for s in segs)),
            timingNotOk=[f"{s['id']} {s['label']} {s['timing']['label']} {s['timing']['ratio']}" for s in segs
                         if (s.get('timing') or {}).get('label', 'ok') != 'ok'],
            withDeviations=sum(bool(s['deviations']) for s in segs),
            deviationsWithDirection=sum('direction' in d for s in segs for d in s['deviations']),
            withHold=sum('hold' in s for s in segs)),
        student=seq(result.get('student')),
        coachRetargeted=seq(result.get('coachRetargeted')),
        alignment=dict(pairs=len(al), userFramesCovered=len({p[0] for p in al}), first=al[:3]),
        frameMap=None if not result.get('frameMap') else dict(
            frames=len(result['frameMap']['coachFrame']), kinds=dict(Counter(result['frameMap']['kind'])),
            first=result['frameMap']['coachFrame'][:5]),
        poseMetrics=None if not pm else dict(repCount=pm.get('repCount'), legLengthCm=pm.get('legLengthCm'),
                                             thresholds=pm.get('thresholds'), reps=len(pm.get('reps', []))),
        feedback=None if not fb else dict(summary=fb['summary'], items=len(fb['items'])))


def get(job, response, client=None):
    """Log a poll only when the job's status changed; on done also keep the exact response."""
    status = response['status']
    with _lock:
        if _last_status.get(job.jobId) == status:
            return
        _last_status[job.jobId] = status
    entry = dict(event='GET /analysis', client=client, jobId=job.jobId, status=status, progress=response['progress'])
    if status == 'failed':
        entry['error'] = response.get('error')
    if status == 'done':
        s = summarize(response['result'])
        entry['result'] = s
        (Path(job.dir) / 'api_response.json').write_text(json.dumps(response, ensure_ascii=False))
    _write(entry, job.dir)
    if status == 'done':
        sg, pm = s['segments'], s['poseMetrics'] or {}
        console.info(f"api GET {job.jobId} done | segments {sg['count']} {sg['verdicts']}, frames "
                     f"{(s['student'] or {}).get('frames')}/{(s['coachRetargeted'] or {}).get('frames')}, alignment "
                     f"{s['alignment']['pairs']}, reps {pm.get('repCount')}, feedback "
                     f"{(s['feedback'] or {}).get('items', 0)} items, {s['bytes'] / 1000:.0f} KB")
    else:
        console.info(f"api GET {job.jobId} {status} {response['progress']}" + (f" {entry.get('error')}" if status == 'failed' else ''))
