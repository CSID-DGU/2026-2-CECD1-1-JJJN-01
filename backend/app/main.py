"""HTTP API for the frontend (contract: frontend/src/types/analysis.ts, docs/backend-api-guide.md).

  POST /analysis          multipart: video, exercise, coachId | coachVideo  -> { jobId }
  GET  /analysis/{jobId}  -> { jobId, status, progress, result? , error? }
  GET  /health

Errors are JSON { "message": "<Korean sentence for the user>" }.
Requests and responses are logged by api_log.py (data/api.log, data/jobs/<jobId>/logs/api.log).

  cd backend
  ../.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import api_log, config
from .jobs import Job, JobStore, demo_runner
from .pipeline_runner import pipeline_runner

app = FastAPI(title='motion analysis API')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
# ANALYSIS_RUNNER=demo: no analysis, every job returns config.DEMO_RETARGET_DIR (frontend work)
RUNNER = os.environ.get('ANALYSIS_RUNNER', 'pipeline')
store = JobStore(demo_runner if RUNNER == 'demo' else pipeline_runner)


class BadRequest(Exception):
    def __init__(self, message, status=400):
        self.message, self.status = message, status


@app.exception_handler(BadRequest)
async def _bad_request(_, e: BadRequest):
    return JSONResponse({'message': e.message}, status_code=e.status)


@app.exception_handler(RequestValidationError)
async def _validation(request: Request, e: RequestValidationError):
    missing = [str(err['loc'][-1]) for err in e.errors()]
    message = f'요청에 빠진 항목이 있어요: {", ".join(missing)}'
    api_log.rejected(f'{request.method} {request.url.path}', message, 400, request.client.host if request.client else None)
    return JSONResponse({'message': message}, status_code=400)


async def save_video(upload: UploadFile, dst_stem: Path, info_out: dict | None = None) -> Path:
    ext = Path(upload.filename or '').suffix.lower()
    if ext not in config.VIDEO_EXT and upload.content_type not in config.VIDEO_MIME:
        raise BadRequest('mp4, mov, webm 영상만 올릴 수 있어요.')
    dst = dst_stem.with_suffix(ext if ext in config.VIDEO_EXT else '.mp4')
    size = 0
    with dst.open('wb') as f:
        while chunk := await upload.read(1 << 20):
            size += len(chunk)
            if size > config.MAX_BYTES:
                f.close()
                dst.unlink(missing_ok=True)
                raise BadRequest('영상은 100MB 이하만 올릴 수 있어요.')
            f.write(chunk)
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                            'format=duration:stream=codec_type', '-of', 'json', str(dst)],
                           capture_output=True, text=True)
    try:
        info = json.loads(probe.stdout)
        duration = float(info['format']['duration'])
        assert info['streams']
    except (ValueError, KeyError, AssertionError, TypeError):
        raise BadRequest('영상 파일을 읽을 수 없어요. 다른 영상으로 다시 시도해 주세요.')
    if info_out is not None:
        info_out.update(bytes=size, durationSec=round(duration, 2))
    if duration > config.MAX_DURATION_SEC:
        raise BadRequest(f'영상은 {config.MAX_DURATION_SEC:.0f}초 이하만 올릴 수 있어요.')
    return dst


@app.get('/health')
def health():
    return {'ok': True, 'runner': RUNNER, 'queued': store.queue.qsize()}


@app.post('/analysis')
async def create_analysis(request: Request, video: UploadFile = File(...), exercise: str = Form(...),
                          coachId: str | None = Form(None), coachVideo: UploadFile | None = File(None)):
    client = request.client.host if request.client else None
    sent = dict(exercise=exercise, coachId=coachId,                 # logged: what the frontend sent
                video=api_log.file_info(video), coachVideo=api_log.file_info(coachVideo))
    try:
        if exercise not in config.EXERCISES:
            raise BadRequest('알 수 없는 종목이에요.')
        if exercise not in config.SUPPORTED_EXERCISES:
            raise BadRequest('아직 스쿼트만 분석할 수 있어요.')
        if bool(coachId) == bool(coachVideo):
            raise BadRequest('코치 영상을 하나 골라 주세요.')
        if coachId and (coachId not in config.COACHES or config.COACHES[coachId]['exercise'] != exercise):
            raise BadRequest('선택한 코치 영상을 찾을 수 없어요.')
        job_id, d = store.new_dir()
        try:
            user_path = await save_video(video, d / 'upload_user', sent['video'])
            coach_path = await save_video(coachVideo, d / 'upload_coach', sent['coachVideo']) if coachVideo else None
        except BadRequest:
            shutil.rmtree(d, ignore_errors=True)
            raise
    except BadRequest as e:
        api_log.post(sent, {'message': e.message}, e.status, client=client)
        raise
    store.submit(Job(jobId=job_id, exercise=exercise, dir=str(d), coachId=coachId,
                     coachVideo=str(coach_path) if coach_path else None, video=str(user_path)))
    response = {'jobId': job_id}
    api_log.post(sent, response, 202, job_dir=d, client=client)
    return JSONResponse(response, status_code=202)


@app.get('/analysis/{job_id}')
def get_analysis(job_id: str, request: Request):
    client = request.client.host if request.client else None
    job = store.get(job_id)
    if job is None:
        api_log.rejected(f'GET /analysis/{job_id}', '분석 작업을 찾을 수 없어요.', 404, client)
        raise BadRequest('분석 작업을 찾을 수 없어요.', status=404)
    response = job.public()
    api_log.get(job, response, client)
    return response
