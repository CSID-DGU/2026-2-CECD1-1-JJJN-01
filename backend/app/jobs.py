"""Job records and the single GPU worker.

Jobs live in memory (and job.json in the job directory, for inspection). One worker thread runs
jobs one at a time in arrival order: the GPU (MIG slice) cannot run two pipelines at once.
The runner is pluggable: DemoRunner (stage 1) walks through the stages and returns an existing
pipeline result; the real pipeline runner replaces it in stage 2.
"""
import json
import queue
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import config
from .result_builder import build_result


class UserFacingError(Exception):
    """Failure whose message is shown to the user as is (Korean)."""


@dataclass
class Job:
    jobId: str
    exercise: str
    dir: str
    coachId: str | None = None
    coachVideo: str | None = None
    video: str = ''
    status: str = 'queued'
    progress: float = 0.
    error: str | None = None
    created: float = field(default_factory=time.time)
    result: dict | None = None

    def public(self):
        out = dict(jobId=self.jobId, status=self.status, progress=round(self.progress, 3))
        if self.status == 'done':
            out['result'] = self.result
        if self.status == 'failed':
            out['error'] = self.error
        return out


class JobStore:
    def __init__(self, runner):
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.queue: queue.Queue[str] = queue.Queue()
        self.runner = runner
        threading.Thread(target=self._worker, daemon=True).start()

    def new_dir(self):
        job_id = uuid.uuid4().hex[:12]
        d = config.JOBS_DIR / job_id
        d.mkdir(parents=True, exist_ok=True)
        return job_id, d

    def submit(self, job: Job):
        with self.lock:
            self.jobs[job.jobId] = job
        self._save(job)
        self.queue.put(job.jobId)

    def get(self, job_id):
        with self.lock:
            return self.jobs.get(job_id)

    def update(self, job, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(job, k, v)
        self._save(job)

    def _save(self, job):
        meta = {k: v for k, v in asdict(job).items() if k != 'result'}
        (Path(job.dir) / 'job.json').write_text(json.dumps(meta, indent=1, ensure_ascii=False))
        if job.result is not None:
            (Path(job.dir) / 'result.json').write_text(json.dumps(job.result, ensure_ascii=False))

    def _worker(self):
        while True:
            job = self.get(self.queue.get())
            try:
                result = self.runner(job, lambda status, progress: self.update(job, status=status, progress=progress))
                self.update(job, status='done', progress=1., result=result)
            except UserFacingError as e:
                self.update(job, status='failed', error=str(e))
            except Exception:                                   # noqa: BLE001 -- never kill the worker
                (Path(job.dir) / 'error.log').write_text(traceback.format_exc())
                self.update(job, status='failed', error='분석 중 오류가 발생했어요. 잠시 후 다시 시도해 주세요.')


def demo_runner(job: Job, set_status):
    """Stage 1: no analysis. Walks through the statuses and returns an existing pipeline result
    (config.DEMO_RETARGET_DIR) so the frontend can be wired to the real result format."""
    stages = ['extracting', 'retargeting', 'scoring']
    for k, s in enumerate(stages):
        set_status(s, k / len(stages))
        time.sleep(config.DEMO_STAGE_SEC)
    return build_result(job.jobId, job.exercise, config.DEMO_RETARGET_DIR)
