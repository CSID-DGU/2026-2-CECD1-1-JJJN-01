"""Demo runner (ANALYSIS_RUNNER=demo): no analysis, for frontend work without a GPU."""
import time

from .. import constants
from ..jobs.job_queue import Job
from ..result.result_builder import build_result


def demo_runner(job: Job, set_status):
    """Stage 1: no analysis. Walks through the statuses and returns an existing pipeline result
    (constants.DEMO_RETARGET_DIR) so the frontend can be wired to the real result format."""
    stages = ['extracting', 'retargeting', 'scoring']
    for k, s in enumerate(stages):
        set_status(s, k / len(stages))
        time.sleep(constants.DEMO_STAGE_SEC)
    return build_result(job.jobId, job.exercise, constants.DEMO_RETARGET_DIR)
