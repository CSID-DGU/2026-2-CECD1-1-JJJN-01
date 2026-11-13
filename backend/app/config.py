"""Paths, limits and the coach library (single source on the backend side)."""
import os as _os
from pathlib import Path

# CPU thread cap (same as engine/pipeline/__init__.py): the container sees 256 CPUs but
# its quota is ~6.8, and the 128-thread default makes the kernel freeze the process (WHAM 4x slower).
# Set before numpy/torch load; pipeline subprocesses inherit it. Values already in the env win.
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    _os.environ.setdefault(_k, '6')

BACKEND = Path(__file__).resolve().parents[1]
PROJECT = BACKEND.parent                                            # monorepo root (backend/, engine/, .venv*)
# Engine location and its interpreter. Defaults are relative to the repo; override with env vars
# when the engine or the venv lives elsewhere.
PIPELINE_REPO = Path(_os.environ.get('PIPELINE_REPO', PROJECT / 'engine'))          # pipeline/ lives here
PIPELINE_PYTHON = Path(_os.environ.get('PIPELINE_PYTHON', PROJECT / '.venv/bin/python'))
DATA_DIR = BACKEND / 'data'
JOBS_DIR = DATA_DIR / 'jobs'

# Upload limits: same values as frontend/src/constants/upload.ts
MAX_BYTES = 100 * 1024 * 1024
MAX_DURATION_SEC = 30.
VIDEO_EXT = {'.mp4', '.mov', '.webm'}
VIDEO_MIME = {'video/mp4', 'video/quicktime', 'video/webm'}

EXERCISES = ('squat', 'deadlift', 'bench_press')
SUPPORTED_EXERCISES = ('squat',)          # segment labelling is squat-only for now

# Placeholder until the total-score formula is decided (frontend requires score.total).
DEFAULT_SCORE_TOTAL = 0

# coachId -> pre-computed run_video output of the coach (same id in frontend/src/constants/coachVideos.ts)
COACHES = {
    'squat-diag': dict(exercise='squat', run=PIPELINE_REPO / 'runs/video_wham/squat_diag_gframe'),
}

# Stage 1 (demo runner): every job returns this existing pipeline result instead of running.
DEMO_RETARGET_DIR = PIPELINE_REPO / 'runs/coach_user/coach_to_student'
DEMO_STAGE_SEC = 1.5


def _load_dotenv(path=BACKEND / '.env'):
    """KEY=value lines of backend/.env into os.environ (existing variables win). Values are never logged."""
    import os
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

# LLM feedback (llm stage). LLM_PROVIDER=none disables it (result without `feedback`).
import os as _os  # noqa: E402
LLM_PROVIDER = _os.environ.get('LLM_PROVIDER', 'gemini')
# Tried in order: the next one on overload (503), rate limit (429), timeout or a removed model (404).
GEMINI_MODELS = [m.strip() for m in _os.environ.get(
    'GEMINI_MODELS', 'gemini-3.5-flash,gemini-3.8-flash,gemini-3.1-flash-lite').split(',') if m.strip()]
LLM_TIMEOUT_SEC = float(_os.environ.get('LLM_TIMEOUT_SEC', '40'))       # per model attempt
LLM_DEADLINE_SEC = float(_os.environ.get('LLM_DEADLINE_SEC', '90'))     # whole llm stage
# feedback: pose differences below these count as "similar" and are not mentioned (draft values; tune on real pairs)
POSE_SAME_DEG = float(_os.environ.get('POSE_SAME_DEG', '8'))
POSE_SAME_LEG = float(_os.environ.get('POSE_SAME_LEG', '0.035'))   # height difference, share of the leg length (3 cm for 85 cm)
POSE_SAME_RATIO = float(_os.environ.get('POSE_SAME_RATIO', '0.15'))   # kneeWidthRatio
