"""Filesystem locations. The official PAN code uses cwd-relative paths, so the pipeline
always runs with the repository root as the working directory (see `enter_repo`)."""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]          # engine/
PROJECT = REPO.parent                                # monorepo root (backend/, engine/, .venv*, third_party/)
MIXAMO_DIR = REPO / 'data_preprocess/Mixamo/Mixamo'
STD_BVH_DIR = MIXAMO_DIR / 'std_bvhs'
MEAN_VAR_DIR = MIXAMO_DIR / 'mean_var'
ASSET_DIR = REPO / 'pipeline/assets'

# External pieces. Defaults are relative to the repo root; env vars override them.
# Virtual environments of the pose estimators (S0 runs as a subprocess in its own venv).
POSE_PYTHON = Path(os.environ.get('PIPELINE_POSE_PYTHON', PROJECT / '.venv-pose/bin/python'))
WHAM_PYTHON = Path(os.environ.get('PIPELINE_WHAM_PYTHON', PROJECT / '.venv-wham/bin/python'))
# WHAM checkout (code + checkpoints + SMPL body models); same default as tools/pose_estimation/wham_extract.py.
WHAM_DIR = Path(os.environ.get('WHAM_DIR', PROJECT / 'third_party/WHAM'))
# PAN training npy files, only needed to rebuild pipeline/assets/priors_*.json (not shipped with the test archive).
TRAIN_NPY_DIR = Path(os.environ.get(
    'PAN_TRAIN_NPY_DIR', PROJECT / 'third_party/pan-motion-retargeting/data_preprocess/Mixamo/Mixamo'))


def enter_repo():
    os.chdir(REPO)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
