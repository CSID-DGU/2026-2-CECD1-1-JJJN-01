#!/usr/bin/env bash
# Recreate the three Python environments at the repo root. Idempotent: an existing venv is reused.
#
#   bash scripts/setup_venvs.sh            # all three
#   bash scripts/setup_venvs.sh wham       # one of: backend | pose | wham
#
# Layout (defaults in backend/app/config.py and engine/pipeline/core/paths.py; env vars override):
#   .venv        backend API + engine (PAN, IK, scoring). --system-site-packages: torch 2.8.0+cu128 comes
#                from the RunPod image, so this venv stays small. Requires a host with that torch installed.
#   .venv-pose   MediaPipe extractor (CPU).                engine/requirements-pose.txt
#   .venv-wham   WHAM extractor, own torch cu128 (~11 GB). engine/requirements-wham.txt
#                + mmpose installed editable from third_party/WHAM/third-party/ViTPose (patched mmcv bound),
#                so third_party/WHAM must be in place first.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SYS_PY="${SYS_PY:-/usr/bin/python3}"          # 3.12 on the RunPod image

setup_backend() {
  [ -x .venv/bin/python ] || "$SYS_PY" -m venv --system-site-packages .venv
  .venv/bin/python -c 'import torch; assert torch.__version__.startswith("2.8."), torch.__version__' \
    || { echo "system torch 2.8 not visible from .venv (image changed?)"; exit 1; }
  .venv/bin/python -m pip install -q --upgrade pip
  .venv/bin/python -m pip install -q -r backend/requirements.txt
  echo ".venv ok: $(.venv/bin/python -c 'import fastapi, torch; print("fastapi", fastapi.__version__, "torch", torch.__version__)')"
}

setup_pose() {
  [ -x .venv-pose/bin/python ] || "$SYS_PY" -m venv .venv-pose
  .venv-pose/bin/python -m pip install -q --upgrade pip
  .venv-pose/bin/python -m pip install -q -r engine/requirements-pose.txt
  echo ".venv-pose ok: $(.venv-pose/bin/python -c 'import mediapipe; print("mediapipe", mediapipe.__version__)')"
}

setup_wham() {
  WHAM="${WHAM_DIR:-$ROOT/third_party/WHAM}"
  [ -d "$WHAM/third-party/ViTPose/mmpose" ] || { echo "WHAM checkout missing at $WHAM (copy it first)"; exit 1; }
  [ -x .venv-wham/bin/python ] || "$SYS_PY" -m venv .venv-wham
  .venv-wham/bin/python -m pip install -q --upgrade pip setuptools wheel
  # chumpy's setup.py imports pip at build time, which the isolated build env lacks: install it last, without
  # isolation, once numpy/scipy are present.
  grep -v '^chumpy' engine/requirements-wham.txt > "$ROOT/.venv-wham/requirements-no-chumpy.txt"
  .venv-wham/bin/python -m pip install -q -r "$ROOT/.venv-wham/requirements-no-chumpy.txt"
  .venv-wham/bin/python -m pip install -q --no-build-isolation --no-deps "$(grep '^chumpy' engine/requirements-wham.txt)"
  # xtcocotools ships no wheel; built in pip's isolated env it links against a different numpy ABI
  # ("numpy.dtype size changed" at import). Rebuild it from source against the numpy installed above.
  .venv-wham/bin/python -m pip install -q --no-cache-dir --no-build-isolation --no-deps --force-reinstall \
    --no-binary :all: "$(grep '^xtcocotools' engine/requirements-wham.txt)"
  .venv-wham/bin/python -m pip install -q --no-deps -e "$WHAM/third-party/ViTPose"
  .venv-wham/bin/python -c 'import xtcocotools._mask, chumpy, mmpose.apis, smplx' || { echo ".venv-wham import check failed"; exit 1; }
  echo ".venv-wham ok: $(.venv-wham/bin/python -c 'import torch, mmcv, mmpose, smplx; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "mmcv", mmcv.__version__, "mmpose", mmpose.__file__)')"
}

case "${1:-all}" in
  backend) setup_backend ;;
  pose)    setup_pose ;;
  wham)    setup_wham ;;
  all)     setup_backend; setup_pose; setup_wham ;;
  *) echo "usage: $0 [backend|pose|wham|all]"; exit 2 ;;
esac
