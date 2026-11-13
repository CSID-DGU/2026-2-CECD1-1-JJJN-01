"""S0 (WHAM): video -> SMPL joints in world coordinates (.npz). Run with the WHAM venv:

  ../.venv-wham/bin/python tools/pose_estimation/wham_extract.py VIDEO OUT.npz     # from engine/

WHAM checkout: $WHAM_DIR, default <repo>/third_party/WHAM (same rule as pipeline/core/paths.py;
this script runs in the WHAM venv and does not import the pipeline package).

Static camera assumed (DPVO not installed): WHAM then predicts the world trajectory from
the image evidence with an identity camera motion.

Output contract (consumed by pipeline/core/wham_format.py):
  joints      (T, 24, 3) float32  SMPL joints, world frame, metres, Y up
  rest_joints (24, 3)    float32  SMPL joints of the subject's body (betas) in the zero pose
  pose_world  (T, 72), trans_world (T, 3), betas (10,)
  detected    (T,) bool           frames where WHAM produced the tracked subject
  fps, source
"""
import argparse
import contextlib
import os
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]        # engine/tools/pose_estimation/ -> repo root
WHAM = Path(os.environ.get('WHAM_DIR', PROJECT / 'third_party/WHAM'))


@contextlib.contextmanager
def skip_random_init():
    """Skip the random weight init of the ViT-H models (ViTPose, HMR2) while they are built.

    Every weight is overwritten by the checkpoint right after construction, but the init runs
    on the CPU over ~1.3 B parameters (~50 s of the ~80 s model loading). Checked: tracking
    keypoints and WHAM output identical with and without (child_squat, 2026-11-07).
    """
    import torch.nn as nn
    import timm.layers.weight_init as wi
    saved = [(nn.Linear, 'reset_parameters', nn.Linear.reset_parameters),
             (nn.modules.conv._ConvNd, 'reset_parameters', nn.modules.conv._ConvNd.reset_parameters),
             (wi, '_trunc_normal_', wi._trunc_normal_)]
    nn.Linear.reset_parameters = lambda self: None
    nn.modules.conv._ConvNd.reset_parameters = lambda self: None
    wi._trunc_normal_ = lambda tensor, *a, **k: tensor
    try:
        yield
    finally:
        for obj, name, fn in saved:
            setattr(obj, name, fn)


def extract(video, work_dir, flip_eval=True, fast_init=True):
    os.chdir(WHAM)
    sys.path.insert(0, str(WHAM))
    import cv2
    import joblib
    import smplx
    import torch
    from configs.config import get_cfg_defaults
    from lib.models import build_body_model, build_network
    import demo

    # PyTorch >= 2.6 loads checkpoints with weights_only=True; WHAM's official checkpoints
    # (fetched from the authors' links in fetch_demo_data.sh) contain numpy scalars/dtypes.
    # Allow-list only those types instead of disabling the safe loader.
    torch.serialization.add_safe_globals([
        (np.core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
        (np.core.multiarray._reconstruct, 'numpy.core.multiarray._reconstruct'),
        np.dtype, np.ndarray, np.dtypes.Float64DType, np.dtypes.Float32DType, np.dtypes.Int64DType])

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')
    cfg.FLIP_EVAL = flip_eval
    smpl = build_body_model(cfg.DEVICE, cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN)
    network = build_network(cfg, smpl)
    network.eval()
    work_dir.mkdir(parents=True, exist_ok=True)
    demo.args = argparse.Namespace(run_smplify=False)   # demo.run reads the module-level CLI args
    # WHAM crashes with an IndexError when the detector found nobody: fail with a clear message
    from lib.models.preproc.detector import DetectionModel
    process = DetectionModel.process

    def process_or_fail(self, fps):
        if len(self.tracking_results['keypoints']) == 0:
            raise RuntimeError('WHAM found no person')
        return process(self, fps)
    DetectionModel.process = process_or_fail
    with (skip_random_init() if fast_init else contextlib.nullcontext()):   # ViTPose/HMR2 are built in run
        demo.run(cfg, str(video), str(work_dir), network, None, run_global=False, save_pkl=True, visualize=False)

    results = joblib.load(work_dir / 'wham_output.pkl')
    if not results:
        raise RuntimeError('WHAM found no person')
    sid = max(results, key=lambda k: len(results[k]['frame_ids']))       # longest track
    r = results[sid]
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    body = smplx.SMPL(str(WHAM / 'dataset/body_models/smpl'), gender='neutral')
    pose = torch.tensor(r['pose_world'], dtype=torch.float32)
    trans = torch.tensor(r['trans_world'], dtype=torch.float32)
    betas = torch.tensor(r['betas'], dtype=torch.float32)
    betas = betas.mean(0, keepdim=True) if betas.ndim == 2 else betas[None]
    with torch.no_grad():
        out = body(betas=betas.expand(len(pose), -1), global_orient=pose[:, :3],
                   body_pose=pose[:, 3:], transl=trans)
        rest = body(betas=betas).joints[0, :24]
    ids = np.asarray(r['frame_ids'])
    T = max(n, int(ids.max()) + 1)
    joints = np.full((T, 24, 3), np.nan, np.float32)
    joints[ids] = out.joints[:, :24].numpy()
    detected = np.zeros(T, bool)
    detected[ids] = True
    full = lambda a: (lambda z: (z.__setitem__(ids, a), z)[1])(np.full((T,) + a.shape[1:], np.nan, np.float32))
    return dict(joints=joints, rest_joints=rest.numpy().astype(np.float32),
                pose_world=full(np.asarray(r['pose_world'], np.float32)),
                trans_world=full(np.asarray(r['trans_world'], np.float32)),
                betas=betas[0].numpy(), detected=detected, fps=float(fps), source=str(video))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('video')
    ap.add_argument('out')
    ap.add_argument('--no-flip', action='store_true', help='disable flip test-time augmentation')
    ap.add_argument('--slow-init', action='store_true', help='keep the random weight init (for checking)')
    opt = ap.parse_args()
    video, out = Path(opt.video).resolve(), Path(opt.out).resolve()
    data = extract(video, out.parent / 'wham_work', flip_eval=not opt.no_flip, fast_init=not opt.slow_init)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    print(f'{len(data["detected"])} frames @ {data["fps"]:.2f} fps, subject in '
          f'{int(data["detected"].sum())}; wrote {out}')


if __name__ == '__main__':
    main()
