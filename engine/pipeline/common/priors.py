"""Data-driven joint priors measured on PAN's own training motions.

Everything here is computed from `<char>.npy` (the training clips PAN was fit on), so the
IK constraints describe the distribution PAN expects instead of hand-picked anatomy:
  * fixed joints  - joints whose rotation never changes in the data (e.g. Mixamo `Hips`)
  * mean pose     - per-joint mean rotation (IK initialisation)
  * limits        - per-joint percentile box on the local rotation vector (soft limits)
  * hinge report  - knee/elbow principal axis and off-axis spread (informational)
  * ground        - median height of the lowest foot joint (root re-anchoring of video input)
The PAN normalisation statistics (mean/var npy) are loaded separately in `load_pan_stats`.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import quat as Q
from .paths import ASSET_DIR, MEAN_VAR_DIR, TRAIN_NPY_DIR
from .skeleton import Skeleton

FORMAT = 2
FEET = ('LeftFoot', 'RightFoot', 'LeftToeBase', 'RightToeBase')
HINGES = ('LeftLeg', 'RightLeg', 'LeftForeArm', 'RightForeArm')


def training_joint_quats(skel, npy_path):
    """Training clips -> list of (T, J, 4) local quats (30 fps, like PAN training)."""
    from outer_utils.Quaternions import Quaternions
    clips = np.load(npy_path, allow_pickle=True)
    out = []
    for clip in clips:
        m = np.asarray(clip, dtype=np.float64)[::2]
        if m.shape[1] != len(skel.edges) * 3 + 3:
            raise ValueError(f'{npy_path}: channel mismatch with skeleton')
        euler = m[:, :-3].reshape(len(m), -1, 3)
        qe = Quaternions.from_euler(np.radians(euler)).qs          # same call as PAN training
        q = np.zeros((len(m), skel.n_joints, 4))
        q[..., 0] = 1
        for e, (a, _) in enumerate(skel.edges):
            q[:, a] = qe[:, e]
        out.append(q)
    return out


def build_priors(character='BigVegas', npy_path=None, lo=0.5, hi=99.5):
    skel = Skeleton.from_character(character)
    npy_path = Path(npy_path or TRAIN_NPY_DIR / f'{character}.npy')
    clips = training_joint_quats(skel, npy_path)
    q = torch.tensor(np.concatenate(clips), dtype=torch.float64)
    q = Q.positive_w(Q.normalize(q))
    rotvec = Q.log(q)                                              # (N, J, 3)
    joints = {}
    for j in skel.rotating_joints:
        name = skel.names[j]
        std = float(q[:, j].std(0).max())
        mean = Q.positive_w(Q.normalize(q[:, j].mean(0)))
        r = rotvec[:, j].numpy()
        info = dict(index=j, fixed=std < 1e-4, quat_std=std, mean=mean.tolist(),
                    limit_lo_deg=np.degrees(np.percentile(r, lo, 0)).tolist(),
                    limit_hi_deg=np.degrees(np.percentile(r, hi, 0)).tolist())
        if name in HINGES:
            big = np.linalg.norm(r, axis=1) > np.radians(15)
            ax = r[big] / np.linalg.norm(r[big], axis=1, keepdims=True)
            axis = np.linalg.eigh(ax.T @ ax)[1][:, -1]
            flex = r @ axis
            off = np.linalg.norm(r - flex[:, None] * axis, axis=1)
            info['hinge'] = dict(axis=axis.tolist(),
                                 flex_range_deg=np.degrees(np.percentile(flex, [lo, hi])).tolist(),
                                 off_axis_median_deg=float(np.degrees(np.median(off))),
                                 off_axis_p95_deg=float(np.degrees(np.percentile(off, 95))))
        joints[name] = info
    # Ground: lowest foot joint per frame, from FK of the training motions.
    root = torch.tensor(np.concatenate([np.asarray(c, dtype=np.float64)[::2, -3:]
                                        for c in np.load(npy_path, allow_pickle=True)]))
    pos = skel.fk(q, root)
    feet = [skel.index(n) for n in FEET]
    lowest = pos[:, feet, 1].min(1).values.numpy()
    ground = dict(lowest_foot_y_median=float(np.median(lowest)),
                  lowest_foot_y_p5=float(np.percentile(lowest, 5)),
                  lowest_foot_y_p95=float(np.percentile(lowest, 95)),
                  height=skel.height(), joints=list(FEET))
    return dict(format=FORMAT, character=character, source=str(npy_path), ground=ground,
                created=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                frames=int(q.shape[0]), clips=len(clips), percentiles=[lo, hi], joints=joints)


def priors_path(character):
    return ASSET_DIR / f'priors_{character}.json'


def load_priors(character='BigVegas', rebuild=False):
    path = priors_path(character)
    if rebuild or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(build_priors(character), indent=1))
    data = json.loads(path.read_text())
    if data.get('format') != FORMAT or data.get('character') != character:
        raise ValueError(f'Stale priors file {path}; rerun with rebuild')
    return data


def load_pan_stats(character):
    """PAN normalisation (mean, std) of raw channels, shape (C,)."""
    mean = np.load(MEAN_VAR_DIR / f'{character}_mean.npy').reshape(-1)
    var = np.load(MEAN_VAR_DIR / f'{character}_var.npy').reshape(-1)
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(var, dtype=torch.float32)
