"""[1] Preprocess: joints -> body-frame bone directions -> SG smoothing, speed, acceleration, arc length."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config

# Bones as (name, parent joint, child joint) in SMPL indices.
SMPL_BONES = (
    ("l_thigh", 1, 4), ("r_thigh", 2, 5), ("l_shank", 4, 7), ("r_shank", 5, 8),
    ("trunk", 0, 12), ("head", 12, 15),
    ("l_upper_arm", 16, 18), ("r_upper_arm", 17, 19), ("l_forearm", 18, 20), ("r_forearm", 19, 21),
)
SMPL_NAMES = ("pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
              "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar",
              "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
              "left_wrist", "right_wrist", "left_hand", "right_hand")
UP_AXES = {"x": 0, "y": 1, "z": 2}


@dataclass
class Track:
    """Raw input: timestamps [N] (s), bone unit vectors [N, K, 3] in a body-centred frame
    (x = lateral, y = up, z = forward), optional confidence [N, K] in 0..1."""
    t: np.ndarray
    bones: np.ndarray
    bone_names: tuple
    conf: np.ndarray | None = None

    def mirrored(self) -> "Track":
        """Left/right swap (student imitating the coach like a mirror)."""
        idx = list(range(len(self.bone_names)))
        for k, n in enumerate(self.bone_names):
            for a, b in (("l_", "r_"), ("r_", "l_")):
                if n.startswith(a) and b + n[2:] in self.bone_names:
                    idx[k] = self.bone_names.index(b + n[2:])
        b = self.bones[:, idx].copy()
        b[..., 0] *= -1
        conf = None if self.conf is None else self.conf[:, idx]
        return Track(self.t, b, self.bone_names, conf)


def _unit(v, axis=-1):
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), 1e-9)


def track_from_joints(joints, t, up="y", conf=None, joint_names=None, bones=SMPL_BONES) -> Track:
    """joints [N, J, 3] (SMPL order, or any order with joint_names) -> heading-normalised bone track."""
    joints = np.asarray(joints, float)
    if joint_names is not None:
        order = [list(joint_names).index(n) for n in SMPL_NAMES[:22]]
        joints = joints[:, order]
        if conf is not None:
            conf = np.asarray(conf)[:, order]
    sign = -1.0 if up.startswith("-") else 1.0
    u = np.zeros(3); u[UP_AXES[up[-1]]] = sign
    # per-frame body frame: x = hip axis projected on the ground plane, y = up, z = forward
    hip = joints[:, 2] - joints[:, 1]
    hip = _unit(hip - (hip @ u)[:, None] * u)
    # up x hip (not hip x up): hip points to the person's right, so this is the way the toes point.
    # Angles do not depend on the sign; only the forward / backward labels of motion_summary do.
    fwd = _unit(np.cross(u, hip))
    R = np.stack([hip, np.broadcast_to(u, hip.shape), fwd], axis=1)       # [N, 3, 3] rows = new axes
    vec = np.stack([joints[:, c] - joints[:, p] for _, p, c in bones], axis=1)
    local = np.einsum("nij,nkj->nki", R, vec)
    bconf = None
    if conf is not None:
        conf = np.asarray(conf, float)
        bconf = np.stack([np.minimum(conf[:, p], conf[:, c]) for _, p, c in bones], axis=1)
    return Track(np.asarray(t, float), _unit(local), tuple(n for n, _, _ in bones), bconf)


def load_npz(path: str | Path, up="y") -> Track:
    """NPZ with joints [T, J, 3] and times [T] (or fps); optional joint_names, confidence [T, J]."""
    z = np.load(path, allow_pickle=False)
    joints = z["joints"]
    if "times" in z:
        t = z["times"]
    elif "timestamps" in z:
        t = z["timestamps"]
    else:
        t = np.arange(len(joints)) / float(z["fps"])
    names = tuple(str(n) for n in z["joint_names"]) if "joint_names" in z else None
    conf = z["confidence"] if "confidence" in z else None
    return track_from_joints(joints, t, up, conf, names)


# ---------------------------------------------------------------- Savitzky-Golay with derivatives
def savgol(t, x, window_s, order):
    """Smoothing + 1st + 2nd derivative along axis 0 (window in seconds; uses the median dt)."""
    dt = float(np.median(np.diff(t)))
    half = max(order, int(round(window_s / dt / 2)))
    xs = np.arange(-half, half + 1) * dt
    P = np.linalg.pinv(np.vander(xs, order + 1, increasing=True))
    flat = x.reshape(len(x), -1)
    pad = np.pad(flat, ((half, half), (0, 0)), mode="edge")
    out = []
    for deriv, fact in ((0, 1.0), (1, 1.0), (2, 2.0)):
        k = P[deriv] * fact
        res = np.stack([np.convolve(pad[:, j], k[::-1], mode="valid") for j in range(flat.shape[1])], 1)
        out.append(res.reshape(x.shape))
    return out


class Clip:
    """Preprocessed motion. Attributes are per frame unless noted."""

    def __init__(self, track: Track, cfg: Config, bone_weights=None):
        self.track = track
        self.t = track.t
        self.names = track.bone_names
        K = len(self.names)
        self.w = np.ones(K) if bone_weights is None else np.asarray(bone_weights, float)
        conf = np.ones((len(self.t), K)) if track.conf is None else np.asarray(track.conf, float)
        self.conf = conf
        self.conf_ok = conf.mean(1) >= cfg.low_conf
        raw = track.bones.copy()
        if not self.conf_ok.all() and self.conf_ok.sum() >= 2:
            # bridge low-confidence frames by interpolation so they do not create fake events / drops;
            # the pipeline withholds judgement on them
            good = np.where(self.conf_ok)[0]
            for k in range(K):
                for c in range(3):
                    raw[:, k, c] = np.interp(self.t, self.t[good], raw[good, k, c])
            raw = _unit(raw)
        b, db, ddb = savgol(self.t, raw, cfg.sg_window_s, cfg.sg_order)
        self.b = _unit(b)                                              # [N, K, 3]
        self.db, self.ddb = db, ddb
        wsum = self.w.sum()
        sq = (np.linalg.norm(db, axis=2) ** 2 * self.w).sum(1) / wsum
        self.speed = np.degrees(np.sqrt(sq))                           # deg/s (RMS over bones)
        # tempo-invariant normalisation by a near-peak speed (p97 is insensitive to how long the clip stands still)
        self.vref = float(np.percentile(self.speed, cfg.v_ref_pct))
        self.v = self.speed / max(self.vref, 1e-9)
        # tangential acceleration d|v|/dt = (v . a) / |v|
        acc =np.degrees(((db * ddb).sum(2) * self.w).sum(1) / wsum / np.maximum(np.radians(self.speed), 1e-9))
        self.acc = acc                                                 # deg/s^2
        self.a = acc / max(float(np.percentile(np.abs(acc), 90)), 1e-9)
        flat = (db * np.sqrt(self.w)[None, :, None]).reshape(len(self.t), -1)
        self.dir = _unit(flat)
        self.dt = np.r_[np.diff(self.t)[0], np.diff(self.t)]
        step = np.degrees(np.linalg.norm(np.diff(self.b, axis=0).reshape(len(self.t) - 1, -1), axis=1))
        # remove the jitter floor (high percentile of the per-frame step while nearly still), so that still
        # periods and small / slow movements are not stretched by accumulated noise
        still = (self.v[1:] < cfg.v_low) & (self.v[:-1] < cfg.v_low)
        floor = float(np.percentile(step[still], cfg.arc_floor_pct)) if still.any() else 0.0
        step = np.clip(step - floor, 0, None)
        self._step = step
        self.arc = np.r_[0.0, np.cumsum(step)]                         # cumulative pose-path length (deg)
        self.s = self.arc / max(self.arc[-1], 1e-9)                    # progress 0..1

    def arc_excluding(self, still):
        """pose-path length that does not grow inside the given still frames (pause / rest runs)."""
        m = ~(still[1:] & still[:-1])
        return np.r_[0.0, np.cumsum(self._step * m)]

    def __len__(self):
        return len(self.t)


def pose_dist(ba, bb, w, ca=None, cb=None, trim=0.0):
    """Weighted mean angle (deg) between corresponding bones. ba [..., K, 3], bb [..., K, 3].
    trim: fraction of the most different bones left out (robust "does most of the body do the same?")."""
    cos = np.clip((ba * bb).sum(-1), -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    ww = np.broadcast_to(w if ca is None else w * ca * cb, ang.shape).copy()
    k = int(round(trim * ang.shape[-1]))
    if k > 0:
        idx = np.argsort(ang, axis=-1)[..., ang.shape[-1] - k:]
        np.put_along_axis(ww, idx, 0.0, axis=-1)
    return (ang * ww).sum(-1) / np.maximum(ww.sum(-1), 1e-9)
