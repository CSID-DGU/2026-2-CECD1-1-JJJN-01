"""[4] Frame alignment: frame cost and two-sided Drop-DTW inside each anchor interval."""
from __future__ import annotations

import numpy as np

from .config import Config
from .step1_preprocess import Clip, pose_dist


def frame_cost(C: Clip, S: Clip, cfg: Config, tau: float):
    """d [nC, nS] (pose only, deg) and cost = d + direction term."""
    d = np.empty((len(C), len(S)))
    for i in range(len(C)):                                            # row-wise keeps memory O(nS*K)
        d[i] = pose_dist(C.b[i][None], S.b, C.w, C.conf[i][None], S.conf, cfg.pose_trim)
    cosd = C.dir @ S.dir.T
    gate = (C.v[:, None] > cfg.dir_gate) & (S.v[None, :] > cfg.dir_gate)
    return d + cfg.beta * tau * (1 - cosd) * gate, d


# ---------------------------------------------------------------- frames
STEP_NAMES = ("match", "warp_c", "warp_s", "drop_c", "drop_s")


def drop_dtw(cost, dc, ds, lam, hold_c=None, hold_s=None):
    """Two-sided Drop-DTW with fixed endpoints and time-weighted steps.

    match   (i-1,j-1)->(i,j): cost * (dc+ds)/2
    warp_c  (i-1,j)  ->(i,j): cost * dc/2        coach advances, student waits
    warp_s  (i,j-1)  ->(i,j): cost * ds/2
    drop_c  (i-1,j)  ->(i,j): lam  * dc/2        coach frame has no counterpart (0 if held)
    drop_s  (i,j-1)  ->(i,j): lam  * ds/2
    Returns [(step, i, j)] from (0,0) to (n-1,m-1).
    """
    n, m = cost.shape
    lam_c = np.where(hold_c, 0.0, lam) if hold_c is not None else np.full(n, lam)
    lam_s = np.where(hold_s, 0.0, lam) if hold_s is not None else np.full(m, lam)
    D = np.full((n, m), np.inf)
    D[0, 0] = 0.0
    bp = np.full((n, m), -1, dtype=np.int8)
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            c = cost[i, j]
            best, b = np.inf, -1
            if i and j:
                v = D[i - 1, j - 1] + c * (dc[i] + ds[j]) / 2
                if v < best: best, b = v, 0
            if i:
                v = D[i - 1, j] + c * dc[i] / 2
                if v < best: best, b = v, 1
                v = D[i - 1, j] + lam_c[i] * dc[i] / 2
                if v < best: best, b = v, 3
            if j:
                v = D[i, j - 1] + c * ds[j] / 2
                if v < best: best, b = v, 2
                v = D[i, j - 1] + lam_s[j] * ds[j] / 2
                if v < best: best, b = v, 4
            D[i, j], bp[i, j] = best, b
    steps, i, j = [], n - 1, m - 1
    while (i, j) != (0, 0):
        b = int(bp[i, j])
        steps.append((STEP_NAMES[b], i, j))
        if b == 0:
            i, j = i - 1, j - 1
        elif b in (1, 3):
            i -= 1
        else:
            j -= 1
    steps.append(("match", 0, 0))
    return steps[::-1]


def plain_dtw(d, dc, ds):
    """Time-weighted DTW without drops (used for calibration) -> [(i, j)]."""
    steps = drop_dtw(d, dc, ds, lam=np.inf)
    return [(i, j) for _, i, j in steps]
