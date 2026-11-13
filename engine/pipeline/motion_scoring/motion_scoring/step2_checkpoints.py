"""[2] Checkpoints: Checkpoint events from normalised speed (what happened) and acceleration (when it started / ended)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .step1_preprocess import Clip


@dataclass
class Event:
    type: str          # rest | pause | reversal
    i: int             # anchor frame
    t: float           # anchor time (s)
    i0: int            # low-speed run start frame
    i1: int            # low-speed run end frame
    t_stop: float      # deceleration peak before the run (acceleration minimum)
    t_go: float        # acceleration peak after the run
    dwell: float       # run length (s)
    s: float           # progress 0..1
    pose: np.ndarray   # bone directions [K, 3]
    vin: np.ndarray    # unit motion direction entering
    vout: np.ndarray   # unit motion direction leaving
    prominence: float  # 1 - min(v) in the run
    i_stop: int = 0    # frame of t_stop
    i_go: int = 0      # frame of t_go
    ctx_in: np.ndarray = None    # unit pose change from the previous event to this one (context)
    ctx_out: np.ndarray = None   # unit pose change from this event to the next one

    def as_dict(self):
        return dict(type=self.type, t=round(self.t, 3), t_stop=round(self.t_stop, 3),
                    t_go=round(self.t_go, 3), dwell=round(self.dwell, 3), s=round(self.s, 3))


def low_runs(X: Clip, cfg: Config):
    """Low-speed runs with hysteresis, flicker merge and edge attachment -> [(i0, i1)]."""
    low, out, i0, k, n = False, [], 0, 0, len(X.v)
    hold = max(1, int(round(cfg.exit_hold_s / np.median(X.dt))))
    while k < n:
        v = X.v[k]
        if not low and v < cfg.v_low:
            low, i0 = True, k
        elif low and v > cfg.v_exit and (X.v[k:k + hold] > cfg.v_exit).all():   # sustained exit only
            low = False
            out.append((i0, k - 1))
        k += 1
    if low:
        out.append((i0, n - 1))
    merged = []
    for a, b in out:
        if merged and X.t[a] - X.t[merged[-1][1]] < cfg.merge_gap_s:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    n = len(X.t)
    return [(0 if X.t[a] - X.t[0] < cfg.edge_s else a, n - 1 if X.t[-1] - X.t[b] < cfg.edge_s else b)
            for a, b in merged]


def _mean_dir(X: Clip, i0, i1):
    i0, i1 = max(i0, 0), min(i1, len(X) - 1)
    if i1 < i0:
        return np.zeros(X.dir.shape[1])
    d = X.dir[i0:i1 + 1].mean(0)
    return d / max(np.linalg.norm(d), 1e-9)


def detect_events(X: Clip, cfg: Config):
    ev = []
    win = max(1, int(round(cfg.dir_window_s / np.median(X.dt))))
    n = len(X)
    rr = low_runs(X, cfg)
    for ri, (i0, i1) in enumerate(rr):
        dur = X.t[i1] - X.t[i0]
        edge = i0 == 0 or i1 == n - 1
        vin, vout = _mean_dir(X, i0 - win, i0 - 1), _mean_dir(X, i1 + 1, i1 + win)
        if edge:
            typ = "rest"
        elif dur >= cfg.pause_min_s:
            typ = "pause"
        elif float(vin @ vout) < 0:
            typ = "reversal"
        else:
            continue                                   # slow-down without a direction change
        # stop / go = acceleration extremum between the run and the neighbouring speed peak
        # (searching up to the peak, not a fixed window, keeps this tempo-invariant)
        lo = i0
        while lo > 0 and (X.v[lo] < 0.5 or X.v[lo - 1] >= X.v[lo]):
            lo -= 1
        hi = i1
        while hi < n - 1 and (X.v[hi] < 0.5 or X.v[hi + 1] >= X.v[hi]):
            hi += 1
        i_stop = i0 if i0 == 0 else lo + int(np.argmin(X.a[lo:i0 + 1]))
        i_go = i1 if i1 == n - 1 else i1 + int(np.argmax(X.a[i1:hi + 1]))
        if typ == "rest":
            im = i1 if i0 == 0 else i0                 # moment the motion starts / ends
        else:
            im = (i0 + i1) // 2
        ev.append(Event(typ, im, float(X.t[im]), i0, i1, float(X.t[i_stop]), float(X.t[i_go]),
                        float(dur), float(X.s[im]), X.b[im].copy(), vin, vout,
                        float(1 - X.v[i0:i1 + 1].min()), i_stop, i_go))
    # context: which movement led into / out of each checkpoint (distinguishes e.g. "standing up from a
    # squat" from "standing up from a bow", which share the same pose)
    for k, e in enumerate(ev):
        def unit(v):
            v = v.ravel(); nv = np.linalg.norm(v)
            return v / nv if nv > 1e-6 else np.zeros_like(v)
        e.ctx_in = unit(e.pose - ev[k - 1].pose) if k else np.zeros(e.pose.size)
        e.ctx_out = unit(ev[k + 1].pose - e.pose) if k + 1 < len(ev) else np.zeros(e.pose.size)
    return ev
