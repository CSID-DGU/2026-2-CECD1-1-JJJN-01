"""[2] Checkpoints: where one movement ends and the next begins.

  rest      stillness (low-speed run) touching a clip edge
  pause     stillness of at least pause_min_s
  reversal  a speed dip without a pause: the pose turns back where it came from (with or without a short
            stop), or a short stop is followed by a different movement (turns)
A dip after which the same movement carries on is a hesitation, not a checkpoint.
The speed decides what happened; the acceleration marks when it started / ended (stop / go).
"""
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
    i0: int            # first frame of the stillness (rest / pause) or of the dip (reversal)
    i1: int            # last frame of it
    t_stop: float      # deceleration peak before it (acceleration minimum)
    t_go: float        # acceleration peak after it
    dwell: float       # i0 .. i1 in seconds
    s: float           # progress 0..1
    pose: np.ndarray   # bone directions [K, 3]
    vin: np.ndarray    # unit motion direction entering
    vout: np.ndarray   # unit motion direction leaving
    prominence: float  # 1 - min(v) in i0 .. i1
    i_stop: int = 0    # frame of t_stop
    i_go: int = 0      # frame of t_go
    ctx_in: np.ndarray = None    # unit pose change from the previous event to this one (context)
    ctx_out: np.ndarray = None   # unit pose change from this event to the next one

    def as_dict(self):
        return dict(type=self.type, t=round(self.t, 3), t_stop=round(self.t_stop, 3),
                    t_go=round(self.t_go, 3), dwell=round(self.dwell, 3), s=round(self.s, 3))


def low_runs(X: Clip, cfg: Config):
    """Low-speed runs with hysteresis, flicker merge and edge attachment -> [(i0, i1)].
    The levels are the clip's own (X.v_low / X.v_exit: the config values, raised for noisy input)."""
    low, out, i0, k, n = False, [], 0, 0, len(X.v)
    hold = max(1, int(round(cfg.exit_hold_s / np.median(X.dt))))
    while k < n:
        v = X.v[k]
        if not low and v < X.v_low:
            low, i0 = True, k
        elif low and v > X.v_exit and (X.v[k:k + hold] > X.v_exit).all():      # sustained exit only
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
    return [(0 if X.t[a] - X.t[0] < cfg.edge_s else a, n - 1 if X.t[-1] - X.t[b] < cfg.edge_s else b)
            for a, b in merged]


def _mean_dir(X: Clip, i0, i1):
    i0, i1 = max(i0, 0), min(i1, len(X) - 1)
    if i1 < i0:
        return np.zeros(X.dir.shape[1])
    d = X.dir[i0:i1 + 1].mean(0)
    return d / max(np.linalg.norm(d), 1e-9)


def _wall(v, f, step, floor, stop):
    """Highest speed on one side of a dip (step = -1 / +1 from frame f), up to the next frame that is as
    slow as the dip (v <= floor) and never reaching frame `stop`: the peak of the movement that separates
    this dip from the next one as deep. A smaller bump on the way is passed over, and nothing depends on
    a time window (tempo-invariant)."""
    best = i = f
    while i + step != stop and 0 <= i + step < len(v):
        i += step
        if v[i] <= floor:
            break
        if v[i] > v[best]:
            best = i
    return best


def turns(X: Clip, cfg: Config, stops, still):
    """Reversals -> [(i0, i, i1)]: speed dips at which one movement ends and another begins.

    A dip is a short stop (`stops`: the low-speed runs too short to be a pause) or, where the body never
    quite stops, a local speed minimum outside the low-speed runs. The second kind is why the shape of the
    dip is judged and not its level: straight back up from the bottom of a squat, the smoothed speed only
    falls to 0.12-0.16 of the reference on a real clip, right at v_low, so a level test finds the same turn
    or not depending on tempo, noise and frame rate.
      shape      the dip is below dip_ratio x the lower of its two walls (_wall), and both walls are peaks
                 of real movements (at least X.v_peak)
      direction  cos between the pose change over the half-movement before the dip and over the same time
                 after it (the nearer wall sets the time, so only what happens at the dip counts):
                   below dip_cos    the body goes back where it just came from: a turn
                   below carry_cos  a different movement follows: a checkpoint if the body stops for it
                                    (a short stop); above it the same movement carries on (a hesitation)
    `still` (the rests and pauses) is skipped and never crossed when looking for the walls."""
    if cfg.dip_ratio <= 0:
        return []
    v, n = X.v, len(X.v)
    near = max(1, int(round(cfg.merge_gap_s / float(np.median(X.dt)))))    # dips closer than this are one dip
    pose = (X.b * np.sqrt(X.w)[None, :, None]).reshape(n, -1)              # weighted like the speed
    low = np.zeros(n, bool)                                                # inside any low-speed run
    for a, b in list(stops) + list(still):
        low[a:b + 1] = True
    dips = [(a, (a + b) // 2, b, True) for a, b in stops]                  # (first, centre, last frame, stopped)
    for m in range(1, n - 1):
        if not low[m] and v[m] <= v[max(m - near, 0):m + near + 1].min():
            i0 = i1 = m                                                    # the frames around the minimum
            while i0 > 0 and not low[i0 - 1] and v[i0 - 1] < 1.25 * v[m]:
                i0 -= 1
            while i1 < n - 1 and not low[i1 + 1] and v[i1 + 1] < 1.25 * v[m]:
                i1 += 1
            dips.append((i0, m, i1, False))
    bounds = sorted({-1, n} | {a for a, _ in still} | {b for _, b in still})
    found = {}                                         # (left wall, right wall) -> the deepest dip between them
    for i0, m, i1, stopped in dips:
        left, right = max(x for x in bounds if x < m), min(x for x in bounds if x > m)
        depth = float(v[i0:i1 + 1].min())
        floor = max(depth, X.v_low)                    # "as slow as this dip": as deep, or still
        pl, pr = _wall(v, i0, -1, floor, left), _wall(v, i1, +1, floor, right)
        peak = min(v[pl], v[pr])
        if peak < X.v_peak or depth > cfg.dip_ratio * peak:
            continue                                   # no real movement on one side, or not a clear dip
        k = min(m - pl, pr - m)
        d_in, d_out = pose[m] - pose[m - k], pose[m + k] - pose[m]
        cos = float(d_in @ d_out) / max(float(np.linalg.norm(d_in) * np.linalg.norm(d_out)), 1e-9)
        if cos >= (cfg.carry_cos if stopped else cfg.dip_cos):
            continue                                   # a hesitation, or a change of movement without a stop
        if (pl, pr) not in found or depth < found[pl, pr][0]:
            found[pl, pr] = (depth, i0, m, i1)
    out = []
    for depth, i0, m, i1 in sorted(found.values(), key=lambda d: d[2]):
        if out and m - out[-1][2] < 2 * near:          # the same dip seen between two pairs of walls
            if depth >= out[-1][0]:
                continue
            out.pop()
        out.append((depth, i0, m, i1))
    return [(i0, m, i1) for _, i0, m, i1 in out]


def detect_events(X: Clip, cfg: Config):
    n = len(X)
    win = max(1, int(round(cfg.dir_window_s / np.median(X.dt))))
    runs = low_runs(X, cfg)
    still = [(i0, i1) for i0, i1 in runs if i0 == 0 or i1 == n - 1 or X.t[i1] - X.t[i0] >= cfg.pause_min_s]
    spans = [(i0, None, i1) for i0, i1 in still] + turns(X, cfg, [r for r in runs if r not in still], still)
    ev = []
    for i0, turn, i1 in sorted(spans, key=lambda s: s[0]):
        if turn is not None:
            typ, im = "reversal", turn
        elif i0 == 0 or i1 == n - 1:
            typ, im = "rest", (i1 if i0 == 0 else i0)  # moment the motion starts / ends
        else:
            typ, im = "pause", (i0 + i1) // 2
        vin, vout = _mean_dir(X, i0 - win, i0 - 1), _mean_dir(X, i1 + 1, i1 + win)
        # stop / go = acceleration extremum between the checkpoint and the neighbouring speed peak
        # (searching up to the peak, not a fixed window, keeps this tempo-invariant)
        lo = i0
        while lo > 0 and (X.v[lo] < X.v_peak or X.v[lo - 1] >= X.v[lo]):
            lo -= 1
        hi = i1
        while hi < n - 1 and (X.v[hi] < X.v_peak or X.v[hi + 1] >= X.v[hi]):
            hi += 1
        i_stop = i0 if i0 == 0 else lo + int(np.argmin(X.a[lo:i0 + 1]))
        i_go = i1 if i1 == n - 1 else i1 + int(np.argmax(X.a[i1:hi + 1]))
        ev.append(Event(typ, im, float(X.t[im]), i0, i1, float(X.t[i_stop]), float(X.t[i_go]),
                        float(X.t[i1] - X.t[i0]), float(X.s[im]), X.b[im].copy(), vin, vout,
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
