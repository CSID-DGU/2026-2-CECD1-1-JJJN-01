"""[6] Playback: which frame of each clip to draw at every instant, to watch the two skeletons side by side.

The frame alignment of step 4 pairs the frames that look alike. That is what the pose labels need, but
played back it stops and jumps, and it shows neither a difference in speed nor a movement only the coach
makes. Playback is read from the units of step 5 instead, one rule per structure:
  match         both start together and move at their own real speed; whoever arrives first waits in the
                pose the unit ends with, so a slower movement is seen as slower
  substitution  the same, and the two meet once more at the two checkpoints that were put against each other
  extra         the student moves, the coach waits
  missing       the coach moves, the student waits
Frames that belong to no unit (the turn at a reversal, the rests at the clip edges, a way back nobody owns)
are played just before the clip's next unit, or earlier while that clip has to wait for the other one: a
coach who waits through an extra movement goes on standing in his own frames instead of freezing. So every
frame of both clips is shown once, in order, and every unit starts with the two in step.

The clock is the student's frame period: a clip with another frame rate is shown at its real speed too.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .step5_segments import build_units


def _ticks(t, span, dt):
    """Instants needed to play the frames span = (a, b) of a clip with timestamps t (0 for no span)."""
    return 0 if span is None else int(round(float(t[span[1]] - t[span[0]]) / dt)) + 1


def _frames(t, span, wait, n, dt):
    """The frame drawn at each of n instants: span played from its first frame at real speed and then held at
    its last one, or the frame `wait` throughout if the clip has nothing to play."""
    if span is None:
        return [int(wait)] * n
    a, b = span
    at = np.minimum(t[a] + np.arange(n) * dt, t[b])
    return (a + np.abs(t[None, a:b + 1] - at[:, None]).argmin(1)).tolist()


def build_playback(res, cfg: Config):
    """-> dict(dt, coach, student, segments).
    coach / student   the frame of each clip to draw at instant k (k * dt seconds), equally long
    segments          per unit of Result.units(), in the same order: its (first, last) instant, or None if
                      nothing is left of it to play"""
    C, S = res.C, res.S
    units = build_units(res, cfg, private=True)[0]
    dt = float(np.median(S.dt))
    out_c, out_s, spans = [], [], []
    pc = ps = 0                                        # the first frame of each clip that has not been played
    free_c, free_s = len(C) - 1, len(S) - 1            # the last frame each clip may show before its next unit

    def play(c_to, s_to):
        """both clips from where they are up to frame c_to / s_to. None: that clip has nothing to play here; it
        goes on through its frames that belong to no unit (up to free_c / free_s) and then waits."""
        nonlocal pc, ps
        c = (pc, c_to) if c_to is not None and c_to >= pc else None
        s = (ps, s_to) if s_to is not None and s_to >= ps else None
        n = max(_ticks(C.t, c, dt), _ticks(S.t, s, dt))
        if n:
            fc = _frames(C.t, c or ((pc, free_c) if free_c >= pc else None), max(pc - 1, 0), n, dt)
            fs = _frames(S.t, s or ((ps, free_s) if free_s >= ps else None), max(ps - 1, 0), n, dt)
            out_c.extend(fc), out_s.extend(fs)
            pc, ps = max(pc, fc[-1] + 1), max(ps, fs[-1] + 1)

    def before_next(key, k, last):
        nxt = next((u[key][0] for u in units[k:] if u[key]), None)
        return last if nxt is None else nxt - 1
    for k, u in enumerate(units):
        c, s = u["_c"], u["_s"]
        free_c, free_s = before_next("_c", k, len(C) - 1), before_next("_s", k, len(S) - 1)
        play(c[0] - 1 if c else None, s[0] - 1 if s else None)             # what nobody has played yet
        first = len(out_c)
        if u.get("_mid") and c and s:                                      # substitution: meet at its checkpoints
            play(min(u["_mid"][0], c[1]), min(u["_mid"][1], s[1]))
        play(c[1] if c else None, s[1] if s else None)
        spans.append((first, len(out_c) - 1) if len(out_c) > first else None)
    play(len(C) - 1, len(S) - 1)                                           # what is left of the two clips
    return dict(dt=dt, coach=out_c, student=out_s, segments=spans)
