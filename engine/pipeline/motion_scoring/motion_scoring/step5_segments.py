"""[5] Segments + labels. Non-overlapping segments on the student timeline (the scoring output handed to the LLM).

The matched checkpoints, between the two clip edges, cut both clips into units:
  hold units   the stillness at a matched checkpoint (coach pause <-> student pause)
  move units   the movement from one matched checkpoint (or the clip start) to the next (or the clip end)
Each unit carries exactly one structure label
  match         both sides do it                    -> timing (slow / fast, ratio) and pose attached
  substitution  same place, different action
  missing       coach only (incl. a coach hold the student does not make) -> student_t is where it belongs
  extra         student only
and its times on both videos. Nothing else (no phase, no sentences): the LLM gets the video and these facts.

A unit that reaches a clip edge which is not a matched rest carries edge = "start" | "end" | "both", and
cut = the clips that are already / still moving there: its movement is not complete in the video, so it
gets a pose label but no timing.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .step1_preprocess import pose_dist
from .step4_frame_align import DROPS

PRIVATE = ("_c", "_s", "_wc", "_ws", "_tc", "_ts", "_short", "_untimed")    # working keys, not in the output


def motion_summary(clip, t0, t1, ref_t=None, top=4):
    """Bones that move most between t0 and t1 (from the pose at ref_t): [{bone, deg, dir}],
    dir = up | down | forward | backward (largest change of the bone direction)."""
    m = (clip.t >= t0) & (clip.t <= t1)
    if m.sum() < 2:
        return []
    b = clip.b[m]
    start = clip.b[int(np.argmin(np.abs(clip.t - (t0 if ref_t is None else ref_t))))]
    ang = np.degrees(np.arccos(np.clip((b * start).sum(-1), -1, 1)))
    peak = ang.max(0)
    out = []
    for k in np.argsort(-peak)[:top]:
        i = int(np.argmax(ang[:, k]))
        dy, dz = b[i, k, 1] - start[k, 1], b[i, k, 2] - start[k, 2]
        d = ("up" if dy > 0 else "down") if abs(dy) >= abs(dz) else ("forward" if dz > 0 else "backward")
        out.append(dict(bone=clip.names[k], deg=round(float(peak[k]), 1), dir=d))
    return out


def _span(X, a, b):
    return [round(float(X.t[a]), 3), round(float(X.t[b]), 3)]


def build_units(res, cfg):
    """-> (units, global tempo)."""
    units = _merge(_structure(res, cfg), res)
    tempo = _timing(units, res, cfg)
    _pose(units, res, cfg)
    for u in units:                                    # what moves in structural units (data, not sentences)
        if u["structure"] in ("substitution", "missing") and u["_c"]:
            u["coach_motion"] = motion_summary(res.C, *u["coach_t"], ref_t=u["coach_t"][0])
        if u["structure"] in ("substitution", "extra") and u["_s"]:
            u["student_motion"] = motion_summary(res.S, *u["student_t"], ref_t=u["student_t"][0])
    for u in units:
        for k in PRIVATE:
            u.pop(k, None)
    return units, tempo


# ---------------------------------------------------------------- structure
def _at(events, k):
    """Checkpoint k of one clip as a node side. w = the frame that bounds a timing window there: the
    checkpoint itself, or the clip edge for a rest at it (a slow onset hidden below the speed threshold
    is then still inside the window)."""
    e = events[k]
    w = e.i0 if e.type == "rest" and k == 0 else e.i1 if e.type == "rest" and k == len(events) - 1 else e.i
    return SimpleNamespace(k=k, type=e.type, i0=e.i0, i=e.i, i1=e.i1, t=e.t, w=w, pose=e.pose)


def _clip_edge(X, events, start, taken):
    """Where the movement of clip X begins (start) / ends, as a node side, and whether the clip rests there.
    It is the end of the rest at that edge, or the first / last frame if the clip is already / still moving
    (cut). taken: that rest is itself the first / last matched checkpoint, so nothing lies beyond it."""
    n = len(X)
    k = 0 if start else len(events) - 1
    rests = bool(events) and events[k].type == "rest" and (events[k].i0 == 0 if start else events[k].i1 == n - 1)
    f = 0 if start else n - 1
    if rests and not taken:
        f = events[k].i1 if start else events[k].i0
    return SimpleNamespace(k=None, type="edge", i0=f, i=f, i1=f, t=float(X.t[f]), w=0 if start else n - 1,
                           pose=X.b[f]), rests


def _nodes(res):
    """The matched checkpoints in order, between the clip start and the clip end (always there)."""
    C, S, D = res.C, res.S, res.debug
    eC, eS, ops = D["eC"], D["eS"], D["ops"]
    real = [SimpleNamespace(n=n, c=_at(eC, o["i"]), s=_at(eS, o["j"]), edge=None, cut=[])
            for n, o in enumerate(ops) if o["kind"] == "match"]

    def edge(start):
        m = real[0 if start else -1] if real else None
        last = lambda ev: 0 if start else len(ev) - 1
        c, c_rests = _clip_edge(C, eC, start, m is not None and m.c.k == last(eC))
        s, s_rests = _clip_edge(S, eS, start, m is not None and m.s.k == last(eS))
        cut = [name for name, rests in (("coach", c_rests), ("student", s_rests)) if not rests]
        return SimpleNamespace(n=-1 if start else len(ops), c=c, s=s, edge="start" if start else "end", cut=cut)
    return [edge(True)] + real + [edge(False)]


def _gap_ops(D, a, b):
    """The unmatched checkpoints between nodes a and b, without the rests: stillness at a clip edge is
    not an action (a rest that is not matched leaves its side of the op empty)."""
    eC, eS = D["eC"], D["eS"]
    out = []
    for o in D["ops"][a.n + 1:b.n]:
        i = o["i"] if o["i"] is not None and eC[o["i"]].type != "rest" else None
        j = o["j"] if o["j"] is not None and eS[o["j"]].type != "rest" else None
        if i is not None or j is not None:
            out.append(dict(kind="substitution" if None not in (i, j) else "missing" if j is None else "extra",
                            i=i, j=j))
    return out


def _moves(X, f0, f1):
    """A real movement happens in frames f0 .. f1."""
    return f1 > f0 and float(X.v[f0:f1 + 1].max()) >= X.v_peak


def _hold(C, S, a):
    """The stillness at a matched checkpoint (clip-boundary rests are not judged)."""
    ec, es = a.c, a.s
    if "rest" in (ec.type, es.type):
        return []
    if ec.type == "pause" and es.type == "pause":
        return [dict(structure="match", coach_t=_span(C, ec.i0, ec.i1), student_t=_span(S, es.i0, es.i1),
                     _c=(ec.i0, ec.i1), _s=(es.i0, es.i1))]
    if ec.type == "pause":
        # the student only slows down here (shorter than pause_min_s): re-judged by the duration ratio
        # once the global tempo is known (short holds in _timing)
        return [dict(structure="missing", coach_t=_span(C, ec.i0, ec.i1), student_t=[round(es.t, 3)] * 2,
                     _c=(ec.i0, ec.i1), _s=None, _short=dict(c=(ec.i0, ec.i1), s=(es.i0, es.i1)))]
    if es.type == "pause":
        return [dict(structure="extra", coach_t=None, student_t=_span(S, es.i0, es.i1),
                     _c=None, _s=(es.i0, es.i1))]
    return []


def _shared(C, S, c, s, wc, ws, at_edge):
    """The unit of a stretch without unmatched checkpoints (coach frames c, student frames s; wc / ws the
    timing windows): a movement both sides make. Next to a clip edge only what really moves counts, and
    one side may have nothing there (its clip starts later / ends earlier)."""
    if at_edge:
        moves_c, moves_s = _moves(C, *c), _moves(S, *s)
        if not moves_c and not moves_s:
            return []
        if not moves_s:
            return [dict(structure="missing", coach_t=_span(C, *c), student_t=_span(S, s[0], s[0]), _c=c, _s=None)]
        if not moves_c:
            return [dict(structure="extra", coach_t=None, student_t=_span(S, *s), _c=None, _s=s)]
    return [dict(structure="match", coach_t=_span(C, *c), student_t=_span(S, *s), _c=c, _s=s, _wc=wc, _ws=ws)]


def _stretch(res, cfg, a, b):
    """The units between two consecutive nodes: one shared movement, or cut at every unmatched checkpoint
    in between (substitution / missing / extra)."""
    C, S, D = res.C, res.S, res.debug
    eC, eS = D["eC"], D["eS"]
    w = np.ones(len(C.names))
    c0, c1, s0, s1 = a.c.i1, b.c.i0, a.s.i1, b.s.i0
    gap = _gap_ops(D, a, b)
    at_edge = a.edge or b.edge
    if not gap:
        units = _shared(C, S, (c0, c1), (s0, s1), (a.c.w, b.c.w), (a.s.w, b.s.w), at_edge)
    else:
        units = []
        cur_c, cur_s = c0, s0
        last_c = max((o["i"] for o in gap if o["i"] is not None), default=None)   # the last unmatched
        last_s = max((o["j"] for o in gap if o["j"] is not None), default=None)   # checkpoint of each side

        def end_of(events, k, last, cap):
            """a unit ends with the stillness / dip of its checkpoint. A reversal followed by another
            unmatched checkpoint also owns the way to it (the bottom of an extra repetition: down and up
            again); what follows the last unmatched checkpoint is decided below."""
            if events[k].type == "reversal" and k != last:
                return min(events[k + 1].i0, cap)
            return min(events[k].i1, cap)
        for o in gap:
            end_c = end_of(eC, o["i"], last_c, c1) if o["i"] is not None else cur_c
            end_s = end_of(eS, o["j"], last_s, s1) if o["j"] is not None else cur_s
            units.append(dict(structure=o["kind"],
                              coach_t=_span(C, cur_c, end_c) if o["i"] is not None else None,
                              student_t=_span(S, cur_s, end_s) if o["j"] is not None else _span(S, cur_s, cur_s),
                              _c=(cur_c, end_c) if o["i"] is not None else None,
                              _s=(cur_s, end_s) if o["j"] is not None else None))
            cur_c, cur_s = end_c, end_s

        def returns(last, base, nxt):
            """what follows the last unmatched checkpoint ends back at the place of node a -> it belongs
            to the unmatched action (e.g. lowering the arms after a missing raise). Not if that checkpoint
            is itself back at node a (a whole extra repetition): then a new, shared movement follows."""
            return pose_dist(nxt, base, w) < cfg.same_place * pose_dist(nxt, last, w)
        u = units[-1]
        back_c = last_c is not None and returns(eC[last_c].pose, a.c.pose, b.c.pose)
        back_s = last_s is not None and returns(eS[last_s].pose, a.s.pose, b.s.pose)
        if back_c or back_s or u["structure"] == "substitution":
            if u["_c"] is not None:
                u["_c"] = (u["_c"][0], c1); u["coach_t"] = _span(C, *u["_c"])
            if u["_s"] is not None:
                u["_s"] = (u["_s"][0], s1); u["student_t"] = _span(S, *u["_s"])
        else:                                          # what is left is a shared movement
            units += _shared(C, S, (cur_c, c1), (cur_s, s1), (cur_c, b.c.w), (cur_s, b.s.w), at_edge)
    if at_edge:
        cut = sorted(set(a.cut) | set(b.cut))
        for u in units:
            u["edge"] = "both" if a.edge and b.edge else at_edge
            if cut:
                u["cut"], u["_untimed"] = cut, True    # the movement is not complete in the video
    return units


def _structure(res, cfg):
    nodes = _nodes(res)
    units = []
    for m, a in enumerate(nodes):
        if a.edge is None:
            units += _hold(res.C, res.S, a)
        if m + 1 < len(nodes):
            units += _stretch(res, cfg, a, nodes[m + 1])
    return units


def _merge(units, res):
    """Consecutive structural units of the same kind are one (e.g. two missing checkpoints in a row)."""
    merged = []
    for u in units:
        p = merged[-1] if merged else None
        if p is not None and u["structure"] == p["structure"] and u["structure"] in ("missing", "extra"):
            for key, kk, X in (("_c", "coach_t", res.C), ("_s", "student_t", res.S)):
                if p[key] is not None and u[key] is not None:
                    p[key] = (p[key][0], u[key][1]); p[kk] = _span(X, *p[key])
            continue
        merged.append(u)
    return merged


# ---------------------------------------------------------------- labels
def _timing(units, res, cfg):
    """timing on match units: duration ratio relative to the global tempo (median over the match units,
    1.0 if there are fewer than tempo_min_units of them) -> the global tempo."""
    C, S = res.C, res.S

    def dur(X, ab, win=None):
        """hold: run length. movement: time between 10 % and 90 % of its own pose-path length (robust to
        noise and to where the speed threshold cuts a slow onset)."""
        if win is None:
            return float(X.t[ab[1]] - X.t[ab[0]])
        f0, f1 = win
        arc = X.arc[f0:f1 + 1] - X.arc[f0]
        if f1 - f0 < 3 or arc[-1] <= 1e-6:
            return float(X.t[ab[1]] - X.t[ab[0]])
        arc = arc + np.arange(len(arc)) * 1e-9
        t10, t90 = (float(np.interp(q * arc[-1], arc, X.t[f0:f1 + 1])) for q in (0.1, 0.9))
        return (t90 - t10) / 0.8
    for u in units:
        if u["structure"] == "match" and u["_c"] is not None and not u.get("_untimed"):
            u["_tc"] = dur(C, u["_c"], u.get("_wc"))
            u["_ts"] = dur(S, u["_s"], u.get("_ws"))
    ratios = [u["_ts"] / u["_tc"] for u in units if "_tc" in u and u["_tc"] > 0.15]
    # a few units do not define a tempo (the median of two or three ratios is one of the units itself,
    # and the others would carry the labels): their durations are then compared with the coach's as they are
    r_global = float(np.median(ratios)) if len(ratios) >= cfg.tempo_min_units else 1.0
    # short holds: a coach pause the student does only briefly is judged by the duration ratio
    #   student slow-down / (coach pause x global tempo), not present / absent at pause_min_s:
    #   >= hold_missing_ratio -> a held pause (timed below like any match: ok, or fast = shorter);
    #   below -> stays missing. Both keep hold = {coach_s, student_s, ratio}.
    for u in units:
        sh = u.get("_short")
        if sh is None or u["structure"] != "missing" or u["_c"] != sh["c"]:
            continue                                    # merged into a larger missing stretch: leave it
        tc = float(C.t[sh["c"][1]] - C.t[sh["c"][0]])
        ts = float(S.t[sh["s"][1]] - S.t[sh["s"][0]])
        rel = ts / (tc * r_global) if tc > 0 else 0.0
        u["hold"] = dict(coach_s=round(tc, 2), student_s=round(ts, 2), ratio=round(rel, 2))
        if rel >= cfg.hold_missing_ratio:
            u.update(structure="match", student_t=_span(S, *sh["s"]), _s=sh["s"], _tc=tc, _ts=ts)
    lo, hi = cfg.rate_ok
    for u in units:
        u["timing"] = None
        if u["structure"] != "match" or "_tc" not in u:
            continue
        tc, ts = u["_tc"], u["_ts"]
        if min(tc, ts) <= 0.05:
            continue                                    # nothing to time on one side
        rel = ts / tc / r_global
        if abs(ts - tc * r_global) >= cfg.dwell_tol_s and (rel > hi or rel < lo):
            u["timing"] = dict(label="slow" if rel > hi else "fast", ratio=round(rel, 2),
                               coach_s=round(tc, 2), student_s=round(ts, 2))
    return r_global


def _pose(units, res, cfg):
    """pose on match units: sustained d > tau on the coach frames of the unit."""
    C, S, D = res.C, res.S, res.debug
    d, tau = D["d"], cfg.tau_deg
    pairs = {}
    for k, i, j in D["path"]:
        if k not in DROPS:
            pairs.setdefault(i, []).append(j)
    for u in units:
        u["pose"] = None
        if u["structure"] != "match" or u["_c"] is None:
            continue
        unit_s = float(C.t[u["_c"][1]] - C.t[u["_c"][0]])
        if unit_s <= 0.05:
            continue
        idx = np.arange(u["_c"][0], u["_c"][1] + 1)
        dm = np.array([min(d[i, j] for j in pairs[i]) if i in pairs else np.nan for i in idx])
        bad = np.nan_to_num(dm, nan=0) > tau
        run, best = 0, 0
        for b in bad:
            run = run + 1 if b else 0
            best = max(best, run)
        # sustained = a share of this unit's own (coach) duration, so a quick movement and a long hold are
        # judged alike; the floor keeps a frame or two of jitter from becoming a label
        if best * float(np.median(C.dt)) >= max(cfg.min_label_frac * unit_s, cfg.min_label_floor_s):
            diffs = [np.degrees(np.arccos(np.clip((C.b[i] * S.b[pairs[i][0]]).sum(-1), -1, 1)))
                     for i, b in zip(idx, bad) if b and i in pairs]
            mdiff = np.mean(diffs, axis=0)
            order = np.argsort(-mdiff)[:2]
            u["pose"] = dict(worst=[dict(bone=C.names[k], diff_deg=round(float(mdiff[k]), 1)) for k in order],
                             mean_diff_deg=round(float(np.nanmean(dm[bad])), 1))
