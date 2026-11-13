"""[5] Segments + labels. Non-overlapping segments on the student timeline (the scoring output handed to the LLM).

Every matched checkpoint and every stretch between two consecutive matched checkpoints becomes a unit:
  hold units   the stillness at a matched checkpoint (coach pause <-> student pause)
  move units   the movement from one matched checkpoint to the next
Each unit carries exactly one structure label
  match         both sides do it                    -> timing (slow / fast, ratio) and pose attached
  substitution  same place, different action
  missing       coach only (incl. a coach hold the student does not make) -> student_t is where it belongs
  extra         student only
and its times on both videos. Nothing else (no phase, no sentences): the LLM gets the video and these facts.
"""
from __future__ import annotations

import numpy as np

from .step1_preprocess import pose_dist


PARENT = {"forearm": "upper_arm", "shank": "thigh", "head": "trunk"}


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
    C, S, D = res.C, res.S, res.debug
    eC, eS, ops = D["eC"], D["eS"], D["ops"]
    d, tau = D["d"], cfg.tau_deg
    pairs = {}
    for k, i, j in D["path"]:
        if k not in ("drop_c", "drop_s"):
            pairs.setdefault(i, []).append(j)
    w = np.ones(len(C.names))

    matched = [(n, o["i"], o["j"]) for n, o in enumerate(ops) if o["kind"] == "match"]
    units = []

    def returns(events, k_last, k_base, k_next):
        """the movement from the last unmatched checkpoint to the next matched one goes back towards the
        pose of the previous matched checkpoint -> it belongs to the unmatched action."""
        return (pose_dist(events[k_next].pose, events[k_base].pose, w)
                < pose_dist(events[k_next].pose, events[k_last].pose, w))

    for m, (n_a, i_a, j_a) in enumerate(matched):
        ec, es = eC[i_a], eS[j_a]
        # ---- hold unit at this matched checkpoint (clip-boundary rests are not judged)
        if ec.type != "rest" and es.type != "rest":
            if ec.type == "pause" and es.type == "pause":
                units.append(dict(structure="match", coach_t=_span(C, ec.i0, ec.i1), student_t=_span(S, es.i0, es.i1),
                                  _c=(ec.i0, ec.i1), _s=(es.i0, es.i1)))
            elif ec.type == "pause":
                # the student only slows down here (a run shorter than pause_min_s): re-judged by the duration
                # ratio once the global tempo is known (see "short holds" below)
                units.append(dict(structure="missing", coach_t=_span(C, ec.i0, ec.i1),
                                  student_t=[round(es.t, 3)] * 2, _c=(ec.i0, ec.i1), _s=None,
                                  _short=dict(c=(ec.i0, ec.i1), s=(es.i0, es.i1))))
            elif es.type == "pause":
                units.append(dict(structure="extra", coach_t=None, student_t=_span(S, es.i0, es.i1),
                                  _c=None, _s=(es.i0, es.i1)))
        if m + 1 == len(matched):
            break
        # ---- the stretch to the next matched checkpoint, cut at every unmatched checkpoint in between
        n_b, i_b, j_b = matched[m + 1]
        c0, c1 = eC[i_a].i1, eC[i_b].i0
        s0, s1 = eS[j_a].i1, eS[j_b].i0
        # timing window for a plain movement: between the checkpoint frames (a rest at the clip edge
        # extends to the edge, so a slow onset hidden below the speed threshold is still inside)
        wc = (eC[i_a].i0 if eC[i_a].type == "rest" and i_a == 0 else eC[i_a].i,
              eC[i_b].i1 if eC[i_b].type == "rest" and i_b == len(eC) - 1 else eC[i_b].i)
        ws = (eS[j_a].i0 if eS[j_a].type == "rest" and j_a == 0 else eS[j_a].i,
              eS[j_b].i1 if eS[j_b].type == "rest" and j_b == len(eS) - 1 else eS[j_b].i)
        gap = ops[n_a + 1:n_b]
        if not gap:
            units.append(dict(structure="match", coach_t=_span(C, c0, c1), student_t=_span(S, s0, s1),
                              _c=(c0, c1), _s=(s0, s1), _wc=wc, _ws=ws))
            continue
        cur_c, cur_s = c0, s0
        last_c, last_s = i_a, j_a
        for o in gap:
            kind = {"substitute": "substitution", "missing": "missing", "extra": "extra"}[o["kind"]]
            # a pause ends with its stillness; a reversal (e.g. top of a raise) also owns the way back,
            # up to the next checkpoint on that side
            def end_of(events, k, cap):
                e = events[k]
                if e.type == "reversal" and k + 1 < len(events):
                    return min(events[k + 1].i0, cap)
                return min(e.i1, cap)
            end_c = end_of(eC, o["i"], c1) if o["i"] is not None else cur_c
            end_s = end_of(eS, o["j"], s1) if o["j"] is not None else cur_s
            units.append(dict(structure=kind,
                              coach_t=_span(C, cur_c, end_c) if o["i"] is not None else None,
                              student_t=_span(S, cur_s, end_s) if o["j"] is not None else _span(S, cur_s, cur_s),
                              _c=(cur_c, end_c) if o["i"] is not None else None,
                              _s=(cur_s, end_s) if o["j"] is not None else None))
            if o["i"] is not None:
                cur_c, last_c = end_c, o["i"]
            if o["j"] is not None:
                cur_s, last_s = end_s, o["j"]
        # what is left before the next matched checkpoint: a return movement belongs to the last structural
        # unit (e.g. lowering the arms after a missing raise); otherwise it is a shared movement
        u = units[-1]
        back_c = last_c != i_a and returns(eC, last_c, i_a, i_b)
        back_s = last_s != j_a and returns(eS, last_s, j_a, j_b)
        if back_c or back_s or u["structure"] == "substitution":
            if u["_c"] is not None:
                u["_c"] = (u["_c"][0], c1); u["coach_t"] = _span(C, *u["_c"])
            if u["_s"] is not None:
                u["_s"] = (u["_s"][0], s1); u["student_t"] = _span(S, *u["_s"])
            if back_c and not back_s and cur_s < s1 and u["structure"] == "missing":
                units.append(dict(structure="match", coach_t=None, student_t=_span(S, cur_s, s1), _c=None, _s=None))
                units.pop()                                     # nothing shared: the student just waits
        else:
            units.append(dict(structure="match", coach_t=_span(C, cur_c, c1), student_t=_span(S, cur_s, s1),
                              _c=(cur_c, c1), _s=(cur_s, s1), _wc=(cur_c, wc[1]), _ws=(cur_s, ws[1])))

    # ---- merge consecutive structural units of the same kind (e.g. two missing checkpoints in a row)
    merged = []
    for u in units:
        p = merged[-1] if merged else None
        if p is not None and u["structure"] == p["structure"] and u["structure"] in ("missing", "extra"):
            for key, kk, X in (("_c", "coach_t", C), ("_s", "student_t", S)):
                if p[key] is not None and u[key] is not None:
                    p[key] = (p[key][0], u[key][1]); p[kk] = _span(X, *p[key])
            continue
        merged.append(u)
    units = merged

    # ---- timing on match units: duration ratio relative to the global tempo (median over match units)
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
        if u["structure"] == "match" and u["_c"] is not None:
            u["_tc"] = dur(C, u["_c"], u.get("_wc"))
            u["_ts"] = dur(S, u["_s"], u.get("_ws"))
    ratios = [u["_ts"] / u["_tc"] for u in units if "_tc" in u and u["_tc"] > 0.15]
    r_global = float(np.median(ratios)) if ratios else 1.0
    # ---- short holds: a coach pause the student does only briefly is judged by the duration ratio
    #      student slow-down / (coach pause x global tempo), not present / absent at pause_min_s:
    #      >= hold_missing_ratio -> a held pause (timed below like any match: ok, or fast = shorter);
    #      below -> stays missing. Both keep hold = {coach_s, student_s, ratio}.
    for u in units:
        sh = u.pop("_short", None)
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
        u["pose"] = None
        if u["structure"] != "match" or "_tc" not in u:
            continue
        tc, ts = u["_tc"], u["_ts"]
        if tc <= 0.05:
            continue
        rel = ts / tc / r_global
        if abs(ts - tc * r_global) >= cfg.dwell_tol_s and (rel > hi or rel < lo):
            u["timing"] = dict(label="slow" if rel > hi else "fast", ratio=round(rel, 2),
                               coach_s=round(tc, 2), student_s=round(ts, 2))
        # ---- pose: sustained d > tau on the coach frames of this unit
        idx = np.arange(u["_c"][0], u["_c"][1] + 1)
        dm = np.array([min(d[i, j] for j in pairs[i]) if i in pairs else np.nan for i in idx])
        bad = np.nan_to_num(dm, nan=0) > tau
        run, best = 0, 0
        for b in bad:
            run = run + 1 if b else 0
            best = max(best, run)
        # sustained = a share of this unit's own (coach) duration, so a quick movement and a long hold are
        # judged alike; the floor keeps a frame or two of jitter from becoming a label
        unit_s = float(C.t[u["_c"][1]] - C.t[u["_c"][0]])
        if best * float(np.median(C.dt)) >= max(cfg.min_label_frac * unit_s, cfg.min_label_floor_s):
            diffs = [np.degrees(np.arccos(np.clip((C.b[i] * S.b[pairs[i][0]]).sum(-1), -1, 1)))
                     for i, b in zip(idx, bad) if b and i in pairs]
            mdiff = np.mean(diffs, axis=0)
            order = np.argsort(-mdiff)[:2]
            u["pose"] = dict(worst=[dict(bone=C.names[k], diff_deg=round(float(mdiff[k]), 1)) for k in order],
                             mean_diff_deg=round(float(np.nanmean(dm[bad])), 1))
    # ---- what moves in structural units (data for the LLM, not sentences)
    for u in units:
        if u["structure"] in ("substitution", "missing") and u["_c"]:
            u["coach_motion"] = motion_summary(C, *u["coach_t"], ref_t=u["coach_t"][0])
        if u["structure"] in ("substitution", "extra") and u["_s"]:
            u["student_motion"] = motion_summary(S, *u["student_t"], ref_t=u["student_t"][0])
    for u in units:
        for k in ("_c", "_s", "_wc", "_ws", "_tc", "_ts"):
            u.pop(k, None)
    return units, r_global
