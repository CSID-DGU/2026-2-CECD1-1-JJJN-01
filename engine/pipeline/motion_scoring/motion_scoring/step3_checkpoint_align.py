"""[3] Checkpoint alignment: two-pass Needleman-Wunsch (match / substitution / missing / extra) + margins."""
from __future__ import annotations

import numpy as np

from .config import Config
from .step1_preprocess import pose_dist


# ---------------------------------------------------------------- checkpoints
def event_cost(a, b, w, tau, kappa, w_ctx=1.0, trim=0.0):
    """Dissimilarity of two events in what they are (content): type + pose + entry/exit direction + context
    (movement from the previous / to the next event). Where they are is a separate cost (visits, align_events)."""
    if a.type == b.type:
        typ = 0.0
    elif {a.type, b.type} == {"pause", "reversal"}:
        typ = 0.25          # holding vs not holding is judged later as timing / a missing hold
    else:
        typ = 1.0

    def dis(u, v):          # 1 - cos, but no evidence (0) when either side has no direction / context
        if np.linalg.norm(u) < 1e-6 or np.linalg.norm(v) < 1e-6:
            return 0.0
        return 1 - float(u @ v)
    return (typ + 2.0 * pose_dist(a.pose, b.pose, w, trim=trim) / (kappa * tau)
            + 0.5 * dis(a.vin, b.vin) + 0.5 * dis(a.vout, b.vout)
            + w_ctx * (dis(a.ctx_in, b.ctx_in) + dis(a.ctx_out, b.ctx_out)) / 2)


def visits(events, w, same_place):
    """For every checkpoint of a clip: how many earlier checkpoints were at the same place (0 = the first time
    there). Same place: closer than same_place x one movement of the clip (the largest distance between two
    checkpoints in a row)."""
    if len(events) < 2:
        return [0] * len(events)
    size = max(pose_dist(a.pose, b.pose, w) for a, b in zip(events, events[1:]))
    places, out = [], []                               # a place: [pose of the first checkpoint there, visits so far]
    for e in events:
        k = next((k for k, (pose, _) in enumerate(places) if pose_dist(e.pose, pose, w) < same_place * size), None)
        if k is None:
            places.append([e.pose, 0])
            k = len(places) - 1
        out.append(places[k][1])
        places[k][1] += 1
    return out


def align_events(eC, eS, w, cfg: Config, tau, kappa):
    """Two passes -> total cost, [dict(kind, i, j, cost, margin)].

    A pair costs content (event_cost) + position. Position = w_pos x how many visits apart the two are (the
    coach's first time at the bottom belongs with the student's first, the second with the second), as a share
    of the coach's visits. So checkpoints are matched in order from the start, and what one clip does more
    often is left over at its end (missing / extra); a clip that is merely longer does not shift every pair,
    as comparing the share of the clip that has passed did (one repetition against three: the middle one).
    Never capped: a substitution must be at the same place.

    1) order-preserving DP with matches (content < sigma) and skips only -> confident correspondences
    2) inside each gap between consecutive matches, the remaining coach and student events are paired in
       order as substitutions (same place, different action); leftovers are missing / extra.
    A substitution can therefore never be "borrowed" from another part of the clip.
    margin (matches) = cost increase of pass 1 when that pair is forbidden."""
    vc, vs = visits(eC, w, cfg.same_place), visits(eS, w, cfg.same_place)
    per_visit = cfg.w_pos / max(max(vc, default=0), 1)
    cost = [[(event_cost(a, b, w, tau, kappa, cfg.w_ctx, cfg.pose_trim), per_visit * abs(p - q))
             for b, q in zip(eS, vs)] for a, p in zip(eC, vc)]
    total, ops = _nw_match_only(eC, eS, cost, cfg.sigma, cfg.gap)
    matches = [(i, j) for k, i, j, _ in ops if k == "match"]
    res = []
    bounds = [(-1, -1)] + matches + [(len(eC), len(eS))]
    for (i0, j0), (i1, j1) in zip(bounds, bounds[1:]):
        ci, sj = list(range(i0 + 1, i1)), list(range(j0 + 1, j1))
        for k in range(max(len(ci), len(sj))):
            if k < len(ci) and k < len(sj):
                c, pos = cost[ci[k]][sj[k]]
                res.append(dict(kind="substitute", i=ci[k], j=sj[k], cost=float(c + pos), margin=None))
            elif k < len(ci):
                res.append(dict(kind="missing", i=ci[k], j=None, cost=cfg.gap, margin=None))
            else:
                res.append(dict(kind="extra", i=None, j=sj[k], cost=cfg.gap, margin=None))
        if i1 < len(eC):
            c, pos = cost[i1][j1]
            alt, _ = _nw_match_only(eC, eS, cost, cfg.sigma, cfg.gap, forbid=(i1, j1))
            res.append(dict(kind="match", i=i1, j=j1, cost=float(c + pos), margin=alt - total))
    return total, res


def _nw_match_only(eC, eS, cost, sigma, gap, forbid=None):
    """DP with matches (content < sigma, cost content + position) and skips (gap)."""
    n, m = len(eC), len(eS)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    bp = {}
    for i in range(n + 1):
        for j in range(m + 1):
            cand = []
            if i and j and forbid != (i - 1, j - 1):
                c, pos = cost[i - 1][j - 1]
                if c < sigma:
                    cand.append((D[i - 1, j - 1] + c + pos, ("match", i - 1, j - 1, c + pos)))
            if i:
                cand.append((D[i - 1, j] + gap, ("missing", i - 1, None, gap)))
            if j:
                cand.append((D[i, j - 1] + gap, ("extra", None, j - 1, gap)))
            if cand:
                v, op = min(cand, key=lambda x: x[0])
                if v < D[i, j]:
                    D[i, j], bp[(i, j)] = v, op
    ops, i, j = [], n, m
    while (i, j) != (0, 0):
        op = bp[(i, j)]
        ops.append(op)
        if op[0] == "match":
            i, j = i - 1, j - 1
        elif op[0] == "missing":
            i -= 1
        else:
            j -= 1
    return float(D[n, m]), ops[::-1]
