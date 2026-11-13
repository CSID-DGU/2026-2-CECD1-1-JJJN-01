"""Pipeline: runs steps [1]-[5] in order.

  input   coach skeleton (retargeted to the student body) + student skeleton
  [1] step1_preprocess        bone directions, SG filter, speed / acceleration
  [2] step2_checkpoints       checkpoints from speed + acceleration
  [3] step3_checkpoint_align  Needleman-Wunsch -> match / substitution / missing / extra, anchors
  [4] step4_frame_align       Drop-DTW between anchors
  [5] step5_segments          segments + structure / timing / pose labels  (Result.units / to_dict)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .step3_checkpoint_align import align_events
from .step4_frame_align import drop_dtw, frame_cost
from .config import Config
from .step2_checkpoints import detect_events
from .step1_preprocess import Clip, Track, pose_dist


@dataclass
class Result:
    events: dict
    checkpoints: list
    coverage: float
    mirrored: bool
    params: dict
    # internals for plotting / debugging
    C: Clip = field(repr=False, default=None)
    S: Clip = field(repr=False, default=None)
    cfg: Config = field(repr=False, default=None)
    debug: dict = field(repr=False, default_factory=dict)

    def units(self, cfg=None):
        """non-overlapping segments on the student timeline: structure + timing + pose (step5_segments.build_units)."""
        from .config import Config
        from .step5_segments import build_units
        return build_units(self, cfg or self.cfg or Config())   # [5] segments + labels

    def to_dict(self):
        units, tempo = self.units()
        return dict(segments=units, global_tempo=round(tempo, 3), coverage=round(self.coverage, 3),
                    mirrored=self.mirrored, events=self.events, checkpoints=self.checkpoints,
                    params=self.params)


def estimate_tau(C: Clip, cfg: Config):
    """tau from the coach clip itself: the coach repeats checkpoints, and same-type checkpoints of the
    repetitions differ only by natural variation + pose-estimation noise. tau = 1.3 x median of the
    distances between consecutive same-type checkpoints (clipped to 3..15 deg); (None, None) if too few."""
    ev = [e for e in detect_events(C, cfg) if e.type != "rest"]
    ds = []
    for typ in ("pause", "reversal"):
        same = [e for e in ev if e.type == typ]
        ds += [pose_dist(a.pose, b.pose, C.w, trim=cfg.pose_trim) for a, b in zip(same, same[1:])]
    ds = [x for x in ds if x < 3 * np.median(ds)] if ds else ds      # drop pairs that are not the same pose
    if len(ds) < 3:
        return None, None
    tau = float(np.clip(1.3 * np.median(ds), 3.0, 15.0))
    # "a different action" scale = the coach's own movement range: the median distance between consecutive
    # *different* coach checkpoints (e.g. standing vs squat bottom). Differing by less than the coach moves
    # within one action is a pose difference, not another action. kappa = that / tau, never below 4
    diff = [pose_dist(a.pose, b.pose, C.w, trim=cfg.pose_trim) for a, b in zip(ev, ev[1:]) if a.type != b.type]
    kappa = max(4.0, float(np.median(diff)) / tau) if diff else None
    return tau, kappa


def score(coach: Track, student: Track, cfg: Config = Config(), bone_weights=None, try_mirror=True) -> Result:
    """Score the student track against the coach track (coach retargeted to the student body)."""
    C = Clip(coach, cfg, bone_weights)                    # [1] preprocess
    if cfg.auto_tau:
        tau, kappa = estimate_tau(C, cfg)
        if tau is not None:
            from dataclasses import replace
            cfg = replace(cfg, tau_deg=tau, kappa=kappa or cfg.kappa)
    res = _evaluate(C, Clip(student, cfg, bone_weights), cfg, mirrored=False)
    if try_mirror and any(n.startswith("l_") for n in student.bone_names):
        alt = _evaluate(C, Clip(student.mirrored(), cfg, bone_weights), cfg, mirrored=True)
        if alt.debug["path_cost"] < 0.8 * res.debug["path_cost"]:   # clearly better only
            res = alt
    return res


# ---------------------------------------------------------------- core
def _evaluate(C: Clip, S: Clip, cfg: Config, mirrored: bool) -> Result:
    tau, kappa = cfg.tau_deg, cfg.kappa
    nC, nS = len(C), len(S)
    eC, eS = detect_events(C, cfg), detect_events(S, cfg)          # [2] checkpoints
    _, ops = align_events(eC, eS, C.w, cfg, tau, kappa)            # [3] checkpoint alignment

    # a rest<->rest match at a clip boundary whose poses differ means the clip was cut mid-motion:
    # do not anchor on it; the coach checkpoint is reported as withheld (clip_edge)
    fixed = []
    for o in ops:
        if (o["kind"] == "match" and eC[o["i"]].type == "rest" and eS[o["j"]].type == "rest"
                and pose_dist(eC[o["i"]].pose, eS[o["j"]].pose, C.w, trim=cfg.pose_trim) > cfg.pose_trust * tau):
            fixed.append(dict(kind="missing", i=o["i"], j=None, cost=cfg.gap, margin=None, edge_cut=True))
            fixed.append(dict(kind="extra", i=None, j=o["j"], cost=cfg.gap, margin=None))
        else:
            fixed.append(o)
    ops = fixed                                       # order stays the alignment order
    # anchors: confident matches, endpoints guaranteed
    anchors = []
    for o in ops:
        boundary_rest = (o["kind"] == "match" and eC[o["i"]].type == "rest" and eS[o["j"]].type == "rest"
                         and o["i"] in (0, len(eC) - 1) and o["j"] in (0, len(eS) - 1))
        if o["kind"] == "match" and (boundary_rest or (o["margin"] is not None and o["margin"] > cfg.margin_min)):
            ci, si = eC[o["i"]].i, eS[o["j"]].i
            if not anchors or (ci > anchors[-1][0] and si > anchors[-1][1]):
                anchors.append((ci, si, o["i"], o["j"]))
    if len(anchors) < 2:                                   # no usable checkpoints -> whole clip
        anchors = [(0, 0, None, None), (nC - 1, nS - 1, None, None)]

    cost, d = frame_cost(C, S, cfg, tau)                           # [4] frame alignment
    hold_c, hold_s = ~C.conf_ok, ~S.conf_ok
    lam = kappa * tau
    path, intervals = [], []
    for a, b in zip(anchors, anchors[1:]):
        ci0, si0, ci1, si1 = a[0], a[1], b[0], b[1]
        st = drop_dtw(cost[ci0:ci1 + 1, si0:si1 + 1], C.dt[ci0:ci1 + 1], S.dt[si0:si1 + 1], lam)
        st = [(k, i + ci0, j + si0) for k, i, j in st]
        path += st if not path else st[1:]
        intervals.append((ci0, ci1, si0, si1, a[2], b[2]))
    ca, cb, sa, sb = anchors[0][0], anchors[-1][0], anchors[0][1], anchors[-1][1]

    act = np.zeros(nC, bool); act[ca:cb + 1] = True

    # coverage: share of the coach span not withheld for low confidence (own frame or its student partner)
    act = np.zeros(nC, bool); act[anchors[0][0]:anchors[-1][0] + 1] = True
    c_hold = act & hold_c
    for k, i, j in path:
        if k not in ("drop_c", "drop_s") and hold_s[j]:
            c_hold[i] = act[i]
    coverage = 1 - float(C.dt[c_hold].sum()) / max(float(C.dt[act].sum()), 1e-9)

    path_cost = float(np.mean([d[i, j] for kk, i, j in path if kk not in ("drop_c", "drop_s")] or [np.inf]))
    return Result(
        events=dict(coach=[e.as_dict() for e in eC], student=[e.as_dict() for e in eS]),
        checkpoints=[dict(kind=o["kind"], coach=o["i"], student=o["j"], cost=round(o["cost"], 3),
                          margin=None if o["margin"] is None else round(o["margin"], 3)) for o in ops],
        coverage=coverage, mirrored=mirrored,
        params=dict(tau_deg=tau, kappa=kappa, sigma=cfg.sigma, gap=cfg.gap),
        C=C, S=S, cfg=cfg,
        debug=dict(eC=eC, eS=eS, ops=ops, anchors=anchors, path=path, d=d, path_cost=path_cost),
    )
