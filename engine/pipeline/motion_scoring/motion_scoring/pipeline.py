"""Pipeline: runs steps [1]-[5] in order.

  input   coach skeleton (retargeted to the student body) + student skeleton
  [1] step1_preprocess        bone directions, SG filter, speed / acceleration
  [2] step2_checkpoints       checkpoints from speed + acceleration
  [3] step3_checkpoint_align  Needleman-Wunsch -> match / substitution / missing / extra, anchors
  [4] step4_frame_align       Drop-DTW between anchors
  [5] step5_segments          segments + structure / timing / pose labels  (Result.units / to_dict)
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .step3_checkpoint_align import align_events
from .step4_frame_align import DROPS, drop_dtw, frame_cost
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
        from .step5_segments import build_units
        return build_units(self, cfg or self.cfg or Config())   # [5] segments + labels

    def to_dict(self):
        units, tempo = self.units()
        return dict(segments=units, global_tempo=round(tempo, 3), coverage=round(self.coverage, 3),
                    mirrored=self.mirrored, events=self.events, checkpoints=self.checkpoints,
                    params=self.params)


def estimate_tau(C: Clip, cfg: Config):
    """tau and kappa from the coach clip itself -> (tau, kappa), or (None, None) if it repeats too little.

    A coach who repeats a movement passes through the same places again and again (the top and the bottom
    of every repetition). Two visits to one place differ only by natural variation + pose-estimation noise,
    which is what "the same pose" has to tolerate: tau = tau_gain x the median distance between a checkpoint
    and the next visit to its place, kept inside tau_range.
    Which checkpoints are the same place is read from the poses: one movement = the median distance between
    consecutive checkpoints, same place = closer than same_place x that. (Not from the checkpoint types: a
    coach who pauses at the bottom as well as at the top has pauses only, and consecutive pauses are top
    and bottom.) "A different action" is that movement scale: kappa = one movement / tau, at least kappa_min.
    Needs three visits, i.e. two repetitions between two rests; one repetition keeps the Config values."""
    ev = detect_events(C, cfg)
    dist = lambda a, b: float(pose_dist(a.pose, b.pose, C.w, trim=cfg.pose_trim))
    step = [dist(a, b) for a, b in zip(ev, ev[1:])]
    if not step:
        return None, None
    move = float(np.median(step))
    again = []
    for i, e in enumerate(ev):
        d = next((x for x in (dist(e, f) for f in ev[i + 1:]) if x < cfg.same_place * move), None)
        if d is not None:
            again.append(d)
    if len(again) < 3:
        return None, None
    tau = float(np.clip(cfg.tau_gain * np.median(again), *cfg.tau_range))
    kappa = max(cfg.kappa_min, float(np.median([d for d in step if d >= cfg.same_place * move])) / tau)
    return tau, kappa


def score(coach: Track, student: Track, cfg: Config = Config(), bone_weights=None, try_mirror=True) -> Result:
    """Score the student track against the coach track (coach retargeted to the student body)."""
    C, S = Clip(coach, cfg, bone_weights), Clip(student, cfg, bone_weights)     # [1] preprocess
    # the same "still" for both: the noisier clip sets the noise speed the levels keep clear of (a hold only
    # lasts equally long in the two clips if both are measured against the same level)
    noise = max(C.noise, S.noise)
    if C.noise < noise:
        C = Clip(coach, cfg, bone_weights, noise_floor=noise)
    if S.noise < noise:
        S = Clip(student, cfg, bone_weights, noise_floor=noise)
    tau_from = "config"
    if cfg.auto_tau:
        tau, kappa = estimate_tau(C, cfg)
        if tau is not None:
            cfg, tau_from = replace(cfg, tau_deg=tau, kappa=kappa), "coach"
    res = _evaluate(C, S, cfg, mirrored=False)
    if try_mirror and any(n.startswith("l_") for n in student.bone_names):
        alt = _evaluate(C, Clip(student.mirrored(), cfg, bone_weights, noise_floor=noise), cfg, mirrored=True)
        if alt.debug["path_cost"] < 0.8 * res.debug["path_cost"]:   # clearly better only
            res = alt
    res.params["tau_from"] = tau_from          # "coach": estimated from its repetitions; "config": too few
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
    # anchors: the confident matches, in order ...
    core = []
    for o in ops:
        boundary_rest = (o["kind"] == "match" and eC[o["i"]].type == "rest" and eS[o["j"]].type == "rest"
                         and o["i"] in (0, len(eC) - 1) and o["j"] in (0, len(eS) - 1))
        if o["kind"] == "match" and (boundary_rest or (o["margin"] is not None and o["margin"] > cfg.margin_min)):
            ci, si = eC[o["i"]].i, eS[o["j"]].i
            if not core or (ci > core[-1][0] and si > core[-1][1]):
                core.append((ci, si, o["i"], o["j"]))
    # ... between the two clip corners, which are always anchors: the stretch before the first and after
    # the last confident checkpoint is aligned too (the first / last movement of a clip whose edge is not
    # a matched rest, or the whole clip if no checkpoint is matched)
    anchors = list(core)
    if not anchors or anchors[0][:2] != (0, 0):
        anchors.insert(0, (0, 0, None, None))
    if anchors[-1][:2] != (nC - 1, nS - 1):
        anchors.append((nC - 1, nS - 1, None, None))

    cost, d = frame_cost(C, S, cfg, tau)                           # [4] frame alignment
    lam = kappa * tau
    path = []
    for a, b in zip(anchors, anchors[1:]):
        ci0, si0, ci1, si1 = a[0], a[1], b[0], b[1]
        st = drop_dtw(cost[ci0:ci1 + 1, si0:si1 + 1], C.dt[ci0:ci1 + 1], S.dt[si0:si1 + 1], lam)
        st = [(k, i + ci0, j + si0) for k, i, j in st]
        path += st if not path else st[1:]

    # coverage: share of the coach clip not withheld for low confidence (own frame or its student partner)
    withheld = ~C.conf_ok
    for k, i, j in path:
        if k not in DROPS and not S.conf_ok[j]:
            withheld[i] = True
    coverage = 1 - float(C.dt[withheld].sum()) / max(float(C.dt.sum()), 1e-9)

    # how well the poses agree where the movement is (between the first and the last confident checkpoint;
    # the whole clip without them): decides whether the mirrored student fits better
    lo, hi = (core[0][0], core[-1][0]) if len(core) >= 2 else (0, nC - 1)
    path_cost = float(np.mean([d[i, j] for k, i, j in path if k not in DROPS and lo <= i <= hi] or [np.inf]))
    return Result(
        events=dict(coach=[e.as_dict() for e in eC], student=[e.as_dict() for e in eS]),
        checkpoints=[dict(kind=o["kind"], coach=o["i"], student=o["j"], cost=round(o["cost"], 3),
                          margin=None if o["margin"] is None else round(o["margin"], 3)) for o in ops],
        coverage=coverage, mirrored=mirrored,
        params=dict(tau_deg=tau, kappa=kappa, sigma=cfg.sigma, gap=cfg.gap),
        C=C, S=S, cfg=cfg,
        debug=dict(eC=eC, eS=eS, ops=ops, anchors=anchors, path=path, d=d, path_cost=path_cost),
    )
