"""All tunable parameters of the scoring pipeline (units in the comments)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- preprocessing
    sg_window_s: float = 0.30        # Savitzky-Golay window (seconds -> fps-invariant)
    sg_order: int = 3

    # --- event detection (normalised speed v = speed / percentile(speed, v_ref_pct))
    v_ref_pct: float = 97.0
    arc_floor_pct: float = 95.0      # jitter floor = this percentile of the per-frame step while still
    v_low: float = 0.15              # enter low-speed state
    v_exit: float = 0.25             # leave low-speed state (hysteresis) ...
    exit_hold_s: float = 0.10        # ... only if v stays above v_exit this long (debounce)
    pause_min_s: float = 0.35        # low-speed run at least this long -> pause
    edge_s: float = 0.10             # run within this of the clip edge is attached to the edge (rest)
    merge_gap_s: float = 0.15        # merge low-speed runs separated by a shorter gap
    dir_window_s: float = 0.15       # window for entry / exit motion direction

    # --- frame cost d (deg) = weighted mean bone-angle difference + direction term
    beta: float = 1.0                # direction term weight, in units of tau
    dir_gate: float = 0.2            # direction term only where both v > dir_gate

    # --- thresholds (deg). Defaults come from a synthetic calibration; for real video the CLI sets
    #     auto_tau=True (tau, kappa estimated from the coach's own repetitions) and pose_trim=0.3.
    tau_deg: float = 4.05            # "same pose" tolerance = p95 of coach-vs-coach matched d
    kappa: float = 5.8               # drop threshold = kappa * tau
    auto_tau: bool = False           # estimate tau from the coach's own repetitions (recommended for real video)
    pose_trim: float = 0.0           # alignment ignores the most different fraction of bones (0.3 for real
                                     # video: a different arm position alone is not a different action)

    # --- checkpoint alignment (Needleman-Wunsch)
    sigma: float = 3.0               # substitution cost  (sigma < 2*gap -> substitute beats miss+extra)
    gap: float = 2.0                 # skip cost (missing / extra event)
    margin_min: float = 0.5          # matched pair becomes an anchor only if margin > margin_min
    w_pos: float = 2.0               # progress-difference weight
    w_ctx: float = 1.0               # context weight: movement into / out of the checkpoint

    # --- judgement
    rate_ok: tuple = (0.8, 1.25)     # (student/coach)/global_tempo outside -> fast / slow
    pose_trust: float = 2.0          # boundary rest pair differing > pose_trust*tau -> clip was cut
    min_label_frac: float = 0.20     # pose flagged only if d > tau lasts this share of the unit (coach time) ...
    min_label_floor_s: float = 0.10  # ... and at least this long (a frame or two of jitter is never a label)
    dwell_tol_s: float = 0.15        # timing differences smaller than this (s) are not judged
    hold_missing_ratio: float = 0.5  # coach pause where the student only slows down briefly: missing only if
                                     # student slow-down < this x (coach pause x global tempo); else a (short) hold
    low_conf: float = 0.30           # frames below this mean joint confidence are interpolated

