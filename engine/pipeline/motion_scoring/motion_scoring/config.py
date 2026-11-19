"""All tunable parameters of the scoring pipeline (units in the comments)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- preprocessing
    sg_window_s: float = 0.30        # Savitzky-Golay window (seconds -> fps-invariant)
    sg_order: int = 3

    # --- speed. v = speed / reference speed, where the speed has the measurement noise removed:
    #     noise makes a motionless body look as if it moved (estimated from the smoothing residual)
    v_ref_pct: float = 97.0          # reference speed = this percentile of the speed (a near-peak speed)
    noise_gain: float = 1.0          # v_low is never below noise_gain x the noise speed (v_exit, v_peak follow
                                     # in proportion); both clips of a comparison use the higher of their two
                                     # levels. 0 = ignore the noise
    arc_floor_pct: float = 95.0      # jitter floor = this percentile of the per-frame step while still

    # --- checkpoints: stillness (low-speed runs with hysteresis)
    v_low: float = 0.15              # enter low-speed state (still)
    v_exit: float = 0.25             # leave low-speed state (moving) ...
    exit_hold_s: float = 0.10        # ... only if v stays above v_exit this long (debounce)
    merge_gap_s: float = 0.15        # merge low-speed runs separated by a shorter gap
    edge_s: float = 0.10             # run within this of the clip edge is attached to the edge (rest)
    pause_min_s: float = 0.35        # low-speed run at least this long -> pause
    dir_window_s: float = 0.15       # window for entry / exit motion direction

    # --- checkpoints: reversals (speed dips at which one movement ends and another begins)
    v_peak: float = 0.5              # a speed peak of a real movement is at least this
    dip_ratio: float = 0.7           # the speed dips below dip_ratio x the lower of its two neighbouring peaks
                                     # (real squat turns: 0.15-0.4; a loose bound, the direction decides) ...
    dip_cos: float = -0.3            # ... and the pose goes back where it came from: cos(change before, change
                                     # after) below this (measured: turns <= -0.45, everything else >= -0.2)
    carry_cos: float = 0.5           # a short stop (v under v_low) is a checkpoint too unless the same movement
                                     # carries on after it (cos above this: a hesitation)

    # --- frame cost d (deg) = weighted mean bone-angle difference + direction term
    beta: float = 1.0                # direction term weight, in units of tau
    dir_gate: float = 0.2            # direction term only where both v > dir_gate

    # --- thresholds (deg). Defaults come from a synthetic calibration; for real video the CLI sets
    #     auto_tau=True (tau, kappa estimated from the coach's own repetitions) and pose_trim=0.3.
    tau_deg: float = 4.05            # "same pose" tolerance = p95 of coach-vs-coach matched d
    kappa: float = 5.8               # drop threshold = kappa * tau
    auto_tau: bool = False           # estimate tau, kappa from the coach's own repetitions (pipeline.estimate_tau):
    tau_gain: float = 1.3            #   tau = tau_gain x median distance between two visits to the same place ...
    tau_range: tuple = (3.0, 15.0)   #   ... kept inside this range (deg)
    same_place: float = 0.5          #   same place = closer than same_place x one movement (the median distance
                                     #   between consecutive checkpoints)
    kappa_min: float = 4.0           #   kappa = one movement / tau, never below this
    pose_trim: float = 0.0           # alignment ignores the most different fraction of bones (0.3 for real
                                     # video: a different arm position alone is not a different action)

    # --- checkpoint alignment (Needleman-Wunsch)
    sigma: float = 3.0               # substitution cost  (sigma < 2*gap -> substitute beats miss+extra)
    gap: float = 2.0                 # skip cost (missing / extra event)
    margin_min: float = 0.5          # matched pair becomes an anchor only if margin > margin_min
    w_pos: float = 2.0               # position weight: cost of being as many visits apart as the coach has
    w_ctx: float = 1.0               # context weight: movement into / out of the checkpoint

    # --- judgement
    rate_ok: tuple = (0.8, 1.25)     # (student/coach)/global_tempo outside -> fast / slow
    tempo_min_units: int = 4         # the global tempo is the median over the timed units if there are at least
                                     # this many; fewer do not define a tempo and it is taken as 1 (durations
                                     # are compared with the coach's as they are)
    pose_trust: float = 2.0          # boundary rest pair differing > pose_trust*tau -> clip was cut
    min_label_frac: float = 0.20     # pose flagged only if d > tau lasts this share of the unit (coach time) ...
    min_label_floor_s: float = 0.10  # ... and at least this long (a frame or two of jitter is never a label)
    dwell_tol_s: float = 0.15        # timing differences smaller than this (s) are not judged
    hold_missing_ratio: float = 0.5  # coach pause where the student only slows down briefly: missing only if
                                     # student slow-down < this x (coach pause x global tempo); else a (short) hold
    low_conf: float = 0.30           # frames below this mean joint confidence are interpolated
