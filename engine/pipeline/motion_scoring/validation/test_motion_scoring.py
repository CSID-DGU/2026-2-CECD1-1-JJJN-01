"""Regression tests of the scoring output (what the backend reads: Result.to_dict() and Result.debug["path"]).

Synthetic squats (synthetic.py, next to this file), so no video, data file or GPU is needed, only numpy:

    python validation/test_motion_scoring.py     # from engine/pipeline/motion_scoring; or: pytest validation/

Every test compares a coach clip with a user clip that differs in ONE known way and checks that exactly
that is reported. Both clips carry a little joint noise with different seeds, as two videos would.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]               # synthetic.py (this folder), the motion_scoring package
import synthetic as syn  # noqa: E402
from motion_scoring import Config, score, track_from_joints  # noqa: E402
from motion_scoring.step1_preprocess import Clip  # noqa: E402
from motion_scoring.step2_checkpoints import detect_events  # noqa: E402

CFG = replace(Config(), auto_tau=True, pose_trim=0.3)      # the preset the backend uses (result_builder.py)
NOISE_CM = 0.2
STRUCTURES = {"match", "substitution", "missing", "extra"}


def track(ph, seed=1, fps=30.0, noise=NOISE_CM, cut=(0.0, 0.0), mirror=False, **body):
    """Phases -> the scorer's input. cut = seconds removed from the start / end of the clip."""
    joints, fps = syn.skeleton(ph, fps=fps, noise_cm=noise, seed=seed, **body)
    joints = joints[int(round(cut[0] * fps)):len(joints) - int(round(cut[1] * fps))]
    if mirror:
        joints = syn.mirrored(joints)
    return track_from_joints(joints, np.arange(len(joints)) / fps, "y")


def compare(coach_ph, user_ph, coach=None, user=None, cfg=CFG):
    """-> (Result, to_dict()) for a coach and a user given as phases (+ keyword arguments of track)."""
    res = score(track(coach_ph, seed=1, **(coach or {})), track(user_ph, seed=2, **(user or {})), cfg)
    return res, res.to_dict()


def structures(d):
    return [s["structure"] for s in d["segments"]]


def timing(d):
    return [(s["timing"] or {}).get("label") for s in d["segments"]]


def length(s, key="student_t"):
    return s[key][1] - s[key][0]


def checkpoints(tr):
    """Types of the checkpoints of one clip: r = rest, p = pause, v = reversal."""
    return "".join("v" if e.type == "reversal" else e.type[0] for e in detect_events(Clip(tr, CFG), CFG))


def nth(ph, k, depth_at_end):
    """Index in ph of the k-th phase (0-based) that ends at that depth and is a movement."""
    hits = [i for i, (_, q) in enumerate(ph) if q == depth_at_end and (ph[i - 1][1] if i else 0.0) != q]
    return hits[k]


# ---------------------------------------------------------------- the same motion
def test_same_motion_has_no_labels():
    ph = syn.phases(reps=3)
    _, d = compare(ph, ph)
    assert structures(d) == ["match"] * 8, structures(d)          # 3 x (down, up) + 2 standing holds
    assert not any(timing(d)) and not any(s["pose"] for s in d["segments"]), d["segments"]
    assert abs(d["global_tempo"] - 1.0) < 0.05 and d["coverage"] == 1.0 and d["mirrored"] is False


def test_same_motion_slower_or_faster_has_no_labels():
    """the user's own tempo is not a timing fault: every phase 1.3x longer"""
    ph = syn.phases(reps=4)
    for factor in (0.8, 1.3):
        _, d = compare(ph, [(s * factor, q) for s, q in ph])
        assert set(structures(d)) == {"match"} and not any(timing(d)), (factor, d["segments"])
        assert abs(d["global_tempo"] - factor) < 0.1, (factor, d["global_tempo"])


def test_noisier_video_keeps_the_structure():
    """0.5 cm of joint noise on the user only: nothing missing, nothing extra"""
    for kind in (dict(), dict(bottom=0.7, top=1.0)):
        ph = syn.phases(reps=3, **kind)
        _, d = compare(ph, ph, user=dict(noise=0.5))
        assert set(structures(d)) == {"match"}, (kind, structures(d))


# ---------------------------------------------------------------- checkpoints
def test_every_turn_and_pause_is_a_checkpoint():
    """one checkpoint at every bottom and every top, whatever the tempo, frame rate or noise"""
    styles = {                                         # phases arguments -> checkpoint types for 3 repetitions
        "turns at the bottom": (dict(), "rvpvpvr"),
        "pauses at the bottom": (dict(bottom=0.7, top=1.0), "rpppppr"),
        "no pause at all": (dict(top=0.0), "rvvvvvr"),
    }
    for name, (kind, expected) in styles.items():
        for tempo in (0.7, 1.0, 1.4):
            ph = [(s * tempo, q) for s, q in syn.phases(reps=3, **kind)]
            for fps, noise in ((30.0, NOISE_CM), (24.0, NOISE_CM), (60.0, NOISE_CM), (30.0, 0.5)):
                got = checkpoints(track(ph, fps=fps, noise=noise))
                assert got == expected, f"{name}, tempo x{tempo}, {fps:.0f} fps, noise {noise} cm: {got} != {expected}"


def test_hesitation_is_not_a_checkpoint():
    """slowing down half-way and carrying on in the same direction"""
    ph = syn.phases(reps=2)
    i = nth(ph, 0, 1.0)
    ph[i:i + 1] = [(0.6, 0.5), (0.6, 1.0)]             # the first descent in two halves
    assert checkpoints(track(ph)) == "rvpvr"


# ---------------------------------------------------------------- tau
def test_tau_comes_from_two_visits_to_the_same_place():
    """also for a coach who pauses at the bottom: only pauses, alternately top and bottom"""
    for kind in (dict(), dict(bottom=0.7, top=1.0), dict(top=0.0)):
        ph = syn.phases(reps=3, **kind)
        _, d = compare(ph, ph)
        p = d["params"]
        assert p["tau_from"] == "coach" and 3.0 <= p["tau_deg"] <= 5.0, (kind, p)
        assert p["kappa"] >= 4.0, (kind, p)


def test_one_repetition_keeps_the_configured_tau():
    ph = syn.phases(reps=1)
    _, d = compare(ph, ph)
    assert d["params"]["tau_from"] == "config" and d["params"]["tau_deg"] == CFG.tau_deg, d["params"]


# ---------------------------------------------------------------- few repetitions, clip edges
def test_single_repetition_is_judged():
    ph = syn.phases(reps=1)
    _, d = compare(ph, ph)
    assert structures(d) == ["match", "match"], d["segments"]             # down, up
    assert not any(timing(d)) and not any(s["pose"] for s in d["segments"])
    _, d = compare(ph, syn.phases(reps=1, bottom=0.8))                    # the user holds the bottom
    assert structures(d) == ["match", "extra", "match"], d["segments"]
    assert abs(length(d["segments"][1]) - 0.8) < 0.3, d["segments"][1]


def test_single_repetition_timing_is_against_the_coach():
    """two movements do not define the user's tempo: twice as long going down is slow, going up is fine"""
    _, d = compare(syn.phases(reps=1), syn.phases(reps=1, down=2.0))
    assert structures(d) == ["match", "match"] and timing(d) == ["slow", None], d["segments"]
    assert abs(d["segments"][0]["timing"]["ratio"] - 2.0) < 0.4 and d["global_tempo"] == 1.0


def test_clip_that_starts_in_motion():
    """the user's video starts half-way down: that movement is compared, but not timed"""
    ph = syn.phases(reps=2)
    _, d = compare(ph, ph, user=dict(cut=(0.8 + 0.5, 0.0)))
    first = d["segments"][0]
    assert first["structure"] == "match" and first["edge"] == "start" and first["cut"] == ["student"], first
    assert first["timing"] is None
    assert set(structures(d)) == {"match"} and "edge" not in d["segments"][1], d["segments"]


def test_clip_without_any_rest():
    """both videos start and end in motion: still one unit per movement"""
    ph = syn.phases(reps=2)
    _, d = compare(ph, ph, coach=dict(cut=(1.1, 1.1)), user=dict(cut=(1.2, 1.2)))
    assert set(structures(d)) == {"match"} and len(d["segments"]) >= 4, d["segments"]
    assert d["segments"][0]["edge"] == "start" and d["segments"][-1]["edge"] == "end"


def test_user_who_does_not_move():
    """no checkpoint to match: the coach's movement is reported, not an empty result"""
    _, d = compare(syn.phases(reps=2), [(6.0, 0.0)])
    assert d["segments"] and set(structures(d)) == {"missing"}, d["segments"]


# ---------------------------------------------------------------- one difference at a time
def test_one_slow_descent():
    ph = syn.phases(reps=4)
    user = list(ph)
    i = nth(user, 1, 1.0)
    user[i] = (1.5, 1.0)                                                  # the second descent: 1.0 s -> 1.5 s
    start = sum(s for s, _ in user[:i])
    _, d = compare(ph, user)
    assert set(structures(d)) == {"match"}, structures(d)
    slow = [s for s in d["segments"] if s["timing"]]
    assert len(slow) == 1 and slow[0]["timing"]["label"] == "slow", [s["timing"] for s in d["segments"]]
    assert abs(slow[0]["student_t"][0] - start) < 0.4 and abs(slow[0]["timing"]["ratio"] - 1.5) < 0.25, slow[0]


def test_extra_hold_at_the_bottom():
    ph = syn.phases(reps=4)
    user = list(ph)
    i = nth(user, 1, 1.0)
    user.insert(i + 1, (1.0, 1.0))                                        # 1 s at the second bottom
    _, d = compare(ph, user)
    extra = [s for s in d["segments"] if s["structure"] == "extra"]
    assert len(extra) == 1 and abs(length(extra[0]) - 1.0) < 0.3, d["segments"]
    assert set(structures(d)) == {"match", "extra"} and extra[0]["coach_t"] is None


def test_missing_rest_at_the_top():
    ph = syn.phases(reps=4, top=0.8)
    user = list(ph)
    del user[nth(user, 1, 0.0) + 1]                                       # no standing rest after the second repetition
    _, d = compare(ph, user)
    missing = [s for s in d["segments"] if s["structure"] == "missing"]
    assert len(missing) == 1 and set(structures(d)) == {"match", "missing"}, d["segments"]
    assert abs(length(missing[0], "coach_t") - 0.8) < 0.3 and missing[0]["hold"]["ratio"] < 0.5, missing[0]


def test_extra_and_missing_repetition():
    three, four = syn.phases(reps=3), syn.phases(reps=4)
    _, d = compare(three, four)
    extra = [s for s in d["segments"] if s["structure"] == "extra"]
    assert len(extra) == 1 and set(structures(d)) == {"match", "extra"}, d["segments"]
    assert structures(d).count("match") == 8 and 1.6 < length(extra[0]) < 3.4, d["segments"]
    _, d = compare(four, three)
    missing = [s for s in d["segments"] if s["structure"] == "missing"]
    assert len(missing) == 1 and set(structures(d)) == {"match", "missing"}, d["segments"]
    assert structures(d).count("match") == 8 and 1.6 < length(missing[0], "coach_t") < 3.4, d["segments"]


def test_mirrored_user():
    ph = syn.phases(reps=3)
    _, d = compare(ph, ph, coach=dict(asym=15.0), user=dict(asym=15.0, mirror=True))
    assert d["mirrored"] is True and set(structures(d)) == {"match"}, (d["mirrored"], structures(d))
    assert not any(s["pose"] for s in d["segments"])
    _, d = compare(ph, ph, coach=dict(asym=15.0), user=dict(asym=15.0))
    assert d["mirrored"] is False


# ---------------------------------------------------------------- the contract with the backend
def test_output_format():
    res, d = compare(syn.phases(reps=3), syn.phases(reps=3, bottom=0.8, down=1.6))
    assert set(d) == {"segments", "global_tempo", "coverage", "mirrored", "events", "checkpoints", "params"}
    assert {"tau_deg", "kappa"} <= set(d["params"])
    for s in d["segments"]:
        assert s["structure"] in STRUCTURES
        assert len(s["student_t"]) == 2 and s["student_t"][0] <= s["student_t"][1]
        assert s["coach_t"] is None or s["coach_t"][0] <= s["coach_t"][1]
        assert s["timing"] is None or (s["timing"]["label"] in ("slow", "fast") and s["timing"]["ratio"] > 0)
        assert s["pose"] is None or all(set(w) == {"bone", "diff_deg"} for w in s["pose"]["worst"])
        assert not any(k.startswith("_") for k in s), s
    starts = [s["student_t"][0] for s in d["segments"]]
    assert starts == sorted(starts)
    assert all(e["type"] in ("rest", "pause", "reversal") for who in ("coach", "student") for e in d["events"][who])
    nC, nS = len(res.C), len(res.S)
    path = res.debug["path"]
    assert {k for k, _, _ in path} <= {"match", "warp_c", "warp_s", "drop_c", "drop_s"}
    assert path[0][1:] == (0, 0) and path[-1][1:] == (nC - 1, nS - 1)       # the whole of both clips is aligned
    assert all(0 <= b[1] - a[1] <= 1 and 0 <= b[2] - a[2] <= 1 for a, b in zip(path, path[1:]))


def main():
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except Exception as e:                           # noqa: BLE001 -- one broken test must not hide the others
            failed += 1
            print(f"FAIL  {name}\n      {type(e).__name__}: {str(e)[:600]}")
    print(f"{len(tests) - failed} / {len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
