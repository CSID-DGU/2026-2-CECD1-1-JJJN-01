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
from motion_scoring import Config, main_weights, score, track_from_joints  # noqa: E402
from motion_scoring.step1_preprocess import SMPL_NAMES, Clip  # noqa: E402
from motion_scoring.step2_checkpoints import detect_events  # noqa: E402

# what the backend scores a squat with (result_builder.scorer_setup): tau and kappa from the coach's clip, and
# the legs and the trunk counting more than the arms and the head when two poses are compared
CFG = replace(Config(), auto_tau=True)
SQUAT = ("trunk", "l_thigh", "r_thigh", "l_shank", "r_shank")
WEIGHTS = main_weights(SQUAT, CFG)
NOISE_CM = 0.2
STRUCTURES = {"match", "substitution", "missing", "extra"}


def track(ph, seed=1, fps=30.0, noise=NOISE_CM, cut=(0.0, 0.0), knees_cm=(0.0, 0.0), arms_out=0.0, mirror=False,
          **body):
    """Phases -> the scorer's input. cut = seconds removed from the start / end of the clip; knees_cm = the
    (left, right) knee moved that far towards the other one; arms_out = both arms turned that many degrees out
    to the sides at the shoulders, all the time; mirror = done like a mirror image, after that."""
    joints, fps = syn.skeleton(ph, fps=fps, noise_cm=noise, seed=seed, **body)
    joints = joints[int(round(cut[0] * fps)):len(joints) - int(round(cut[1] * fps))]
    for side, sign, cm in (("left", -1.0, knees_cm[0]), ("right", 1.0, knees_cm[1])):     # the left is +X
        joints[:, SMPL_NAMES.index(f"{side}_knee"), 0] += sign * cm
    for side, sign in (("left", 1.0), ("right", -1.0)) if arms_out else ():
        a = sign * np.radians(arms_out)                                    # about the forward axis, away from the body
        shoulder = joints[:, SMPL_NAMES.index(f"{side}_shoulder")]
        for k in (SMPL_NAMES.index(f"{side}_elbow"), SMPL_NAMES.index(f"{side}_wrist")):
            x, y, z = (joints[:, k] - shoulder).T
            joints[:, k] = shoulder + np.stack([x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a), z], 1)
    if mirror:
        joints = syn.mirrored(joints)
    return track_from_joints(joints, np.arange(len(joints)) / fps, "y")


def compare(coach_ph, user_ph, coach=None, user=None, cfg=CFG, weights=WEIGHTS):
    """-> (Result, to_dict()) for a coach and a user given as phases (+ keyword arguments of track)."""
    res = score(track(coach_ph, seed=1, **(coach or {})), track(user_ph, seed=2, **(user or {})), cfg, weights)
    return res, res.to_dict()


def structures(d):
    return [s["structure"] for s in d["segments"]]


def timing(d):
    return [(s["timing"] or {}).get("label") for s in d["segments"]]


def length(s, key="student_t"):
    return s[key][1] - s[key][0]


def checkpoints(tr):
    """Types of the checkpoints of one clip: r = rest, p = pause, v = reversal."""
    return "".join("v" if e.type == "reversal" else e.type[0] for e in detect_events(Clip(tr, CFG, WEIGHTS), CFG))


def nth(ph, k, depth_at_end):
    """Index in ph of the k-th phase (0-based) that ends at that depth and is a movement."""
    hits = [i for i, (_, q) in enumerate(ph) if q == depth_at_end and (ph[i - 1][1] if i else 0.0) != q]
    return hits[k]


def arm_exercise(ph, seed, part="upper"):
    """The body stands still and, with the timing of the phases ph, part = "upper": both upper arms swing out
    to the sides to 90 deg and back while the forearms keep pointing the way they did (two bones carry the
    movement, as in a bench press); part = "fore": the forearms swing forward to 90 deg and back instead, the
    upper arms stay (another movement with the same timing)."""
    stand, fps = syn.skeleton([(1.0, 0.0)])
    t = np.arange(int(round(sum(s for s, _ in ph) * fps)) + 1) / fps
    a = np.radians(90.0) * syn._depth(ph, t)
    joints = np.repeat(stand[:1], len(t), 0)
    for side, sign in (("left", 1.0), ("right", -1.0)):                    # the left is +X
        sh, el, wr = (stand[0, SMPL_NAMES.index(f"{side}_{n}")] for n in ("shoulder", "elbow", "wrist"))
        if part == "upper":
            elbow = sh + np.linalg.norm(el - sh) * np.stack([sign * np.sin(a), -np.cos(a), 0 * a], 1)
            wrist = elbow + (wr - el)
        else:
            elbow = np.repeat(el[None], len(t), 0)
            wrist = elbow + np.linalg.norm(wr - el) * np.stack([0 * a, -np.cos(a), np.sin(a)], 1)
        joints[:, SMPL_NAMES.index(f"{side}_elbow")], joints[:, SMPL_NAMES.index(f"{side}_wrist")] = elbow, wrist
    joints = joints + np.random.default_rng(seed).normal(0.0, NOISE_CM, joints.shape)
    return track_from_joints(joints, t, "y")


def bones_at(d, t, end=True):
    """bones of the match segment that ends (end=False: starts) nearest to the student time t -> {bone: entry}."""
    seg = min((s for s in d["segments"] if "bones" in s), key=lambda s: abs(s["student_t"][1 if end else 0] - t))
    return {b["bone"]: b for b in seg["bones"]}


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


def test_one_repetition_still_gives_the_size_of_the_movement():
    """one repetition cannot give tau, but it shows how large one movement is: kappa x tau comes out as with
    three repetitions, not as the configured value, which no exercise in particular has"""
    _, one = compare(syn.phases(reps=1), syn.phases(reps=1))
    _, three = compare(syn.phases(reps=3), syn.phases(reps=3))
    p1, p3 = one["params"], three["params"]
    assert (p1["tau_from"], p1["kappa_from"]) == ("config", "clip") and p3["kappa_from"] == "coach", (p1, p3)
    size1, size3 = p1["kappa"] * p1["tau_deg"], p3["kappa"] * p3["tau_deg"]
    assert abs(size1 / size3 - 1.0) < 0.15 and abs(size1 - CFG.kappa * CFG.tau_deg) > 10.0, (size1, size3)


# ---------------------------------------------------------------- which bones count
def test_main_weights():
    names = track(syn.phases(reps=1)).bone_names
    w = main_weights(SQUAT, CFG)
    assert [float(w[names.index(n)]) for n in SQUAT] == [1.0] * 5 and float(w.sum()) == 5 + 5 * CFG.other_bones, w
    assert main_weights(None) is None and main_weights(()) is None        # every bone alike
    try:
        main_weights(("torso",))
    except ValueError:
        pass
    else:
        raise AssertionError("a bone name the scorer does not have was accepted")


def test_arms_held_differently_are_not_another_action():
    """the user holds both arms 60 deg out to the sides: every unit is still a match, the checkpoints pair
    clearly more cheaply than with every bone counting alike (about half), and the bones say how the arms
    differ (they are reported whatever their weight)"""
    ph = syn.phases(reps=3)
    cost = lambda res: max(o["cost"] for o in res.debug["ops"] if o["kind"] == "match")
    res, d = compare(ph, ph, user=dict(arms_out=60.0))
    alike, _ = compare(ph, ph, user=dict(arms_out=60.0), weights=None)
    assert structures(d) == ["match"] * 8, structures(d)
    assert cost(res) < 0.7 * cost(alike) and cost(res) < CFG.sigma / 3, (cost(res), cost(alike))
    arm = bones_at(d, 0.8 + 2.0, end=False)["l_upper_arm"]                # standing after the first repetition
    assert 45.0 < arm["deg"] < 65.0 and arm["dir"]["out"] > 30.0, arm


def test_a_movement_of_two_bones_is_seen():
    """an exercise only two bones carry (the upper arms, as in a bench press): the same movement is a match,
    another movement with the same timing (the forearms instead) is not. No bone is left out of the
    comparison: leaving out the three that differ most leaves out such a movement itself"""
    ph = syn.phases(reps=3)
    w = main_weights(("l_upper_arm", "r_upper_arm", "l_forearm", "r_forearm"), CFG)
    d = score(arm_exercise(ph, 1), arm_exercise(ph, 2), CFG, w).to_dict()
    assert structures(d) == ["match"] * 8 and not any(s["pose"] for s in d["segments"]), d["segments"]
    d = score(arm_exercise(ph, 1), arm_exercise(ph, 2, part="fore"), CFG, w).to_dict()
    assert structures(d).count("substitution") >= 3, structures(d)


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


def test_visits_are_counted_on_the_scale_of_the_clip():
    """which visit to its place a checkpoint is: the top 0, 1, 2 and the bottom 0, 1 of two repetitions. The
    tops alone are still one place when the bottoms were not found: "the same place" is measured against how
    far the clip moves at all, not against the distance between the checkpoints that happen to be there"""
    from motion_scoring.step2_checkpoints import _count_visits
    clip = Clip(track(syn.phases(reps=2)), CFG, WEIGHTS)
    ev = detect_events(clip, CFG)
    assert [("v" if e.type == "reversal" else e.type[0], e.visit) for e in ev] == \
        [("r", 0), ("v", 0), ("p", 1), ("v", 1), ("r", 2)], [(e.type, e.visit) for e in ev]
    tops = [e for e in ev if e.type != "reversal"]
    _count_visits(clip, tops, CFG)
    assert [e.visit for e in tops] == [0, 1, 2], [e.visit for e in tops]


def test_another_number_of_repetitions():
    """the coach's repetitions are compared with the student's first ones, one whole repetition each, and what
    the student does more often is one added stretch at the end (the other way round: missing)"""
    ph = {n: syn.phases(reps=n) for n in (1, 2, 3, 5)}
    start = lambda k: 0.8 + 2.6 * k                                       # when repetition k (0-based) begins
    for coach, user in ((1, 3), (1, 5), (2, 5), (3, 1), (5, 2)):
        _, d = compare(ph[coach], ph[user])
        k, n = min(coach, user), max(coach, user)
        more, key = ("extra", "student_t") if user > coach else ("missing", "coach_t")
        # k repetitions = 2 k movements + the k - 1 rests between them, then everything else in one unit
        assert structures(d) == ["match"] * (3 * k - 1) + [more], (coach, user, structures(d))
        t0, t1 = d["segments"][-1][key]
        assert abs(t0 - start(k)) < 0.3 and abs(t1 - (start(n - 1) + 2.0)) < 0.3, (coach, user, t0, t1)


def test_another_number_of_repetitions_done_differently():
    """the same where every repetition is another action at the bottom (a substitution): each of the coach's
    repetitions is put against one whole repetition of the student, down and up on both sides, and what one
    of them does more often begins where that repetition ends"""
    start = lambda k: 0.8 + 2.6 * k                                       # when repetition k (0-based) begins
    cfg = replace(CFG, sigma=0.6)                                         # strict matching: a shallow bottom is no match
    for coach, user in ((1, 3), (2, 3), (3, 1), (3, 2)):
        _, d = compare(syn.phases(reps=coach), syn.phases(reps=user, depth=0.35), cfg=cfg)
        k, n = min(coach, user), max(coach, user)
        more, key = ("extra", "student_t") if user > coach else ("missing", "coach_t")
        # k repetitions = k substitutions + the k - 1 rests between them, then everything else in one unit
        assert structures(d) == (["substitution", "match"] * k)[:-1] + [more], (coach, user, structures(d))
        for r, s in enumerate(d["segments"][:-1:2]):
            for t0, t1 in (s["coach_t"], s["student_t"]):
                assert t0 < start(r) + 0.3 and t1 > start(r) + 1.7, (coach, user, r, s)
        t0, t1 = d["segments"][-1][key]
        assert t0 > start(k - 1) + 1.7 and abs(t1 - (start(n - 1) + 2.0)) < 0.3, (coach, user, t0, t1)


def test_mirrored_user():
    ph = syn.phases(reps=3)
    _, d = compare(ph, ph, coach=dict(asym=15.0), user=dict(asym=15.0, mirror=True))
    assert d["mirrored"] is True and set(structures(d)) == {"match"}, (d["mirrored"], structures(d))
    assert not any(s["pose"] for s in d["segments"])
    _, d = compare(ph, ph, coach=dict(asym=15.0), user=dict(asym=15.0))
    assert d["mirrored"] is False


# ---------------------------------------------------------------- playback: what the viewer draws
def playback(coach_ph, user_ph, **kw):
    """-> (Result, to_dict(), playback, coach frame per instant, student frame per instant)"""
    res, d = compare(coach_ph, user_ph, **kw)
    pb = res.playback()
    return res, d, pb, np.array(pb["coach"]), np.array(pb["student"])


def test_playback_shows_every_frame_in_order():
    """both clips from the first frame to the last and no frame skipped, whatever differs"""
    three, four = syn.phases(reps=3), syn.phases(reps=4)
    i = nth(three, 0, 1.0)
    hold = three[:i + 1] + [(1.0, 1.0)] + three[i + 1:]
    for coach, user, kw in ((three, three, {}), (three, four, {}), (four, three, {}), (three, hold, {}),
                            (three, three, dict(user=dict(cut=(1.3, 0.0)))), (syn.phases(reps=2), [(6.0, 0.0)], {})):
        res, d, pb, pc, ps = playback(coach, user, **kw)
        assert len(pc) == len(ps) and len(pb["segments"]) == len(d["segments"])
        for seq, n in ((pc, len(res.C)), (ps, len(res.S))):
            assert seq[0] == 0 and seq[-1] == n - 1 and set(np.diff(seq)) <= {0, 1}, (n, sorted(set(np.diff(seq))))
        spans = [s for s in pb["segments"] if s]
        assert all(a <= b for a, b in spans) and all(x[1] < y[0] for x, y in zip(spans, spans[1:])), spans


def test_playback_waits_where_only_one_side_moves():
    """a movement only the student makes: the coach stands and waits; one only the coach makes: the student does"""
    three, four = syn.phases(reps=3), syn.phases(reps=4)
    # the largest change of a bone direction over the frames drawn (a squat: more than 1; standing: noise)
    moved = lambda clip, frames: float(np.ptp(clip.b[frames.min():frames.max() + 1], axis=0).max())
    res, d, pb, pc, ps = playback(three, four)
    n = structures(d).index("extra")
    k0, k1 = pb["segments"][n]
    assert moved(res.C, pc[k0:k1 + 1]) < 0.1 and (np.diff(ps[k0:k1 + 1]) == 1).all(), moved(res.C, pc[k0:k1 + 1])
    assert abs((k1 - k0) * pb["dt"] - length(d["segments"][n])) < 0.1, (k0, k1, d["segments"][n])
    res, d, pb, pc, ps = playback(four, three)
    n = structures(d).index("missing")
    k0, k1 = pb["segments"][n]                         # it has no time of its own on the student's video
    assert moved(res.S, ps[k0:k1 + 1]) < 0.1 and (np.diff(pc[k0:k1 + 1]) == 1).all(), moved(res.S, ps[k0:k1 + 1])
    assert abs((k1 - k0) * pb["dt"] - length(d["segments"][n], "coach_t")) < 0.1, (k0, k1, d["segments"][n])


def test_playback_keeps_the_real_speed():
    """a slower descent is seen as slower: the coach arrives first and waits; the next units start in step"""
    ph = syn.phases(reps=3)
    user = list(ph)
    user[nth(user, 1, 1.0)] = (1.5, 1.0)                                  # the second descent: 1.0 s -> 1.5 s
    res, d, pb, pc, ps = playback(ph, user)
    n = next(k for k, s in enumerate(d["segments"]) if s["timing"])
    k0, k1 = pb["segments"][n]
    waits = int((np.diff(pc[k0:k1 + 1]) == 0).sum())
    # as much longer as the unit takes the user: about 0.3 of the 0.5 s (the rest before it takes the rest)
    longer = length(d["segments"][n]) - length(d["segments"][n], "coach_t")
    assert (np.diff(ps[k0:k1 + 1]) == 1).all() and longer > 0.25 and abs(waits * pb["dt"] - longer) < 0.1, (waits, longer)
    assert (np.diff(pc[k0:k1 + 1])[-waits:] == 0).all(), "the coach waits at the end, it does not slow down"
    frame = lambda t: round(t / pb["dt"])
    for s, span in list(zip(d["segments"], pb["segments"]))[n + 1:]:      # each at the first frame of its own unit,
        at = (int(pc[span[0]]), int(ps[span[0]]))                         # or the next if the unit before ends on it
        assert 0 <= at[0] - frame(s["coach_t"][0]) <= 1 and 0 <= at[1] - frame(s["student_t"][0]) <= 1, (s, span, at)


def test_playback_meets_at_a_substituted_checkpoint():
    """a substitution: the two checkpoints put against each other are drawn at the same instant although the two
    get there at different times, and the unit ends with both at its last frames"""
    ph = syn.phases(reps=3)
    user = list(ph)
    k = nth(user, 1, 1.0)
    user[k:k + 2] = [(0.5, 0.35), (1.5, 0.0)]                             # the second repetition: shallow, quickly down
    res, d, pb, pc, ps = playback(ph, user, cfg=replace(CFG, sigma=0.6))  # strict matching: that bottom is no match
    assert structures(d).count("substitution") == 1, structures(d)
    n = structures(d).index("substitution")
    k0, k1 = pb["segments"][n]
    o = next(o for o in res.debug["ops"] if o["kind"] == "substitute")
    bottoms = (res.debug["eC"][o["i"]].i, res.debug["eS"][o["j"]].i)
    assert bottoms in set(zip(pc[k0:k1 + 1].tolist(), ps[k0:k1 + 1].tolist())), (bottoms, k0, k1)
    frame = lambda t: round(t / pb["dt"])
    s = d["segments"][n]
    assert (pc[k1], ps[k1]) == (frame(s["coach_t"][1]), frame(s["student_t"][1])), (pc[k1], ps[k1], s)


# ---------------------------------------------------------------- bones: how every bone differs
def test_bones_on_every_match_segment():
    """all the bones of every match segment, labelled or not; none on a segment only one side has"""
    _, d = compare(syn.phases(reps=3), syn.phases(reps=4))                # the same motion, one repetition more
    names = set(track(syn.phases(reps=1)).bone_names)
    assert sum("bones" in s for s in d["segments"]) == 8, structures(d)
    for s in d["segments"]:
        assert ("bones" in s) == (s["structure"] == "match"), s
        for b in s.get("bones", []):
            assert b["student_bone"] == b["bone"], b                      # not mirrored: the same side
            assert set(b["dir"]) == {"up", "forward", "out" if b["bone"][:2] in ("l_", "r_") else "right"}, b
            assert abs(np.linalg.norm(list(b["dir"].values())) - b["deg"]) < 0.2, b  # rounding only
            assert b["deg"] < 3.0, b                                      # nothing differs but the joint noise
        assert not s.get("bones") or {b["bone"] for b in s["bones"]} == names, s


def test_bones_measure_a_known_difference():
    """the user leans the trunk further forward at the bottom: by how much and which way, on the way down and
    on the way up. Too little for a pose label (one bone of ten), but the bones show it"""
    ph = syn.phases(reps=3)
    coach, user = track(ph, seed=1), track(ph, seed=2, lean=1.25)         # 50 deg instead of 40 deg
    d = score(coach, user, CFG, WEIGHTS).to_dict()
    k = coach.bone_names.index("trunk")
    for rep in range(3):
        i = nth(ph, rep, 1.0)
        bottom, top = sum(s for s, _ in ph[:i + 1]), sum(s for s, _ in ph[:i + 2])
        f = int(round(bottom * 30.0))
        truth = float(np.degrees(np.arccos(coach.bones[f, k] @ user.bones[f, k])))    # from the joints themselves
        down, up = bones_at(d, bottom), bones_at(d, bottom, end=False)    # the way down ends there, the way up starts
        assert 8.0 < truth < 12.0, truth
        for b in (down, up):
            assert abs(b["trunk"]["deg"] - truth) < 1.5, (truth, b["trunk"])
            assert b["trunk"]["dir"]["forward"] > 5.0 and b["trunk"]["dir"]["up"] < -3.0, b["trunk"]   # forward and down
            assert all(e["deg"] < 3.0 for n, e in b.items() if "thigh" in n or "shank" in n), b
        assert bones_at(d, top, end=False)["trunk"]["deg"] < 3.0 or rep == 2, bones_at(d, top, end=False)["trunk"]
    assert not any(s["pose"] for s in d["segments"]), [s["pose"] for s in d["segments"]]


def test_bones_do_not_depend_on_the_frame_alignment():
    """a shallower squat: the thighs differ by what the two bottoms differ, on the way down (where it ends) and
    on the way up (where it starts), the same in every repetition. The partners of the frame alignment would
    give another value in every repetition; the pose label names the same bones with the same values"""
    ph, shallow = syn.phases(reps=3), syn.phases(reps=3, depth=0.8)
    coach, user = track(ph, seed=1), track(shallow, seed=2)
    d = score(coach, user, CFG, WEIGHTS).to_dict()
    assert structures(d) == ["match"] * 8, structures(d)
    k = coach.bone_names.index("l_thigh")
    seen = []
    for rep in range(3):
        bottom = sum(s for s, _ in ph[:nth(ph, rep, 1.0) + 1])
        f = int(round(bottom * 30.0))
        truth = float(np.degrees(np.arccos(coach.bones[f, k] @ user.bones[f, k])))
        for b in (bones_at(d, bottom), bones_at(d, bottom, end=False)):
            assert abs(b["l_thigh"]["deg"] - truth) < 2.0 and abs(b["r_thigh"]["deg"] - truth) < 2.0, (truth, b["l_thigh"])
            assert b["l_thigh"]["dir"]["up"] < -5.0, b["l_thigh"]         # his thigh stays further from level
            seen.append(b["l_thigh"]["deg"])
    assert max(seen) - min(seen) < 2.0, seen
    for s in d["segments"]:
        if s["pose"]:                                                     # worst = the two largest of the bones
            deg = {b["bone"]: b["deg"] for b in s["bones"]}
            assert all(deg[w["bone"]] == w["diff_deg"] for w in s["pose"]["worst"]), s["pose"]
            assert sorted(w["diff_deg"] for w in s["pose"]["worst"]) == sorted(deg.values())[-2:], s["pose"]
    assert any(s["pose"] for s in d["segments"])


def test_bones_name_the_students_own_side():
    ph = syn.phases(reps=3)
    # both knees 4 cm closer together: about 5 deg in each of the four leg bones, and the bones show which way
    # (with the legs as main bones that is just over tau where tau is at its floor of 3 deg: a pose label or not)
    _, d = compare(ph, ph, user=dict(knees_cm=(4.0, 4.0)))
    assert set(structures(d)) == {"match"}, d["segments"]
    for s in d["segments"]:
        b = {x["bone"]: x for x in s["bones"]}
        assert all(-9.0 < b[n]["dir"]["out"] < -3.0 for n in ("l_thigh", "r_thigh")), (b["l_thigh"], b["r_thigh"])
        assert all(3.0 < b[n]["dir"]["out"] < 9.0 for n in ("l_shank", "r_shank")), (b["l_shank"], b["r_shank"])
    # a mirrored user: the knee moved is his own right one, and he tilts 5 deg further to his own right
    _, d = compare(ph, ph, coach=dict(asym=15.0), user=dict(asym=20.0, knees_cm=(4.0, 0.0), mirror=True))
    assert d["mirrored"] is True and set(structures(d)) == {"match"}, (d["mirrored"], structures(d))
    b = bones_at(d, sum(s for s, _ in ph[:nth(ph, 0, 1.0) + 1]))          # the way down to the first bottom
    assert b["l_thigh"]["student_bone"] == "r_thigh" and b["trunk"]["student_bone"] == "trunk", b["l_thigh"]
    assert b["l_thigh"]["dir"]["out"] < -3.0, b["l_thigh"]
    assert abs(b["r_thigh"]["dir"]["out"]) < 2.5, b["r_thigh"]
    assert 3.0 < b["trunk"]["dir"]["right"] < 8.0, b["trunk"]


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
        assert all(set(b) == {"bone", "student_bone", "deg", "dir"} for b in s.get("bones", []))
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
