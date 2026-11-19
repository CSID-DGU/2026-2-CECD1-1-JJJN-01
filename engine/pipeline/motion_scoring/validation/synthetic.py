"""Synthetic squat skeletons for the tests and the example (no video, no data files).

    joints, fps = skeleton(phases(reps=3))            # [T, 22, 3], SMPL-22 order, cm, Y up, facing +Z

A clip is a list of phases [(seconds, depth at the end)], depth 0 = standing .. 1 = full squat; a phase
that ends at the depth it starts from is a hold. `phases()` writes the usual list, and a test changes it
(a longer descent, an extra hold, one repetition less) before turning it into joints.

Body parts do not turn at the same instant: the trunk follows the legs and the arms lead them, as in real
video. Without that the speed would fall to exactly zero at every turn and any detector would find it;
with it the speed only dips, to 0.14 of the reference for a 1 s movement (0.12-0.16 measured on a real
clip), 0.20 for a 0.7 s one and 0.09 for a 1.5 s one: the same turn is shallower or deeper with the tempo.
"""
from __future__ import annotations

import numpy as np

from motion_scoring.step1_preprocess import SMPL_NAMES      # the script that imports this puts the package on sys.path

TRUNK_LAG_S = 0.09       # the trunk follows the legs
ARM_LEAD_S = 0.07        # the arms lead them
DEG = np.pi / 180.0


def phases(reps=3, down=1.0, up=1.0, bottom=0.0, top=0.6, lead=0.8, tail=0.8, depth=1.0):
    """The phases of `reps` squats: `down` / `up` seconds per movement, `bottom` seconds held at the bottom
    (0 = straight back up), `top` seconds standing between repetitions (0 = straight back down), `lead` /
    `tail` seconds standing still before / after (0 = the clip starts / ends at the movement)."""
    out = [(lead, 0.0)] if lead else []
    for r in range(reps):
        out.append((down, depth))
        if bottom:
            out.append((bottom, depth))
        out.append((up, 0.0))
        if top and r + 1 < reps:
            out.append((top, 0.0))
    return out + ([(tail, 0.0)] if tail else [])


def _depth(ph, t):
    """Depth at the times t for the phases ph (starting from standing). A movement follows half a cosine:
    it starts and ends with zero speed, and turns as sharply as a real one."""
    q = np.zeros(len(t))
    t0, q0 = 0.0, 0.0
    for seconds, q1 in ph:
        m = (t >= t0) & (t < t0 + seconds)
        q[m] = q0 + (q1 - q0) * (1 - np.cos(np.pi * (t[m] - t0) / seconds)) / 2
        t0, q0 = t0 + seconds, q1
    q[t >= t0] = q0
    return q


def skeleton(ph, fps=30.0, noise_cm=0.0, seed=0, lean=1.0, asym=0.0):
    """-> (joints [T, 22, 3] in SMPL-22 order, fps).

    noise_cm  independent noise on every joint coordinate (seeded)
    lean      forward lean of the trunk at the bottom, 1 = the usual 40 deg
    asym      deg: the body tilts to its left and raises the left arm higher as it goes down, so that a
              mirrored copy is a different motion (0 = left / right symmetric)"""
    t = np.arange(int(round(sum(s for s, _ in ph) * fps)) + 1) / fps
    legs, trunk, arms = _depth(ph, t), _depth(ph, t - TRUNK_LAG_S), _depth(ph, t + ARM_LEAD_S)
    zero = np.zeros(len(t))
    xyz = lambda x, y, z: np.stack([x + zero, y + zero, z + zero], 1)
    shank, thigh = 28 * DEG * legs, 95 * DEG * legs            # shank tips forward, thigh back
    knee_y, knee_z = 8 + 42 * np.cos(shank), 42 * np.sin(shank)
    hip_y, hip_z = knee_y + 44 * np.cos(thigh), knee_z - 44 * np.sin(thigh)
    fwd, side = lean * 40 * DEG * trunk, asym * DEG * trunk      # trunk: forward lean, tilt to the left
    pelvis = xyz(0, hip_y + 5, hip_z)
    neck = pelvis + 50 * xyz(np.sin(side), np.cos(fwd) * np.cos(side), np.sin(fwd))
    J = dict(pelvis=pelvis, neck=neck,
             head=neck + 18 * xyz(np.sin(side), np.cos(0.4 * fwd), np.sin(0.4 * fwd)))
    for k, f in enumerate((0.25, 0.5, 0.75), 1):
        J[f"spine{k}"] = pelvis + f * (neck - pelvis)
    for name, sx in (("left", 1.0), ("right", -1.0)):            # the person's left is +X
        J[f"{name}_ankle"] = xyz(sx * 12, 8, 0)
        J[f"{name}_foot"] = xyz(sx * 12, 2, 14)
        J[f"{name}_knee"] = xyz(sx * (12 + 3 * legs), knee_y, knee_z)
        J[f"{name}_hip"] = xyz(sx * 9, hip_y, hip_z)
        J[f"{name}_collar"] = neck + xyz(sx * 7, -3, 0)
        J[f"{name}_shoulder"] = neck + xyz(sx * 18, -5, 0)
        raise_ = (15 + 65 * arms + (1.5 * asym * arms if name == "left" else 0)) * DEG   # arms reach forward
        J[f"{name}_elbow"] = J[f"{name}_shoulder"] + 28 * xyz(0, -np.cos(raise_), np.sin(raise_))
        J[f"{name}_wrist"] = J[f"{name}_elbow"] + 26 * xyz(0, -np.cos(raise_ + 10 * DEG), np.sin(raise_ + 10 * DEG))
    joints = np.stack([J[n] for n in SMPL_NAMES[:22]], 1)
    if noise_cm:
        joints = joints + np.random.default_rng(seed).normal(0.0, noise_cm, joints.shape)
    return joints, float(fps)


def mirrored(joints):
    """The same motion done like a mirror image (left and right swapped)."""
    names = list(SMPL_NAMES[:22])
    swap = [names.index(n.replace("left", "right") if n.startswith("left") else n.replace("right", "left"))
            for n in names]
    out = joints[:, swap].copy()
    out[..., 0] *= -1
    return out
