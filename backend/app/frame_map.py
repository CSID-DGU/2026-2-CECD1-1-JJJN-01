"""frameMap: for every user frame, which coach pose to draw (a fractional coach frame) and how the two relate.

The frontend plays the user's video frame by frame and overlays the coach. CP-DTW's path (Result.debug['path'],
steps (kind, coach i, user j)) can pair one user frame with many coach frames (warp_c: the coach goes through
several frames while the user stays) or many user frames with one coach frame (warp_s: the coach waits). Picking
the first pair per user frame (the old frontend rule) skips coach frames: 17 -> 30 in one step, about half the
coach frames never shown. Here, per user frame:

  centre   the middle of its coach range (17..29 -> 23), so a fast stretch is split evenly on both sides
  edges    outside the path (before the first / after the last anchor) the user and coach clip ends are joined
           by a straight line, so the coach starts / finishes moving instead of standing frozen
  smooth   a centred moving average (SMOOTH frames) spreads the remaining jumps over neighbouring frames; it
           keeps the sequence non-decreasing and leaves long waits (the user's bottom pause) flat in the middle
  kind     together (both advance) | coachAhead (the coach goes through several frames here) | coachWaits
           (the coach stays on one pose while the user moves) | userOnly (drop_s) | outside (no CP-DTW pair)

Output (columnar, short): {coachFrame: [float per user frame], kind: [str per user frame]}. The frontend
draws coach pose = interpolation between floor / ceil of coachFrame.
"""
import numpy as np

SMOOTH = 9          # frames (0.3 s at 30 fps): drawn coach moves <= ~8 cm / frame (user's own max ~5 cm),
                    # a 0.85 s bottom pause stays flat for ~0.55 s (only its edges blend)


def frame_map(path, n_user, n_coach, smooth=SMOOTH):
    by_user = {}
    for kind, i, j in path:
        by_user.setdefault(int(j), []).append((int(i), kind))
    raw = np.full(n_user, np.nan)
    kind = ['outside'] * n_user
    for j, pairs in by_user.items():
        if not 0 <= j < n_user:
            continue
        cs = [i for i, _ in pairs]
        ks = {k for _, k in pairs}
        raw[j] = (min(cs) + max(cs)) / 2
        kind[j] = ('userOnly' if ks <= {'drop_s'} else 'coachAhead' if 'warp_c' in ks or len(set(cs)) > 1
                   else 'together' if 'match' in ks else 'coachWaits')
    have = np.where(~np.isnan(raw))[0]
    if not len(have):                                       # no path at all: plain proportional mapping
        raw = np.linspace(0, n_coach - 1, n_user)
    else:
        a, b = int(have[0]), int(have[-1])
        if a > 0:                                           # join the clip starts
            raw[:a + 1] = np.linspace(0, raw[a], a + 1)
        if b < n_user - 1:                                  # join the clip ends
            raw[b:] = np.linspace(raw[b], n_coach - 1, n_user - b)
        inside = np.arange(a, b + 1)                        # user frames dropped inside the path: hold
        raw[inside] = np.maximum.accumulate(np.where(np.isnan(raw[inside]), -np.inf, raw[inside]))
    if smooth > 1 and n_user >= smooth:
        pad = smooth // 2
        padded = np.r_[np.full(pad, raw[0]), raw, np.full(pad, raw[-1])]
        raw = np.convolve(padded, np.ones(smooth) / smooth, mode='valid')
    raw = np.clip(raw, 0, n_coach - 1)
    return dict(coachFrame=[round(float(x), 2) for x in raw], kind=kind)


def first_pair_lookup(alignment, n_user):
    """The previous frontend rule (utils/skeleton.ts buildAlignmentLookup), for comparisons."""
    lk = [-1] * n_user
    for s, c in alignment:
        if 0 <= s < n_user and lk[s] == -1:
            lk[s] = c
    last = 0
    for i in range(n_user):
        if lk[i] == -1:
            lk[i] = last
        else:
            last = lk[i]
    return lk
