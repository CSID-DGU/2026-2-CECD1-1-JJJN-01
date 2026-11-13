"""S1: bring hip-centred observations into PAN's Mixamo convention.

  fill_gaps      linear interpolation of low-confidence / missing frames per joint
  resample       to 30 fps (PAN training rate)
  face_forward   yaw so the initial facing direction is +Z (Mixamo convention)
  anchor_root    rebuild the global root trajectory, which MediaPipe does not provide:
                 vertical  - lowest observed foot joint on the training ground height
                 horizontal- the lower (stance) foot stays fixed in the world
                 Assumes at least one foot on the ground in every frame (true for squats,
                 lunges, walking; wrong during flight phases such as jumps).
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

FEET = ('LeftFoot', 'RightFoot', 'LeftToeBase', 'RightToeBase')


def fill_gaps(obs, weight, max_gap_ratio=0.5):
    """Interpolate frames where a joint has weight 0 (not detected), if observed often enough.

    Leading/trailing gaps hold the nearest observed value (np.interp); they are given a
    low weight so the IK prior dominates there.
    """
    obs, weight = obs.copy(), weight.copy()
    T = len(obs)
    t = np.arange(T)
    for j in range(1, obs.shape[1]):
        ok = weight[:, j] > 0
        if ok.all() or ok.sum() < max(2, (1 - max_gap_ratio) * T):
            continue
        for c in range(3):
            obs[~ok, j, c] = np.interp(t[~ok], t[ok], obs[ok, j, c])
        weight[~ok, j] = 0.25 * weight[ok, j].mean()          # interpolated: low confidence
    return obs, weight


def resample(x, fps_in, fps_out=30.):
    """Linear resampling along axis 0."""
    if abs(fps_in - fps_out) < 1e-3:
        return np.asarray(x)
    T = len(x)
    t_out = np.arange(0, (T - 1) / fps_in + 1e-9, 1. / fps_out)
    t_in = np.arange(T) / fps_in
    flat = np.asarray(x, dtype=np.float64).reshape(T, -1)
    out = np.stack([np.interp(t_out, t_in, flat[:, k]) for k in range(flat.shape[1])], 1)
    return out.reshape((len(t_out),) + np.shape(x)[1:])


def yaw_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def facing_yaw(obs, weight, skel, frames=10):
    """Yaw (rad, about +Y) of the initial facing direction, from hips and shoulders."""
    left = np.zeros(3)
    for a, b in (('LeftUpLeg', 'RightUpLeg'), ('LeftArm', 'RightArm')):
        i, k = skel.index(a), skel.index(b)
        ok = (weight[:frames, i] > 0) & (weight[:frames, k] > 0)
        if ok.any():
            left += (obs[:frames, i] - obs[:frames, k])[ok].mean(0)
    facing = np.cross(left, [0., 1., 0.])                    # left x up = forward
    if np.linalg.norm(facing[[0, 2]]) < 1e-6:
        raise ValueError('Cannot determine facing direction')
    return float(np.arctan2(facing[0], facing[2]))


def face_forward(obs, weight, skel):
    theta = facing_yaw(obs, weight, skel)
    R = yaw_matrix(-theta)
    return obs @ R.T, theta


def anchor_root(rel, weight, skel, ground_y, smooth_sigma=1.):
    """rel: hip-centred (T, J, 3) -> world positions with a reconstructed root path."""
    feet = [skel.index(n) for n in FEET if weight[:, skel.index(n)].any()]
    if not feet:
        raise ValueError('No foot joints observed')
    y = np.where(weight[:, feet] > 0, rel[:, feet, 1], np.inf)
    lowest = y.min(1)
    lowest = np.where(np.isfinite(lowest), lowest, np.nan)
    if np.isnan(lowest).all():
        raise ValueError('Feet never observed')
    idx = np.arange(len(rel))
    lowest = np.interp(idx, idx[~np.isnan(lowest)], lowest[~np.isnan(lowest)])
    shift_y = ground_y - lowest
    # Stance foot = lower ankle; keep it fixed horizontally.
    la, ra = skel.index('LeftFoot'), skel.index('RightFoot')
    stance = np.where(rel[:, la, 1] <= rel[:, ra, 1], la, ra)
    root_xz = np.zeros((len(rel), 2))
    for t in range(1, len(rel)):
        f = stance[t]
        root_xz[t] = root_xz[t - 1] - (rel[t, f, [0, 2]] - rel[t - 1, f, [0, 2]])
    offset = np.stack([root_xz[:, 0], shift_y, root_xz[:, 1]], 1)
    if smooth_sigma > 0:
        offset = gaussian_filter1d(offset, smooth_sigma, axis=0, mode='nearest')
    return rel + offset[:, None]


def ground_align(obs, weight, skel, ground_y, percentile=5., mode='clip', sigma=3.):
    """Shift vertically so the lowest observed foot joint rests on `ground_y` (for estimators
    that already provide a world trajectory, e.g. WHAM).

    mode='clip': one shift for the whole clip, from a low percentile over frames of the
      per-frame lowest foot joint, so brief foot lifts do not move the floor (jumps, steps).
    mode='frame': per-frame shift (Gaussian-smoothed, `sigma` frames) so the lowest foot joint
      is on the floor in every frame. Removes WHAM's slow vertical drift (seen: ~15 cm during a
      squat hold) but assumes a foot on the ground at all times -- not for jumps.
    """
    feet = [skel.index(n) for n in FEET if weight[:, skel.index(n)].any()]
    if not feet:
        raise ValueError('No foot joints observed')
    y = np.where(weight[:, feet] > 0, obs[:, feet, 1], np.inf).min(1)
    out = obs.copy()
    if mode == 'clip':
        out[..., 1] += ground_y - np.percentile(y[np.isfinite(y)], percentile)
    elif mode == 'frame':
        idx = np.arange(len(y))
        ok = np.isfinite(y)
        y = np.interp(idx, idx[ok], y[ok])
        if sigma > 0:
            y = gaussian_filter1d(y, sigma, mode='nearest')
        out[..., 1] += (ground_y - y)[:, None]
    else:
        raise ValueError(f'Unknown ground mode {mode}')
    return out

