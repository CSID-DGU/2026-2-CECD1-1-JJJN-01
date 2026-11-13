"""Temporal filtering of observed joint positions (per clip, never across clip borders)."""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def smooth_positions(obs, segments, sigma):
    """Gaussian low-pass along time. obs: (T, J, 3); segments: (T,) clip ids; sigma in frames."""
    if sigma <= 0:
        return obs
    out = np.array(obs, dtype=np.float64, copy=True)
    segments = np.asarray(segments)
    for s in np.unique(segments):
        idx = np.flatnonzero(segments == s)
        out[idx] = gaussian_filter1d(out[idx], sigma, axis=0, mode='nearest')
    return out
