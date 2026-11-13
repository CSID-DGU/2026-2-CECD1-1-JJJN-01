"""Person skeleton estimation from observed joint positions.

Keeps the source character's rest directions (so rotations stay comparable to PAN's
training data) and replaces bone lengths with what the observations support:
  * parent-child bones: median distance where both joints are observed
  * widths: for a left/right joint pair whose connecting bones are unobserved (hips via
    Pelvis, shoulder joints via Spine2 -> clavicles), every bone on the path to their
    common ancestor is scaled so the rest-pose left-right distance matches the observed
    one. Mixamo BigVegas has hip joints 37 cm and arm joints 81 cm apart; MediaPipe's
    hips/shoulders are ~20-25 cm apart, which otherwise forces extreme clavicle/pelvis
    rotations far outside PAN's training distribution.
  * everything else: median scale of the measured parent-child bones
"""
import numpy as np

WIDTHS = (('LeftUpLeg', 'RightUpLeg'), ('LeftArm', 'RightArm'))


def _path_to(skel, j, ancestor):
    path = []
    while j != ancestor:
        path.append(j)
        j = skel.parents[j]
    return path


def _ancestors(skel, j):
    out = []
    while j != -1:
        out.append(j)
        j = skel.parents[j]
    return out


def rest_positions(skel, offsets=None):
    off = skel.offsets if offsets is None else offsets
    pos = np.zeros((skel.n_joints, 3))
    for j in range(1, skel.n_joints):
        pos[j] = pos[skel.parents[j]] + off[j]
    return pos


def estimate_skeleton(skel, obs, mask, unreliable=(), widths=WIDTHS):
    lengths = np.linalg.norm(skel.offsets, axis=1)
    est = lengths.copy()
    measured = np.zeros(len(lengths), dtype=bool)
    for j in range(1, skel.n_joints):
        p = skel.parents[j]
        both = mask[:, j] & mask[:, p]
        if lengths[j] > 1e-6 and both.any() and skel.names[j] not in unreliable:
            est[j] = np.median(np.linalg.norm(obs[both, j] - obs[both, p], axis=1))
            measured[j] = True
    for j, n in enumerate(skel.names):                         # left/right symmetry
        if n.startswith('Left'):
            k = skel.index('Right' + n[4:])
            if measured[j] and measured[k]:
                est[j] = est[k] = (est[j] + est[k]) / 2
    by_width = np.zeros(len(lengths), dtype=bool)
    rest = rest_positions(skel)
    for a, b in widths:
        i, k = skel.index(a), skel.index(b)
        both = mask[:, i] & mask[:, k]
        anc_i = _ancestors(skel, i)
        common = next(x for x in _ancestors(skel, k) if x in anc_i)
        chain = _path_to(skel, i, common) + _path_to(skel, k, common)
        if not both.any() or measured[chain].any():
            continue
        ratio = np.median(np.linalg.norm(obs[both, i] - obs[both, k], axis=1)) / \
            np.linalg.norm(rest[i] - rest[k])
        est[chain] = lengths[chain] * ratio
        by_width[chain] = True
    ok_scale = measured & (lengths > 1e-6)
    scale = float(np.median(est[ok_scale] / lengths[ok_scale])) if ok_scale.any() else 1.
    rest_of = ~measured & ~by_width
    est[rest_of] = lengths[rest_of] * scale
    direction = np.divide(skel.offsets, lengths[:, None], out=np.zeros_like(skel.offsets),
                          where=lengths[:, None] > 1e-6)
    return skel.with_offsets(direction * est[:, None])
