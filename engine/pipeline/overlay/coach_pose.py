"""The coach's pose in the common format of the overlay, from either source (PLAN.md 5).

Common format (place.py): per coach BVH frame (what pb['coach'] counts)
  tilt       (T, 3, 3)  root rotation without the heading (yaw), world Y up
  body_pose  (T, 69)    SMPL local rotations, axis-angle
No betas and no position: the student's are used when placing.

  from_wham     method 1: the coach's WHAM SMPL rotations (pose_world) as they are, no PAN
  from_pan_bvh  method 2: PAN output BigVegas rotations -> SMPL (PLAN.md 5, core in bigvegas_to_smpl)
"""
import numpy as np

from .place import npz_frame, pose_from_npz
from .smpl_np import SMPL22, aa_to_mat, mat_to_aa, split_yaw

SMPL_NAMES24 = SMPL22 + ['left_hand', 'right_hand']

# SMPL kinematic tree (24 joints) and the SMPL joint -> BigVegas joint map of pipeline/bvh_to_npz.py
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

# SMPL joint index pairs (left, right); body_pose holds joints 1..23
MIRROR_PAIRS = [(1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23)]
FLIP = np.diag([-1., 1., 1.])          # x = the body's left-right axis


def from_wham(npz, report, n_bvh):
    """Method 1. -> pose indexed by coach BVH frame (0..n_bvh-1). Frames WHAM missed take the nearest
    frame it saw (run_video fills the same gaps before the BVH)."""
    pose = pose_from_npz(npz)
    seen = np.flatnonzero(np.isfinite(pose['body_pose'][:, 0]))
    idx = np.array([npz_frame(i, report) for i in range(n_bvh)])
    idx = np.minimum(idx, len(pose['body_pose']) - 1)
    idx = seen[np.abs(seen[None, :] - idx[:, None]).argmin(1)]
    return dict(tilt=pose['tilt'][idx], body_pose=pose['body_pose'][idx], npz_frame=idx)


def mirror(pose):
    """The same movement done the other way round (left <-> right): swap the paired joints and reflect every
    rotation in the body's sagittal plane (axis-angle (x, y, z) -> (x, -y, -z); a matrix R -> F R F).
    The same meaning as the backend's shown_coach (x -> -x, left/right names swapped)."""
    T = len(pose['body_pose'])
    full = np.concatenate([np.zeros((T, 3)), pose['body_pose']], 1).reshape(T, 24, 3)
    for a, b in MIRROR_PAIRS:
        full[:, [a, b]] = full[:, [b, a]]
    full = full * np.array([1., -1., -1.])
    return dict(pose, tilt=FLIP @ pose['tilt'] @ FLIP, body_pose=full[:, 1:].reshape(T, 69))


# ---------- method 2: BigVegas (PAN output) -> SMPL ----------
def min_rotation(a, b):
    """(3,) unit a -> unit b: the smallest rotation taking a onto b, (3, 3)."""
    v, c = np.cross(a, b), float(np.dot(a, b))
    if c < -1 + 1e-9:                                         # opposite: half turn about any perpendicular axis
        axis = np.cross(a, [1., 0., 0.] if abs(a[0]) < 0.9 else [0., 1., 0.])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def fit_rotation(src, dst):
    """Least-squares rotation R with dst ~= R src, per frame. src (N, 3), dst (T, N, 3) -> (T, 3, 3).
    The same problem as common/quat.horn (Horn 1987), solved with the SVD (Kabsch) in numpy."""
    H = np.einsum('ni,tnj->tij', src, dst)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.swapaxes(Vt, -1, -2) @ np.swapaxes(U, -1, -2)))
    D = np.zeros_like(H)
    D[:, 0, 0], D[:, 1, 1], D[:, 2, 2] = 1., 1., d
    return np.swapaxes(Vt, -1, -2) @ D @ np.swapaxes(U, -1, -2)


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def chordal_mean(R):
    """(T, 3, 3) rotations -> (3, 3) the rotation closest to their mean matrix."""
    U, _, Vt = np.linalg.svd(R.mean(0))
    return U @ np.diag([1., 1., np.sign(np.linalg.det(U @ Vt))]) @ Vt


def _convert(G_bv, P_bv, offsets, parents, names, smpl_from_bv, pick):
    """SMPL global rotations G_j = D_j . K_j, joint by joint down the SMPL tree. D_j, what BigVegas gives:
      one mapped child c   the global rotation of the BigVegas joint right above c (the one turning that bone)
      several (pelvis, spine3)  the least-squares rotation of the BigVegas child bones, rest -> posed
      none (wrists, feet, head, hands)  the SMPL parent's global rotation (K is then a local rotation)
    pick(j, kind, D, default) -> K_j (3, 3); default is the rest-pose match (see bigvegas_to_smpl)."""
    T = len(G_bv)
    idx = {j: names.index(b) for j, b in smpl_from_bv.items()}
    rest_pos = np.zeros((len(parents), 3))                   # BigVegas rest pose: every rotation identity
    for b, p in enumerate(parents):
        rest_pos[b] = rest_pos[p] + offsets[b] if p >= 0 else 0.
    G = np.zeros((T, 24, 3, 3))
    for j in range(24):                                       # parents come first in SMPL order
        ch = [c for c in range(24) if SMPL_PARENTS[c] == j and c in idx]
        if j in idx and len(ch) == 1:
            G[:, j] = G_bv[:, parents[idx[ch[0]]]] @ pick(j, 'bone', G_bv[:, parents[idx[ch[0]]]], ch)
        elif j in idx and len(ch) > 1:
            rest = unit(np.stack([rest_pos[idx[c]] - rest_pos[idx[j]] for c in ch]))
            posed = unit(np.stack([P_bv[:, idx[c]] - P_bv[:, idx[j]] for c in ch], 1))
            D = fit_rotation(rest, posed)
            G[:, j] = D @ pick(j, 'fit', D, (ch, rest))
        else:
            D = G[:, SMPL_PARENTS[j]] if j else np.broadcast_to(np.eye(3), (T, 3, 3))
            G[:, j] = D @ pick(j, 'leaf', D, None)
    return G


def calibrate(G_bv, P_bv, offsets, parents, names, smpl_from_bv, G_target):
    """The fixed rotations K_j that take what BigVegas gives (D_j above) to G_target, SMPL global rotations of
    the same motion in the same world frame (the coach's WHAM, against the coach's IK BVH). One K per joint:
    the part of the difference that stays the same over the frames, i.e. how the two rigs define their
    joints (BigVegas' clavicles point ~35 deg up, its chest and neck are split differently, the IK leaves
    clavicles / Spine2 / Neck to the prior). Frames where G_target is not finite are left out.
    -> K (24, 3, 3), spread_deg {smpl joint: mean angle left over the frames}."""
    ok = np.isfinite(G_target).all((1, 2, 3))
    K, spread = np.zeros((24, 3, 3)), {}

    def pick(j, kind, D, info):
        rel = np.swapaxes(D[ok], -1, -2) @ G_target[ok, j]
        K[j] = chordal_mean(rel)
        spread[SMPL_NAMES24[j]] = round(float(np.degrees(np.arccos(np.clip(
            (np.trace(np.swapaxes(rel, -1, -2) @ K[j], axis1=-2, axis2=-1) - 1) / 2, -1, 1))).mean()), 3)
        return K[j]
    _convert(G_bv, P_bv, offsets, parents, names, smpl_from_bv, pick)
    return K, spread


def bigvegas_to_smpl(G_bv, P_bv, offsets, parents, names, smpl_rest, smpl_from_bv, calib=None):
    """Core of method 2, arrays only (tested without model files).
    calib (24, 3, 3), from calibrate: the fixed per-joint rotations K_j used instead of the rest-pose match
    below (G_j = D_j . K_j, see _convert); smpl_rest is then not used.

    G_bv (T, J, 3, 3) global rotations and P_bv (T, J, 3) positions of every BigVegas joint (official FK,
    `*_virtual` joints included); offsets (J, 3) and parents (J,) the BVH hierarchy, whose zero-rotation
    pose is the BigVegas rest pose; smpl_rest (24, 3) SMPL joints of the zero pose (student betas);
    smpl_from_bv {smpl joint index: BigVegas name}.
    -> SMPL global rotations (T, 24, 3, 3), Y up as the BVH.

    A SMPL joint with one mapped child: G_smpl = G_bv(d) . R_rest, d the BigVegas joint right above that
    child (the one that turns that bone), R_rest the smallest rotation from the SMPL rest bone to the
    BigVegas rest bone: the posed bones then point the same way, and the twist about the bone is BigVegas'.
    Several mapped children (pelvis, spine3): the same with fitted rotations (least squares over the
    children): SMPL rest -> BigVegas rest, then BigVegas rest -> posed. Exact when the children move as one;
    PAN also turns each bone out of the pelvis / Spine2 on its own (`*_virtual`), which one SMPL rotation
    cannot follow, so those bones keep a residual (compare.json fit_residual_deg). No mapped child (wrists, head, feet, hands): local rotation identity.
    The rest-pose match points the bones as BigVegas' do, so it carries over what the two rigs define
    differently (clavicle angle, chest split); calib takes that fixed part out."""
    if calib is not None:
        return _convert(G_bv, P_bv, offsets, parents, names, smpl_from_bv, lambda j, kind, D, info: calib[j])
    smpl_rest = np.asarray(smpl_rest, np.float64)

    def pick(j, kind, D, info):
        if kind == 'bone':
            c = info[0]
            return min_rotation(unit(smpl_rest[c] - smpl_rest[j]), unit(offsets[names.index(smpl_from_bv[c])]))
        if kind == 'fit':
            # the same two steps as above: SMPL rest -> BigVegas rest (a fixed fit: the two rest poses have
            # differently shaped pelvises / chests), then BigVegas rest -> posed (D)
            ch, rest = info
            return fit_rotation(unit(np.stack([smpl_rest[c] - smpl_rest[j] for c in ch])), rest[None])[0]
        return np.eye(3)
    return _convert(G_bv, P_bv, offsets, parents, names, smpl_from_bv, pick)


def local_to_globals(root, body_pose):
    """root (T, 3, 3) and body_pose (T, 69) SMPL local rotations -> global rotations (T, 24, 3, 3)."""
    L = aa_to_mat(np.asarray(body_pose, np.float64).reshape(len(root), 23, 3))
    G = np.zeros((len(root), 24, 3, 3))
    G[:, 0] = root
    for j in range(1, 24):
        G[:, j] = G[:, SMPL_PARENTS[j]] @ L[:, j - 1]
    return G


def globals_to_pose(G):
    """SMPL global rotations (T, 24, 3, 3) -> the common pose format (tilt, body_pose)."""
    par = np.array(SMPL_PARENTS[1:])
    local = np.swapaxes(G[:, par], -1, -2) @ G[:, 1:]
    _, tilt = split_yaw(G[:, 0])
    return dict(tilt=tilt, body_pose=mat_to_aa(local).reshape(len(G), 69))


def load_bigvegas(bvh):
    """A BigVegas BVH -> (G_bv, P_bv, offsets, parents, names), smpl_from_bv: the arrays _convert takes."""
    import sys
    from pipeline.bvh_to_npz import SMPL_FROM_BIGVEGAS
    from pipeline.common.paths import REPO
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import outer_utils.Animation as Animation
    import outer_utils.BVH as BVH
    anim, names, _ = BVH.load(str(bvh))
    M = Animation.transforms_global(anim)
    smpl_from_bv = {SMPL22.index(s): b for s, b in SMPL_FROM_BIGVEGAS.items()}
    return (M[..., :3, :3], M[..., :3, 3], np.asarray(anim.offsets), np.asarray(anim.parents), list(names)), smpl_from_bv


def from_pan_bvh(bvh, smpl_rest, calib=None):
    """Method 2: a BigVegas BVH (PAN output, or the IK BVH for the round trip) -> pose indexed by its frames.
    calib: from calibrate_from_wham (None: the rest-pose match)."""
    args, smpl_from_bv = load_bigvegas(bvh)
    G = bigvegas_to_smpl(*args, smpl_rest, smpl_from_bv, calib=calib)
    return dict(globals=G, fit_residual_deg=rigid_fit_residual(*args, smpl_from_bv), **globals_to_pose(G))


def calibrate_from_wham(ik_bvh, npz, report):
    """The calibration (calibrate) of a run_video run: its IK BVH (source_pan_input.bvh, BigVegas) against its
    own WHAM rotations (pose_world), frame for frame. The BVH world is WHAM's turned to face +Z
    (report initial_yaw_deg, run_video S1); BVH frame i is npz frame place.npz_frame(i).
    -> K (24, 3, 3), spread_deg."""
    from pipeline.skeleton_extract.to_mixamo_format.canonicalize import yaw_matrix
    args, smpl_from_bv = load_bigvegas(ik_bvh)
    f = np.array([npz_frame(i, report) for i in range(len(args[0]))])
    f = np.minimum(f, len(npz['pose_world']) - 1)
    p = np.asarray(npz['pose_world'], np.float64)[f]
    G = local_to_globals(aa_to_mat(np.nan_to_num(p[:, :3])), np.nan_to_num(p[:, 3:]))
    G = yaw_matrix(-np.radians(report['initial_yaw_deg'])) @ G
    G[~np.isfinite(p).all(1)] = np.nan
    return calibrate(*args, smpl_from_bv, G)


def rigid_fit_residual(G_bv, P_bv, offsets, parents, names, smpl_from_bv):
    """How far the bones out of the pelvis / spine3 move apart from one rigid motion (PAN turns them through
    the `*_virtual` joints on their own; one SMPL rotation follows only their best common motion).
    -> {joint: {child: {mean, max}}} degrees."""
    from pipeline.overlay.smpl_np import SMPL22
    idx = {j: names.index(b) for j, b in smpl_from_bv.items()}
    rest_pos = np.zeros((len(parents), 3))
    for b, p in enumerate(parents):
        rest_pos[b] = rest_pos[p] + offsets[b] if p >= 0 else 0.
    out = {}
    for j in range(24):
        ch = [c for c in range(24) if SMPL_PARENTS[c] == j and c in idx]
        if j not in idx or len(ch) < 2:
            continue
        rest = unit(np.stack([rest_pos[idx[c]] - rest_pos[idx[j]] for c in ch]))
        posed = unit(np.stack([P_bv[:, idx[c]] - P_bv[:, idx[j]] for c in ch], 1))
        fit = np.einsum('tij,nj->tni', fit_rotation(rest, posed), rest)
        e = np.degrees(np.arccos(np.clip((fit * posed).sum(-1), -1, 1)))
        out[SMPL22[j]] = {SMPL22[c]: dict(mean=round(float(e[:, i].mean()), 3), max=round(float(e[:, i].max()), 3))
                          for i, c in enumerate(ch)}
    return out
