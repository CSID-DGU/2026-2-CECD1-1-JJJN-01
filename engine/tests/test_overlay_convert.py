"""Method 2 of the SMPL overlay (pipeline/overlay/coach_pose.py): BigVegas rotations -> SMPL rotations.

A synthetic skeleton with the BigVegas topology (the `*_virtual` joints included) is posed with random
rotations; after the conversion, SMPL forward kinematics must point every bone the way BigVegas FK does.
numpy + scipy only: no SMPL model file, no BVH file.

Run: python tests/test_overlay_convert.py   (from anywhere)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from pipeline.bvh_to_npz import SMPL_FROM_BIGVEGAS  # noqa: E402
from pipeline.overlay import coach_pose as C  # noqa: E402
from pipeline.overlay.smpl_np import SMPL22, aa_to_mat  # noqa: E402

# BigVegas hierarchy as PAN writes it (coach_on_user_pan.bvh), offsets rounded, cm
BV = [  # name, parent, offset
    ('Pelvis', None, (0, 0, 0)),
    ('LeftUpLeg_virtual', 'Pelvis', (0, 0, 0)), ('LeftUpLeg', 'LeftUpLeg_virtual', (6.6, -2.0, -1.6)),
    ('LeftLeg', 'LeftUpLeg', (0, -39.2, 0.6)), ('LeftFoot', 'LeftLeg', (0, -40.8, -5.5)),
    ('LeftToeBase', 'LeftFoot', (0, -6.5, 12.0)),
    ('RightUpLeg_virtual', 'Pelvis', (0, 0, 0)), ('RightUpLeg', 'RightUpLeg_virtual', (-6.6, -2.0, -1.6)),
    ('RightLeg', 'RightUpLeg', (0, -39.2, 0.6)), ('RightFoot', 'RightLeg', (0, -40.8, -5.5)),
    ('RightToeBase', 'RightFoot', (0, -6.5, 12.0)),
    ('Hips_virtual', 'Pelvis', (0, 0, 0)), ('Hips', 'Hips_virtual', (0, 3.0, 0.5)),
    ('Spine', 'Hips', (0, 9.0, 1.0)), ('Spine1', 'Spine', (0, 11.0, 0.5)), ('Spine2', 'Spine1', (0, 12.0, -0.5)),
    ('Neck_virtual', 'Spine2', (0, 0, 0)), ('Neck', 'Neck_virtual', (0, 14.0, 1.5)), ('Head', 'Neck', (0, 9.0, 2.0)),
    ('LeftShoulder_virtual', 'Spine2', (0, 0, 0)), ('LeftShoulder', 'LeftShoulder_virtual', (5.0, 10.0, 0.5)),
    ('LeftArm', 'LeftShoulder', (11.0, -1.0, -1.0)), ('LeftForeArm', 'LeftArm', (26.0, 0, 0)),
    ('LeftHand', 'LeftForeArm', (25.0, 0, 0)),
    ('RightShoulder_virtual', 'Spine2', (0, 0, 0)), ('RightShoulder', 'RightShoulder_virtual', (-5.0, 10.0, 0.5)),
    ('RightArm', 'RightShoulder', (-11.0, -1.0, -1.0)), ('RightForeArm', 'RightArm', (-26.0, 0, 0)),
    ('RightHand', 'RightForeArm', (-25.0, 0, 0)),
]
NAMES = [b[0] for b in BV]
PARENTS = np.array([-1 if p is None else NAMES.index(p) for _, p, _ in BV])
OFFSETS = np.array([o for _, _, o in BV], float)
SMPL_FROM_BV = {SMPL22.index(s): b for s, b in SMPL_FROM_BIGVEGAS.items()}
BONES = [(C.SMPL_PARENTS[c], c) for c in range(1, 22) if C.SMPL_PARENTS[c] in SMPL_FROM_BV]   # mapped bones


def bv_fk(R_local, root=np.zeros(3)):
    """Same composition as outer_utils.Animation.transforms_global (local translation = offset)."""
    T = len(R_local)
    G, P = np.zeros((T, len(NAMES), 3, 3)), np.zeros((T, len(NAMES), 3))
    for j, p in enumerate(PARENTS):
        if p < 0:
            G[:, j], P[:, j] = R_local[:, j], root
        else:
            G[:, j] = G[:, p] @ R_local[:, j]
            P[:, j] = P[:, p] + (G[:, p] @ OFFSETS[j][:, None])[..., 0]
    return G, P


def smpl_rest_like(rng, scale=(0.8, 1.25), noise_deg=0.):
    """SMPL rest joints built bone by bone from the BigVegas rest pose: other lengths, and the same directions
    unless noise_deg turns each bone by a random small rotation."""
    _, P = bv_fk(np.broadcast_to(np.eye(3), (1, len(NAMES), 3, 3)))
    rest = np.zeros((24, 3))
    for c in range(1, 24):
        p = C.SMPL_PARENTS[c]
        if c in SMPL_FROM_BV:
            d = P[0, NAMES.index(SMPL_FROM_BV[c])] - P[0, NAMES.index(SMPL_FROM_BV[p])]
        else:
            d = np.array([8., 0, 0]) * (1 if c % 2 == 0 else -1)        # hands: no BigVegas joint
        if noise_deg:
            d = Rotation.from_rotvec(rng.normal(size=3) * np.radians(noise_deg) / np.sqrt(3)).apply(d)
        rest[c] = rest[p] + d * rng.uniform(*scale)
    return rest


def smpl_bone_dirs(G, rest):
    """Posed SMPL bone directions from global rotations: bone p->c = G_p (rest_c - rest_p)."""
    return {(p, c): C.unit((G[:, p] @ (rest[c] - rest[p])[:, None])[..., 0]) for p, c in BONES}


def bv_bone_dirs(P):
    i = lambda j: NAMES.index(SMPL_FROM_BV[j])
    return {(p, c): C.unit(P[:, i(c)] - P[:, i(p)]) for p, c in BONES}


def angle_deg(a, b):
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def random_pose(rng, T=50, deg=60):
    R = Rotation.from_rotvec(rng.normal(size=(T * len(NAMES), 3)) * np.radians(deg) / np.sqrt(3))
    return R.as_matrix().reshape(T, len(NAMES), 3, 3)


def convert(G, P, rest):
    return C.bigvegas_to_smpl(G, P, OFFSETS, PARENTS, NAMES, rest, SMPL_FROM_BV)


VIRTUAL = [j for j, n in enumerate(NAMES) if n.endswith('_virtual')]
UNMAPPED = [NAMES.index(n) for n in NAMES if n not in SMPL_FROM_BIGVEGAS.values()]   # virtual + Hips


def test_same_rest_directions_every_bone_exact():
    """SMPL rest bones parallel to BigVegas' (other lengths) and the `*_virtual` joints unrotated (the bones
    out of the pelvis and out of Spine2 move as one, as SMPL's single rotation can): every posed bone
    matches, the pelvis and spine3 (several children, fitted rotation) included. (Hips, between the pelvis and
    Spine, has no SMPL joint either and is held still too.) With the virtual joints
    turned on their own (PAN does), no single rotation can follow all three bones; compare.json measures how
    far apart they are on real output (fit_residual_deg)."""
    rng = np.random.default_rng(0)
    R = random_pose(rng)
    R[:, UNMAPPED] = np.eye(3)
    G_bv, P_bv = bv_fk(R)
    rest = smpl_rest_like(rng)
    G = convert(G_bv, P_bv, rest)
    a, b = smpl_bone_dirs(G, rest), bv_bone_dirs(P_bv)
    worst = max(angle_deg(a[k], b[k]).max() for k in BONES)
    assert worst < 1e-4, worst


def test_other_rest_directions_single_child_bones_exact():
    """SMPL rest bones turned by ~10 deg against BigVegas' and the virtual joints turned on their own: bones
    out of a one-child joint still match exactly (R_rest takes the rest difference out)."""
    rng = np.random.default_rng(1)
    G_bv, P_bv = bv_fk(random_pose(rng))
    rest = smpl_rest_like(rng, noise_deg=10.)
    G = convert(G_bv, P_bv, rest)
    a, b = smpl_bone_dirs(G, rest), bv_bone_dirs(P_bv)
    multi = {j for j in range(24) if sum(C.SMPL_PARENTS[c] == j and c in SMPL_FROM_BV for c in range(24)) > 1}
    single = [k for k in BONES if k[0] not in multi]
    assert len(single) == 15
    assert max(angle_deg(a[k], b[k]).max() for k in single) < 1e-4


def test_twist_kept_from_bigvegas():
    """Turning a BigVegas joint about its own bone turns the SMPL joint the same way (the twist survives)."""
    rng = np.random.default_rng(2)
    R = random_pose(rng, T=2)
    R[1] = R[0]
    j = NAMES.index('LeftArm')
    axis = OFFSETS[NAMES.index('LeftForeArm')] / np.linalg.norm(OFFSETS[NAMES.index('LeftForeArm')])
    R[1, j] = R[0, j] @ Rotation.from_rotvec(axis * 0.7).as_matrix()      # 0.7 rad about the upper arm
    G_bv, P_bv = bv_fk(R)
    rest = smpl_rest_like(rng, noise_deg=5.)
    G = convert(G_bv, P_bv, rest)
    s = SMPL22.index('left_shoulder')
    d = Rotation.from_matrix(G[1, s] @ G[0, s].T).magnitude()
    assert abs(d - 0.7) < 1e-6, d


def test_local_rotations_rebuild_globals():
    """globals_to_pose: the local rotations chained down the SMPL tree give the global rotations back
    (up to the heading taken out of the root)."""
    rng = np.random.default_rng(3)
    G_bv, P_bv = bv_fk(random_pose(rng))
    G = convert(G_bv, P_bv, smpl_rest_like(rng))
    pose = C.globals_to_pose(G)
    L = aa_to_mat(pose['body_pose'].reshape(-1, 23, 3))
    G2 = np.zeros_like(G)
    G2[:, 0] = pose['tilt']
    for j in range(1, 24):
        G2[:, j] = G2[:, C.SMPL_PARENTS[j]] @ L[:, j - 1]
    yaw = G[:, 0] @ np.swapaxes(pose['tilt'], -1, -2)                       # the heading taken out
    assert np.abs(yaw[:, None] @ G2 - G).max() < 1e-6


def test_mirror_twice_is_identity():
    rng = np.random.default_rng(4)
    G_bv, P_bv = bv_fk(random_pose(rng))
    pose = C.globals_to_pose(convert(G_bv, P_bv, smpl_rest_like(rng)))
    back = C.mirror(C.mirror(pose))
    assert np.abs(back['tilt'] - pose['tilt']).max() < 1e-12
    assert np.abs(back['body_pose'] - pose['body_pose']).max() < 1e-12


def random_rotations(rng, n, deg):
    return Rotation.from_rotvec(rng.normal(size=(n, 3)) * np.radians(deg) / np.sqrt(3)).as_matrix()


def test_calibration_finds_a_fixed_offset():
    """A target made from BigVegas with a fixed rotation per joint (how the two rigs differ): calibrate gets
    those rotations back, and converting with them rebuilds the target."""
    rng = np.random.default_rng(5)
    G_bv, P_bv = bv_fk(random_pose(rng))
    K_true = random_rotations(rng, 24, 40)
    args = (G_bv, P_bv, OFFSETS, PARENTS, NAMES, SMPL_FROM_BV)
    target = C._convert(*args, lambda j, kind, D, info: K_true[j])
    K, spread = C.calibrate(*args, target)
    assert np.abs(K - K_true).max() < 1e-9
    assert max(spread.values()) < 1e-4
    G = C.bigvegas_to_smpl(G_bv, P_bv, OFFSETS, PARENTS, NAMES, None, SMPL_FROM_BV, calib=K)
    assert np.abs(G - target).max() < 1e-9


def test_calibration_noise_averages_out():
    """Each target frame turned by its own small random rotation (what IK / PAN do differently frame to frame):
    K stays near the fixed part, and the spread reports the noise."""
    rng = np.random.default_rng(6)
    G_bv, P_bv = bv_fk(random_pose(rng, T=400))
    K_true = random_rotations(rng, 24, 40)
    args = (G_bv, P_bv, OFFSETS, PARENTS, NAMES, SMPL_FROM_BV)
    target = C._convert(*args, lambda j, kind, D, info: K_true[j])
    target = target @ random_rotations(rng, 400 * 24, 5).reshape(400, 24, 3, 3)
    target[:10] = np.nan                                                     # frames without a target
    K, spread = C.calibrate(*args, target)
    err = [Rotation.from_matrix(K[j] @ K_true[j].T).magnitude() for j in range(24)]
    assert np.degrees(max(err)) < 2., np.degrees(max(err))
    assert all(1. < s < 10. for s in spread.values()), spread


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
