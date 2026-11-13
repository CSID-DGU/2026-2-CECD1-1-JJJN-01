"""MediaPipe Pose (33 landmarks) <-> Mixamo group-A joints.

MediaPipe world landmarks: metres, origin at the hip centre, x right (image), y down,
z away from the camera. Mixamo: centimetres, Y up, character faces +Z, its left is +X.
(x, -y, -z) keeps the frame right-handed.

Only joints with a direct anatomical counterpart are observed; Pelvis, Hips, spine,
neck, head and clavicles are left to the IK (constraints + priors).
"""
import numpy as np

LANDMARKS = dict(nose=0, left_ear=7, right_ear=8, left_shoulder=11, right_shoulder=12,
                 left_elbow=13, right_elbow=14, left_wrist=15, right_wrist=16,
                 left_hip=23, right_hip=24, left_knee=25, right_knee=26,
                 left_ankle=27, right_ankle=28, left_heel=29, right_heel=30,
                 left_foot_index=31, right_foot_index=32)

# Mixamo joint -> (landmark, weight). foot_index is the toe tip, Mixamo ToeBase is the
# ball of the foot (and BigVegas' rest foot points steeply down): kept with a low weight
# so the foot pitch is mostly left to the data prior instead of an invented correction.
TO_MIXAMO = {
    'LeftUpLeg': ('left_hip', 1.), 'RightUpLeg': ('right_hip', 1.),
    'LeftLeg': ('left_knee', 1.), 'RightLeg': ('right_knee', 1.),
    'LeftFoot': ('left_ankle', 1.), 'RightFoot': ('right_ankle', 1.),
    'LeftToeBase': ('left_foot_index', .25), 'RightToeBase': ('right_foot_index', .25),
    'LeftArm': ('left_shoulder', 1.), 'RightArm': ('right_shoulder', 1.),
    'LeftForeArm': ('left_elbow', 1.), 'RightForeArm': ('right_elbow', 1.),
    'LeftHand': ('left_wrist', 1.), 'RightHand': ('right_wrist', 1.),
}
AXES = np.array([1., -1., -1.])
# Observed, but not at the Mixamo joint: do not use for bone-length estimation.
UNRELIABLE_LENGTH = ('LeftToeBase', 'RightToeBase')


def mp_to_yup_cm(world):
    return np.asarray(world, dtype=np.float64) * AXES * 100.


def yup_cm_to_mp(pos):
    return np.asarray(pos, dtype=np.float64) / 100. * AXES


def landmarks_to_joints(world, visibility, skel, min_visibility=0.05):
    """(T, 33, 3) MediaPipe world, (T, 33) visibility -> obs (T, J, 3) cm Y-up, weight (T, J).

    Weight = mapping weight x visibility. Low-visibility (occluded) landmarks keep
    MediaPipe's own estimate with a small weight rather than being dropped: dropping them
    made gap filling copy poses from distant frames. Missing frames get weight 0.
    The root slot holds the hip centre as an initial guess.
    """
    world = mp_to_yup_cm(world)
    T = world.shape[0]
    obs = np.zeros((T, skel.n_joints, 3))
    weight = np.zeros((T, skel.n_joints))
    for joint, (lm, w) in TO_MIXAMO.items():
        j, k = skel.index(joint), LANDMARKS[lm]
        ok = np.isfinite(world[:, k]).all(-1) & (visibility[:, k] >= min_visibility)
        obs[ok, j] = world[ok, k]
        weight[ok, j] = w * visibility[ok, k]
    hips = world[:, [LANDMARKS['left_hip'], LANDMARKS['right_hip']]].mean(1)
    obs[:, 0] = np.nan_to_num(hips)
    return obs, weight
