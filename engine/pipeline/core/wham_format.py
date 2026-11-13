"""WHAM / SMPL (24 joints) <-> Mixamo group-A joints.

WHAM world joints: metres, Y up (return_y_up=True), constant body shape for the track.
Unlike MediaPipe, SMPL has pelvis, three spine joints, neck, head and clavicles, and its
`foot` joint sits near the ball of the foot like Mixamo's ToeBase. Joints whose definition
differs noticeably between the two rigs (pelvis height, spine split, head centre vs skull
base, clavicle start) get weight 0.5; their bone lengths are still measured.
"""
import numpy as np

SMPL_NAMES = ('pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2',
              'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck',
              'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
              'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand')
SMPL_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)
SMPL_EDGES = tuple((p, j) for j, p in enumerate(SMPL_PARENTS) if p >= 0)

TO_MIXAMO = {
    'Pelvis': ('pelvis', .5),
    'LeftUpLeg': ('left_hip', 1.), 'RightUpLeg': ('right_hip', 1.),
    'LeftLeg': ('left_knee', 1.), 'RightLeg': ('right_knee', 1.),
    'LeftFoot': ('left_ankle', 1.), 'RightFoot': ('right_ankle', 1.),
    'LeftToeBase': ('left_foot', 1.), 'RightToeBase': ('right_foot', 1.),
    'Spine': ('spine1', .5), 'Spine1': ('spine2', .5), 'Spine2': ('spine3', .5),
    'Neck': ('neck', .5), 'Head': ('head', .5),
    'LeftShoulder': ('left_collar', .5), 'RightShoulder': ('right_collar', .5),
    'LeftArm': ('left_shoulder', 1.), 'RightArm': ('right_shoulder', 1.),
    'LeftForeArm': ('left_elbow', 1.), 'RightForeArm': ('right_elbow', 1.),
    'LeftHand': ('left_wrist', 1.), 'RightHand': ('right_wrist', 1.),
}
# Pelvis -> hip offsets differ in definition (Mixamo pelvis sits higher); keep the hip
# width measurement instead of the pelvis-to-hip bone length.
UNRELIABLE_LENGTH = ('LeftUpLeg', 'RightUpLeg')


# Upper-torso definitions differ between the rigs: SMPL clavicles are near-horizontal and
# its shoulder joints sit lower relative to spine3/neck, while BigVegas' rest clavicles point
# ~35 deg up/out with the arm joints near neck height. Fitting clavicles, or chest + neck
# together with the arms, forces clavicle rotations ~50-60 deg off PAN's training mean
# (measured on the squat videos: PAN-input |z|>4 4.5-5.9 %). Skipping clavicles, Spine2 and
# Neck lets the spine absorb the difference (0.7-2.2 %); shoulder width is then measured
# from the arm joints.
DEFAULT_SKIP = ('LeftShoulder', 'RightShoulder', 'Spine2', 'Neck')


def joints_to_obs(joints, detected, skel, skip=DEFAULT_SKIP):
    """(T, 24, 3) metres -> obs (T, J, 3) cm Y-up, weight (T, J); missing frames weight 0."""
    J = np.nan_to_num(np.asarray(joints, dtype=np.float64)) * 100.
    T = len(J)
    obs = np.zeros((T, skel.n_joints, 3))
    weight = np.zeros((T, skel.n_joints))
    det = np.asarray(detected, bool)
    for joint, (name, w) in TO_MIXAMO.items():
        if joint in skip:
            continue
        j, k = skel.index(joint), SMPL_NAMES.index(name)
        obs[:, j] = J[:, k]
        weight[det, j] = w
    return obs, weight
