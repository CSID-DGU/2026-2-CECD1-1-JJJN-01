"""Temporary synthetic motion for pipeline validation, not basketball ground truth."""
import numpy as np
from scipy.spatial.transform import Rotation
from smpl_sim.smpllib.smpl_joint_names import SMPLH_BONE_ORDER_NAMES as NAMES
from .smplh_motion import SMPLHMotion

FPS = 30
DURATION = 3.0
STAGES = (
    ('stand', 0., .35),
    ('knee_bend', .35, .85),
    ('knee_extend', .85, 1.30),
    ('arm_raise', 1.30, 1.95),
    ('elbow_extend', 1.95, 2.50),
    ('wrist_flick', 2.50, 3.),
)
TRACKED_JOINTS = [f'{side}_{part}_{axis}' for side in ('L', 'R')
                  for part, axis in [('Hip', 'x'), ('Knee', 'x'), ('Shoulder', 'x'),
                                     ('Elbow', 'y'), ('Wrist', 'x')]]


def stage_at(t):
    """Right-open phase boundaries; final endpoint belongs to follow-through."""
    for name, _, end in STAGES:
        if t < end - 1e-9:
            return name
    return STAGES[-1][0]


def transition(t, start, end):
    # Quintic smoothstep: zero velocity AND acceleration at phase boundaries.
    u = np.clip((np.asarray(t) - start) / (end - start), 0., 1.)
    return u**3 * (10. - 15.*u + 6.*u**2)


def generate(root_position):
    """One source motion, with root_position expressed in SMPL Y-up meters."""
    root_position = np.asarray(root_position, dtype=float)
    if root_position.shape != (3,) or not np.isfinite(root_position).all():
        raise ValueError('root_position must be finite (3,)')
    times = np.arange(round(DURATION * FPS) + 1) / FPS
    euler = np.zeros((len(times), 52, 3))
    bend = transition(times, .35, .85) * (1. - transition(times, .85, 1.30))
    lift = transition(times, 1.30, 1.95)
    extend = transition(times, 1.95, 2.50)
    flick = transition(times, 2.50, 2.75)
    recover = transition(times, 2.75, 3.)
    for side, sign in [('L', -1.), ('R', 1.)]:
        def set_angle(part, axis, angle):
            euler[:, NAMES.index(side + '_' + part), axis] = angle
        set_angle('Hip', 0, -.05 - .125*bend)
        set_angle('Knee', 0, .10 + .25*bend)
        set_angle('Ankle', 0, -.05 - .125*bend)
        set_angle('Shoulder', 2, sign*1.35)
        set_angle('Shoulder', 0, -1.15*lift - .15*extend)
        # Preparation flexion during lifting, then extension toward 0.10 rad.
        set_angle('Elbow', 1, sign*(.90*lift - .80*extend))
        set_angle('Wrist', 0, -.30*extend + .50*flick - .10*recover)
    poses = Rotation.from_euler('XYZ', euler.reshape(-1, 3)).as_rotvec().reshape(-1, 156)
    return SMPLHMotion(poses, np.tile(root_position, (len(times), 1)), FPS,
                      coordinate_system='smpl', translation_kind='root_position')
