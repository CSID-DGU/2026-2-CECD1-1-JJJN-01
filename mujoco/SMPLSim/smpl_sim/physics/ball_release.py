"""Kinematic grip and one-way release from actual simulated wrist motion.

Freejoint translation velocity is world-frame; angular velocity is ball-local.
No actuator, fitted motion, original XML or existing controller is changed.
"""
import numpy as np
import mujoco
from smpl_sim.physics.basketball import HumanoidControllerBridge


class ReleaseControllerBridge(HumanoidControllerBridge):
    """Original PD + COM; optional, explicitly test-only wrist target perturbation."""
    def __init__(self, humanoid, model, motion, hand='right', wrist_amplitude=0.):
        super().__init__(humanoid, model, motion)
        self.wrist_amplitude = wrist_amplitude
        self.wrist_qpos = int(self.model.joint(('R' if hand == 'right' else 'L')+'_Wrist_x').qposadr[0])

    def target(self, t):
        q = super().target(t)
        # Smooth temporary goal, never mutate bound motion or NPZ.
        if getattr(self, 'wrist_amplitude', 0.):
            x = np.clip((t-1.5)/1., 0., 1.)
            q[self.wrist_qpos] += self.wrist_amplitude * (6*x**5-15*x**4+10*x**3)
        return q


def hand_ball_state(model, data, hand, offset):
    """Position, quaternion(wxyz), world v, world omega at hand-local offset."""
    if hand not in ('right', 'left'): raise ValueError('Invalid hand')
    bid = model.body(('R' if hand == 'right' else 'L')+'_Wrist').id
    rotation = data.xmat[bid].reshape(3, 3)
    position = data.xpos[bid] + rotation @ np.asarray(offset)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jp, jr, position, bid)
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, rotation.ravel())
    return position, quaternion, jp @ data.qvel, jr @ data.qvel


class HandBallRelease:
    """Hold by state overwrite; after release update() cannot write ball state.

    Offset is in wrist body axes, metres (not world axes or anatomical angles).
    Default sits off the palm surface; left offset mirrors the local x axis.
    Ball mass does not load the hand during the kinematic hold.
    """
    def __init__(self, model, hand='right', offset=None, release_time=2.5, hand_collision='shooting'):
        if hand not in ('right', 'left'): raise ValueError('hand must be right or left')
        if hand_collision not in ('none', 'shooting', 'both'): raise ValueError('Invalid hand collision mode')
        if not np.isfinite(release_time) or release_time < 0: raise ValueError('Invalid release time')
        self.model, self.hand, self.release_time = model, hand, float(release_time)
        self.offset = np.asarray(offset if offset is not None else [-.07 if hand == 'right' else .07, -.19, 0.], dtype=float)
        if self.offset.shape != (3,) or not np.isfinite(self.offset).all(): raise ValueError('Expected finite xyz offset')
        joint = model.joint('basketball_free')
        self.qa, self.da = int(joint.qposadr[0]), int(joint.dofadr[0])
        self.geom = model.geom('basketball_geom').id
        self.released, self.event = False, None
        self.release_masks = (8, 16 | (32 if hand_collision != 'none' else 0))
        if hand_collision != 'none':
            sides = ('L', 'R') if hand_collision == 'both' else (('R' if hand == 'right' else 'L'),)
            for i in range(model.ngeom):
                name = model.body(int(model.geom_bodyid[i])).name
                if any(name == s+'_Wrist' or (name.startswith(s+'_') and any(f in name for f in ('Index','Middle','Ring','Pinky','Thumb'))) for s in sides):
                    model.geom_contype[i] |= 32
                    model.body_contype[int(model.geom_bodyid[i])] |= 32
        model.geom_contype[self.geom] = 0
        model.geom_conaffinity[self.geom] = 0
        model.body_contype[int(model.geom_bodyid[self.geom])] = 0
        model.body_conaffinity[int(model.geom_bodyid[self.geom])] = 0

    def update(self, data):
        if self.released: return False
        p, q, v, w = hand_ball_state(self.model, data, self.hand, self.offset)
        rotation = data.xmat[self.model.body(('R' if self.hand == 'right' else 'L')+'_Wrist').id].reshape(3, 3)
        data.qpos[self.qa:self.qa+7] = np.r_[p, q]
        data.qvel[self.da:self.da+6] = np.r_[v, rotation.T @ w]
        if data.time + 1e-10 >= self.release_time:
            before_p, before_v = data.qpos[self.qa:self.qa+7].copy(), data.qvel[self.da:self.da+6].copy()
            self.released = True
            self.model.geom_contype[self.geom], self.model.geom_conaffinity[self.geom] = self.release_masks
            bid = int(self.model.geom_bodyid[self.geom])
            self.model.body_contype[bid], self.model.body_conaffinity[bid] = self.release_masks
            self.event = {'requested_time_s': self.release_time, 'actual_time_s': float(data.time),
                          'position_m': p.tolist(), 'velocity_world_m_s': v.tolist(), 'angular_velocity_world_rad_s': w.tolist(),
                          'position_jump_m': float(np.linalg.norm(data.qpos[self.qa:self.qa+3]-before_p[:3])),
                          'velocity_jump_m_s': float(np.linalg.norm(data.qvel[self.da:self.da+3]-before_v[:3])),
                          'angular_velocity_jump_rad_s': float(np.linalg.norm(data.qvel[self.da+3:self.da+6]-before_v[3:]))}
        mujoco.mj_forward(self.model, data)
        return True
