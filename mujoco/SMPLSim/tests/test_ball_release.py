"""Physical release and independent kinematic derivative checks."""
import unittest
from pathlib import Path
import numpy as np
import mujoco
from smpl_sim.physics.ball_release import HandBallRelease, hand_ball_state, ReleaseControllerBridge

ROOT=Path(__file__).resolve().parents[1]
SCENE=ROOT/'outputs/m4_basketball/body_01_basketball.xml'
HUMANOID=ROOT.parents[2]/'basketball-r1/SMPLSim/outputs/smplh_physics/body_01.xml'
MOTION=ROOT/'outputs/m3_spl_fitting/spl_P0002_T0049.npz'

@unittest.skipUnless(SCENE.exists() and MOTION.exists() and HUMANOID.exists(),'Local assets required')
class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.m=mujoco.MjModel.from_xml_path(str(SCENE));self.d=mujoco.MjData(self.m)
        self.b=ReleaseControllerBridge(HUMANOID,self.m,MOTION);self.b.initialize(self.m,self.d)

    def test_point_velocity_and_spin_match_finite_difference_both_hands(self):
        rng=np.random.default_rng(32);self.d.qvel[:159]=rng.normal(0,.3,159)
        mujoco.mj_forward(self.m,self.d)
        old=self.d.qpos.copy();eps=1e-6
        for hand in ('right','left'):
            p,q,v,w=hand_ball_state(self.m,self.d,hand,[.06,-.16,.03])
            mujoco.mj_integratePos(self.m,self.d.qpos,self.d.qvel,eps);mujoco.mj_forward(self.m,self.d)
            p2,q2,_,_=hand_ball_state(self.m,self.d,hand,[.06,-.16,.03])
            np.testing.assert_allclose((p2-p)/eps,v,atol=2e-6)
            from scipy.spatial.transform import Rotation
            r1=Rotation.from_quat(q[[1,2,3,0]]);r2=Rotation.from_quat(q2[[1,2,3,0]])
            np.testing.assert_allclose((r2*r1.inv()).as_rotvec()/eps,w,atol=2e-6)
            self.d.qpos[:]=old;mujoco.mj_forward(self.m,self.d)

    def test_release_never_writes_state_after_handoff_and_gravity_is_physical(self):
        self.d.qvel[:159]=np.linspace(-.2,.2,159);mujoco.mj_forward(self.m,self.d)
        grip=HandBallRelease(self.m,offset=[0,-.3,0],release_time=0,hand_collision='none')
        grip.update(self.d);p=self.d.qpos[160:163].copy();v=self.d.qvel[159:162].copy()
        # Independent angular-velocity measurement verifies local freejoint storage.
        velocity=np.empty(6);mujoco.mj_objectVelocity(self.m,self.d,mujoco.mjtObj.mjOBJ_BODY,self.m.body('basketball').id,velocity,0)
        np.testing.assert_allclose(velocity[:3],grip.event['angular_velocity_world_rad_s'],atol=1e-12)
        np.testing.assert_allclose(velocity[3:],v,atol=1e-12)
        for _ in range(20):
            mujoco.mj_step(self.m,self.d);mujoco.mj_forward(self.m,self.d)
            state=self.d.qpos.copy();speed=self.d.qvel.copy();self.assertFalse(grip.update(self.d))
            np.testing.assert_array_equal(self.d.qpos,state);np.testing.assert_array_equal(self.d.qvel,speed)
        t=self.d.time;np.testing.assert_allclose(self.d.qvel[159:162],v+[0,0,-9.81*t],atol=1e-10)
        expected=p+v*t+[0,0,-.5*9.81*t*t]
        self.assertLess(np.linalg.norm(self.d.qpos[160:163]-expected),.001)

    def test_held_ball_has_no_contact_or_human_state_change(self):
        grip=HandBallRelease(self.m,offset=[0,0,0],release_time=10,hand_collision='both')
        # Intentional hand overlap cannot transfer forces before release.
        before=self.d.qpos[:160].copy();grip.update(self.d)
        np.testing.assert_array_equal(self.d.qpos[:160],before)
        for c in self.d.contact:self.assertNotIn(grip.geom,(c.geom1,c.geom2))
        self.assertFalse(grip.released)

    def test_offset_validation_left_selection_and_test_motion_is_separate(self):
        grip=HandBallRelease(self.m,hand='left',release_time=1)
        self.assertGreater(grip.offset[0],0);grip.update(self.d)
        np.testing.assert_allclose(self.d.qpos[160:163],hand_ball_state(self.m,self.d,'left',grip.offset)[0])
        with self.assertRaises(ValueError):HandBallRelease(self.m,offset=[0,np.nan,0])
        with self.assertRaises(ValueError):HandBallRelease(self.m,release_time=-1)
        original=self.b.bound.sample(2.3).copy()
        variant=ReleaseControllerBridge(HUMANOID,self.m,MOTION,wrist_amplitude=.3)
        self.assertGreater(abs(variant.target(2.3)[variant.wrist_qpos]-self.b.target(2.3)[variant.wrist_qpos]),.2)
        np.testing.assert_array_equal(self.b.bound.sample(2.3),original)

    def test_released_ball_can_contact_hand_broad_and_narrow_phase(self):
        grip=HandBallRelease(self.m,offset=[0,0,0],release_time=0,hand_collision='shooting')
        grip.update(self.d)
        handcontacts=[]
        for c in self.d.contact:
            if grip.geom in (c.geom1,c.geom2):
                other=int(c.geom2 if c.geom1==grip.geom else c.geom1)
                if self.m.geom_bodyid[other]>0:handcontacts.append(c)
        self.assertGreater(len(handcontacts),0)
