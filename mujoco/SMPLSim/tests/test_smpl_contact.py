"""Contact scene/goal adapter invariants; do not interpret them as Holding success."""
import tempfile
import unittest
from pathlib import Path
import mujoco
import numpy as np
from examples.m4_smpl_contact import build_scene,ContactTargets,initialize_ball,is_finger
from smpl_sim.physics.basketball import HumanoidControllerBridge

ROOT=Path(__file__).resolve().parents[1]
MODELS=ROOT.parents[2]/'basketball-r1/SMPLSim/outputs/smplh_physics'
MOTION=ROOT/'outputs/m3_spl_fitting/spl_P0002_T0049.npz'

@unittest.skipUnless((MODELS/'body_01.xml').exists() and MOTION.exists(),'Local models required')
class SMPLContactTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory()
    def tearDown(self):self.temp.cleanup()
    def fixture(self,body='body_01',pad=False):
        original=MODELS/(body+'.xml');m=mujoco.MjModel.from_xml_path(str(build_scene(original,Path(self.temp.name)/(body+'.xml'),pad)))
        b=HumanoidControllerBridge(original,m,MOTION);d=mujoco.MjData(m);b.initialize(m,d)
        a=ContactTargets(b,'hold');d.qpos[:b.model.nq]=a.target(0);mujoco.mj_forward(m,d)
        q,v=initialize_ball(m,d,-.10)
        return m,d,b,a,q,v
    def test_original_human_and_m4_ball_properties_all_shapes(self):
        for body in ['body_01','body_02','body_03','body_04']:
            for pad in [False,True]:
                m,d,b,a,q,v=self.fixture(body,pad)
                self.assertEqual(m.nu,153);self.assertEqual(m.neq,0)
                np.testing.assert_allclose(m.body_mass[:b.model.nbody],b.model.body_mass)
                np.testing.assert_allclose(m.body_inertia[:b.model.nbody],b.model.body_inertia)
                self.assertAlmostEqual(m.body('basketball').mass[0],.6)
                np.testing.assert_allclose(m.body('basketball').inertia,[2/3*.6*.121**2]*3)
    def test_masks_enable_only_right_hand_and_environment(self):
        m,d,b,a,q,v=self.fixture();ball=m.geom('basketball_geom').id
        for name in ['R_Wrist','R_Index1','R_Thumb3','floor','backboard','rim_00']:
            g=m.geom(name).id
            self.assertTrue((m.geom_contype[g]&m.geom_conaffinity[ball]) or (m.geom_contype[ball]&m.geom_conaffinity[g]))
        for name in ['Pelvis','L_Wrist']:
            g=m.geom(name).id
            self.assertFalse((m.geom_contype[g]&m.geom_conaffinity[ball]) or (m.geom_contype[ball]&m.geom_conaffinity[g]))
    def test_palm_up_and_finger_pd_ablation(self):
        m,d,b,a,q,v=self.fixture()
        self.assertAlmostEqual(-d.xmat[m.body('R_Wrist').id].reshape(3,3)[2,1],1.,places=10)
        a.control(d,a.target(0),True)
        for i in range(m.nu):
            if is_finger(m.actuator(i).name):self.assertEqual(d.ctrl[i],0.)
    def test_controller_does_not_overwrite_ball_state(self):
        m,d,b,a,q,v=self.fixture()
        for _ in range(30):
            before=d.qpos[q:q+7].copy();beforev=d.qvel[v:v+6].copy()
            a.control(d,a.target(d.time,d))
            np.testing.assert_array_equal(d.qpos[q:q+7],before);np.testing.assert_array_equal(d.qvel[v:v+6],beforev)
            self.assertFalse(d.qfrc_applied.any());self.assertFalse(d.xfrc_applied.any())
            mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
    def test_hand_pair_damping_preserves_environment_ball_model(self):
        source=MODELS/'body_01.xml'
        a=mujoco.MjModel.from_xml_path(str(build_scene(source,Path(self.temp.name)/'base.xml')))
        b=mujoco.MjModel.from_xml_path(str(build_scene(source,Path(self.temp.name)/'damped.xml',soft_hand_contact=True)))
        self.assertEqual(b.npair,16);self.assertEqual(b.neq,0)
        for name in ['floor','basketball_geom','backboard','rim_00']:
            ga,gb=a.geom(name).id,b.geom(name).id
            for field in ['geom_size','geom_friction','geom_solref','geom_solimp']:
                np.testing.assert_array_equal(getattr(a,field)[ga],getattr(b,field)[gb])

if __name__=='__main__':unittest.main()
