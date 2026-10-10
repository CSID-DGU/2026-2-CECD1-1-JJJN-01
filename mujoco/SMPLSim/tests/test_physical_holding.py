"""Physical equality load, activation alignment and velocity-preserving release."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import mujoco
from smpl_sim.physics.basketball import HumanoidControllerBridge
from smpl_sim.physics.physical_holding import build_holding_xml,WeldBallRelease
from smpl_sim.physics.ball_release import hand_ball_state

ROOT=Path(__file__).resolve().parents[1]
MODELS=ROOT.parents[2]/'basketball-r1/SMPLSim/outputs/smplh_physics'
MOTION=ROOT/'outputs/m3_spl_fitting/spl_P0002_T0049.npz'

@unittest.skipUnless((MODELS/'body_01.xml').exists() and MOTION.exists(),'Local assets required')
class PhysicalHoldingTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory()
    def tearDown(self):self.temp.cleanup()
    def fixture(self,body='body_01',hand='right',release=2.5,mass=.6):
        path=build_holding_xml(MODELS/(body+'.xml'),Path(self.temp.name)/(body+hand+str(mass)+'.xml'),hand=hand,test_mass=mass)
        m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m)
        b=HumanoidControllerBridge(MODELS/(body+'.xml'),m,MOTION);b.initialize(m,d)
        grip=WeldBallRelease(m,hand=hand,release_time=release);grip.initialize(d)
        return m,d,b,grip

    def test_all_bodies_and_both_hands_initially_aligned(self):
        for body in ('body_01','body_02','body_03','body_04'):
            for hand in ('right','left'):
                m,d,b,g=self.fixture(body,hand)
                self.assertLess(g.initial_error,1e-12);self.assertLess(g.initial_force,1e-9)
                self.assertEqual(g.state_write_count,1);self.assertTrue(d.eq_active[g.equality])
                np.testing.assert_allclose(d.qpos[g.qa:g.qa+3],hand_ball_state(m,d,hand,g.offset)[0],atol=1e-12)
                self.assertAlmostEqual(m.body('basketball').mass[0],.6)
                self.assertEqual(m.nu,b.model.nu)

    def test_solver_holds_without_state_writes_and_transfers_gravity_load(self):
        m,d,b,g=self.fixture();vertical=[]
        for _ in range(250):
            b.control(d);mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
            q=d.qpos.copy();v=d.qvel.copy();g.update(d)
            np.testing.assert_array_equal(d.qpos,q);np.testing.assert_array_equal(d.qvel,v)
            mask=(d.efc_type==mujoco.mjtConstraint.mjCNSTR_EQUALITY)&(d.efc_id==g.equality)
            load=np.zeros(m.nv);mujoco.mj_mulJacTVec(m,d,load,np.where(mask,d.efc_force,0.))
            if d.time>.3:vertical.append(load[g.da+2])
        self.assertGreater(np.mean(vertical),4.);self.assertLess(np.mean(vertical),8.)
        self.assertLess(np.linalg.norm(d.qpos[g.qa:g.qa+3]-hand_ball_state(m,d,'right',g.offset)[0]),.01)
        self.assertEqual(g.state_write_count,1)

    def test_release_only_disables_constraint_and_preserves_actual_velocity(self):
        m,d,b,g=self.fixture(release=.15)
        while d.time+1e-10<.15:
            b.control(d);mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
        q=d.qpos.copy();v=d.qvel.copy();g.update(d)
        self.assertTrue(g.released);self.assertFalse(d.eq_active[g.equality])
        np.testing.assert_array_equal(d.qpos,q);np.testing.assert_array_equal(d.qvel,v)
        np.testing.assert_array_equal(g.event['velocity_world_m_s'],v[g.da:g.da+3])
        for _ in range(10):
            b.control(d);mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
            q=d.qpos.copy();v=d.qvel.copy();g.update(d)
            np.testing.assert_array_equal(d.qpos,q);np.testing.assert_array_equal(d.qvel,v)
        self.assertEqual(g.state_write_count,1)

    def test_contact_masks_hold_keeps_environment_release_enables_existing_hand(self):
        m,d,b,g=self.fixture(release=.1)
        ball=g.geom;wrist=m.geom('R_Wrist').id
        def allowed(a,c):return bool(m.geom_contype[a]&m.geom_conaffinity[c] or m.geom_contype[c]&m.geom_conaffinity[a])
        self.assertFalse(allowed(ball,wrist))
        for name in ('floor','rim_00','backboard'):self.assertTrue(allowed(ball,m.geom(name).id))
        self.assertGreater(m.geom_size[wrist,0],0)
        d.time=.1;g.update(d);self.assertTrue(allowed(ball,wrist))
        for name in ('floor','rim_00','backboard'):self.assertTrue(allowed(ball,m.geom(name).id))

    def test_mass_changes_weld_gravity_force(self):
        loads=[]
        for mass in (.3,.6):
            m,d,b,g=self.fixture(mass=mass);force=[]
            for _ in range(250):
                b.control(d);mujoco.mj_step(m,d);mujoco.mj_forward(m,d);g.update(d)
                mask=(d.efc_type==mujoco.mjtConstraint.mjCNSTR_EQUALITY)&(d.efc_id==g.equality)
                load=np.zeros(m.nv);mujoco.mj_mulJacTVec(m,d,load,np.where(mask,d.efc_force,0.))
                if d.time>.3:force.append(load[g.da+2])
            loads.append(np.mean(force))
        self.assertGreater(loads[1]/loads[0],1.9);self.assertLess(loads[1]/loads[0],2.1)
