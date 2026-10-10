"""M4-1 structure/filtering, independent ball dynamics, passage and controller parity."""
import tempfile
import unittest
from pathlib import Path
import mujoco
import numpy as np
from smpl_sim.physics.basketball import (build_xml,HumanoidControllerBridge,test_ball_initial_state as set_test_ball_state,
    HoopPassage,RIM_CENTER,BALL_RADIUS,BOARD_FACE_X,BALL_MASS)

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parents[2]
HUMANOID=BASE/'basketball-r1/SMPLSim/outputs/smplh_physics/body_01.xml'
MOTION=ROOT/'outputs/m3_spl_fitting/spl_P0002_T0049.npz'

@unittest.skipUnless(HUMANOID.exists() and MOTION.exists(),'Local humanoid/motion required')
class BasketballTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.path=build_xml(HUMANOID,Path(cls.temp.name)/'basketball.xml')
        cls.model=mujoco.MjModel.from_xml_path(str(cls.path))
        cls.base=mujoco.MjModel.from_xml_path(str(HUMANOID))
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_humanoid_structure_mass_and_actuators_preserved(self):
        m,b=self.model,self.base
        self.assertEqual((m.nq,m.nv,m.nu,m.njnt),(167,165,153,155))
        for j in range(b.njnt):self.assertEqual(m.joint(j).name,b.joint(j).name)
        for attr in ('jnt_type','jnt_axis','jnt_range','jnt_qposadr','jnt_dofadr'):
            np.testing.assert_array_equal(getattr(m,attr)[:b.njnt],getattr(b,attr))
        for attr in ('body_mass','body_inertia','body_pos','body_parentid'):
            np.testing.assert_array_equal(getattr(m,attr)[:b.nbody],getattr(b,attr))
        for attr in ('actuator_trnid','actuator_gear','actuator_ctrlrange','actuator_forcerange'):
            np.testing.assert_array_equal(getattr(m,attr),getattr(b,attr))
        self.assertAlmostEqual(m.body('basketball').mass[0],BALL_MASS)

    def test_masks_disable_ball_human_and_keep_ball_environment(self):
        m=self.model;ball=m.geom('basketball_geom').id
        def allowed(a,b):return bool((m.geom_contype[a]&m.geom_conaffinity[b]) or (m.geom_contype[b]&m.geom_conaffinity[a]))
        for i in range(m.ngeom):
            bid=int(m.geom_bodyid[i])
            if 0<bid<self.base.nbody:self.assertFalse(allowed(ball,i))
        for name in ('floor','backboard','rim_00'):self.assertTrue(allowed(ball,m.geom(name).id))
        bridge=HumanoidControllerBridge(HUMANOID,m,MOTION);d=mujoco.MjData(m);bridge.initialize(m,d)
        # Intentional overlap at wrist still must not produce a ball-human contact.
        set_test_ball_state(m,d,d.xpos[m.body('R_Wrist').id].copy(),[0,0,0])
        for c in d.contact:
            if ball in (c.geom1,c.geom2):
                other=c.geom2 if c.geom1==ball else c.geom1
                self.assertEqual(m.geom_bodyid[other],0)

    def test_test_launcher_sets_independent_position_velocity_and_spin(self):
        m=self.model;d=mujoco.MjData(m);before=d.qpos[:160].copy()
        set_test_ball_state(m,d,[2,.5,3],[1,2,3],[.1,.2,.3])
        np.testing.assert_array_equal(d.qpos[:160],before)
        np.testing.assert_allclose(d.qpos[160:167],[2,.5,3,1,0,0,0])
        np.testing.assert_allclose(d.qvel[159:],[1,2,3,.1,.2,.3])
        with self.assertRaises(ValueError):set_test_ball_state(m,d,[np.nan,0,0],[0,0,0])

    def test_passage_rejects_lateral_and_upward_crossing(self):
        c=RIM_CENTER
        det=HoopPassage();self.assertFalse(det.update(c-[0,0,.2],c+[0,0,.2],1,.01))
        det=HoopPassage();self.assertFalse(det.update(c+[.4,0,.2],c+[.4,0,-.2],1,.01))
        det=HoopPassage();self.assertFalse(det.update(c+[0,0,.2],c-[0,0,.01],1,.01))
        self.assertTrue(det.update(c-[0,0,.01],c-[0,0,.2],1.01,.01))

    def test_oblique_passage_uses_sphere_cross_section_not_full_radius_at_exit(self):
        c=RIM_CENTER;det=HoopPassage()
        det.update(c+[0,0,.1],c+[.02,0,-.02],1,.01)
        self.assertTrue(det.update(c+[.02,0,-.02],c+[.13,0,-.2],1.01,.01))

    def test_four_second_humanoid_parity_with_disconnected_bouncing_ball(self):
        saved=ROOT/'outputs/m3_spl_physics/aligned/physical_states.npz'
        if not saved.exists():self.skipTest('Previous physical reference absent')
        m=self.model;d=mujoco.MjData(m);bridge=HumanoidControllerBridge(HUMANOID,m,MOTION)
        bridge.initialize(m,d);set_test_ball_state(m,d,[2,-1,1.921],[0,0,0])
        with np.load(saved) as states:
            times,qpos=states['time_s'],states['qpos']
            steps=np.rint(times/m.opt.timestep).astype(int);index=0;maximum=0.
            for step in range(int(steps[-1])+1):
                if step:bridge.control(d);mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
                if step==steps[index]:
                    maximum=max(maximum,float(np.max(np.abs(d.qpos[:160]-qpos[index]))))
                    index+=1
                    if index==len(steps):break
        print(f'M4 humanoid vs M3 max qpos difference: {maximum:.12g}')
        self.assertLess(maximum,1e-5)
        self.assertEqual(sum(w.number for w in d.warning),0)

if __name__=='__main__':unittest.main()
