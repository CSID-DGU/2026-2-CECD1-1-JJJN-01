"""Velocity bounds, rotating-frame derivative, no-state-write and branch-free goals."""
import argparse,json,tempfile,unittest
from pathlib import Path
import mujoco,numpy as np
from examples.m4_dynamic_grasp import ROOT,lift,ContinuousGoals,DynamicTargets,GuideTargets,local_ball,candidates,continuous_rotvec
class GoalServoTests(unittest.TestCase):
    def test_large_target_jumps_do_not_jump_goals_or_velocities(self):
        s=ContinuousGoals(np.zeros(2),np.array([3.,2.]),np.array([20.,10.]));last=s.q.copy();vel=s.v.copy();dt=1/450
        for i in range(1000):
            q=s.step(np.array([3.1,-3.1])*(1 if i<500 else -1),dt)
            self.assertTrue(np.all(abs(q-last)<=s.vmax*dt+1e-12));self.assertTrue(np.all(abs(s.v-vel)<=s.amax*dt+1e-12));last=q;vel=s.v.copy()
    def test_so3_path_crosses_pi_without_direction_flip(self):
        from scipy.spatial.transform import Rotation
        angles=np.linspace(2.9,3.4,100);path=Rotation.from_rotvec(np.c_[angles,np.zeros((100,2))]).as_matrix();v=continuous_rotvec(path)
        self.assertLess(np.max(np.linalg.norm(np.diff(v,axis=0),axis=1)),.006)
        np.testing.assert_allclose(Rotation.from_rotvec(v).as_matrix(),path,atol=1e-12)
    def test_candidates_do_not_mutate_baseline(self):
        c=candidates();b=c[0];source=next(x for x in json.loads((lift.PREVIOUS/'candidates.json').read_text()) if x['name']=='fk_fit_2');self.assertEqual(b,source)
        for x in c[1:]:
            self.assertLessEqual(abs(x['center_x']-b['center_x']),.006000001)
            self.assertLessEqual(max(abs(x['angles'][k]-v) for k,v in b['angles'].items()),.08000001)
@unittest.skipUnless((lift.PREVIOUS/'candidates.json').exists(),'Local model assets required')
class DynamicGoalTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();opts=argparse.Namespace(output=Path(self.temp.name),models=ROOT.parents[2]/'basketball-r1/SMPLSim/outputs/smplh_physics',source=ROOT/'outputs/m3_spl_fitting/spl_P0002_T0049.npz')
        self.m,self.d,self.b,_,_=lift.hand.setup(opts);self.c=candidates()[0];a=lift.hand.GraspTargets(self.b,self.c);lift.initialize(self.m,self.d,self.b,a,self.c)
    def tearDown(self):self.temp.cleanup()
    def test_relative_velocity_includes_rotating_frame(self):
        q=int(self.m.joint('basketball_free').qposadr[0]);v=int(self.m.joint('basketball_free').dofadr[0]);j=self.m.joint('R_Wrist_x');self.d.qvel[int(j.dofadr[0])]=1.2
        w=self.m.body('R_Wrist').id;jp,jr=np.zeros((3,self.m.nv)),np.zeros((3,self.m.nv));mujoco.mj_jac(self.m,self.d,jp,jr,self.d.qpos[q:q+3],w)
        self.d.qvel[v:v+3]=jp@self.d.qvel
        _,rv=local_ball(self.m,self.d);np.testing.assert_allclose(rv,0,atol=1e-12)
    def test_guide_updates_during_negative_prehold_clock_without_state_writes(self):
        adr=int(self.m.joint('basketball_free').qposadr[0]);a=GuideTargets(self.b,self.c,1.,self.d.qpos[adr:adr+3]);a.calibrate(self.d);before=self.d.qpos.copy();vel=self.d.qvel.copy()
        q=a.target(-2.5,self.d);self.assertIsNotNone(a.guide_last_fit)
        r=a.target(-2.5+1/450,self.d);self.assertLessEqual(np.max(abs(r[a.left_ids]-q[a.left_ids])),3/450+1e-9)
        np.testing.assert_array_equal(before,self.d.qpos);np.testing.assert_array_equal(vel,self.d.qvel)
    def test_dynamic_goals_preserve_physics_and_have_bounded_steps(self):
        a=DynamicTargets(self.b,self.c,'C',1.);a.calibrate(self.d);before=self.d.qpos.copy();vel=self.d.qvel.copy();motion=self.b.motion.poses.copy();last=None
        for i in range(1600):
            q=a.target(1+i/450,self.d)
            if last is not None:self.assertLess(np.max(abs(q[a.ids]-last)),6/450+1e-9)
            last=q[a.ids].copy();self.assertTrue(np.isfinite(q).all())
        np.testing.assert_array_equal(before,self.d.qpos);np.testing.assert_array_equal(vel,self.d.qvel);np.testing.assert_array_equal(motion,self.b.motion.poses)
        a.control(self.d,q);np.testing.assert_array_equal(before,self.d.qpos);np.testing.assert_array_equal(vel,self.d.qvel);self.assertEqual(self.m.neq,0);self.assertFalse(self.d.xfrc_applied.any());self.assertFalse(self.d.qfrc_applied.any())
if __name__=='__main__':unittest.main()
