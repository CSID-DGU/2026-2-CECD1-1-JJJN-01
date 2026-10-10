"""Geodesic motion goals, source preservation and measured transition continuity."""
import argparse,json,tempfile,unittest
from pathlib import Path
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from examples.m4_spl_contact_integration import ROOT,lift,IntegrationTargets,geodesic_goal
@unittest.skipUnless((lift.PREVIOUS/'candidates.json').exists(),'Local contact grasp required')
class SPLContactIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();args=argparse.Namespace(output=Path(self.temp.name),models=ROOT.parents[2]/'basketball-r1/SMPLSim/outputs/smplh_physics',source=ROOT/'outputs/m3_spl_fitting/spl_P0002_T0049.npz')
        self.m,self.d,self.b,a,q=lift.hand.setup(args);self.c=next(c for c in json.loads((lift.PREVIOUS/'candidates.json').read_text()) if c['name']=='fk_fit_2')
        self.b.initialize(self.m,self.d);self.d.qpos[:self.b.model.nq]=lift.hand.GraspTargets(self.b,self.c).target(0);mujoco.mj_forward(self.m,self.d)
    def tearDown(self):self.temp.cleanup()
    def test_geodesic_rotation_bound_and_endpoints(self):
        R0=Rotation.from_euler('xyz',[.2,.3,.4]).as_matrix();R1=R0@Rotation.from_rotvec([.7,.3,-.4]).as_matrix()
        for a in [0,.25,.5,1]:
            R,alpha=geodesic_goal(R0,R1,a,15);self.assertLessEqual(np.degrees(Rotation.from_matrix(R0.T@R).magnitude()),15+1e-9);np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-12);self.assertAlmostEqual(np.linalg.det(R),1.)
        np.testing.assert_allclose(geodesic_goal(R0,R1,0,15)[0],R0)
        np.testing.assert_allclose(geodesic_goal(R0,R1,1,180)[0],R1,atol=1e-12)
    def test_transition_continuity_keeps_grasp_without_mutating_motion(self):
        before=self.b.motion.poses.copy();a=IntegrationTargets(self.b,self.c,1,2,'C',15)
        for boundary in [1.,3.]:
            left=a.target(boundary-1e-6,self.d);right=a.target(boundary+1e-6,self.d);self.assertLess(np.max(np.abs(left-right)),1e-4)
        for t in [0,2,3.5,5.5]:
            q=a.target(t,self.d);self.assertAlmostEqual(np.linalg.norm(q[3:7]),1.)
            for name,val in self.c['angles'].items():self.assertAlmostEqual(q[int(self.m.joint(name).qposadr[0])],val)
        np.testing.assert_array_equal(self.b.motion.poses,before)
    def test_real_source_arm_goals_and_time_varying_orientation(self):
        a=IntegrationTargets(self.b,self.c,1,2,'C',15);q=a.target(3.5,self.d);source=self.b.target(.5)
        arm=[int(self.m.joint(n).qposadr[0]) for n in lift.ARM[:6]];np.testing.assert_allclose(q[arm],source[arm])
        a.target(1,self.d);R1=a.R_goal.copy();a.target(3.5,self.d);self.assertGreater(Rotation.from_matrix(R1.T@a.R_goal).magnitude(),.01)
    def test_goal_and_controller_preserve_ball_and_apply_no_external_force(self):
        a=IntegrationTargets(self.b,self.c,1,2,'C',15)
        for t in [1,2,3.5,5]:
            before=self.d.qpos.copy();velocity=self.d.qvel.copy();goal=a.target(t,self.d);a.control(self.d,goal)
            np.testing.assert_array_equal(self.d.qpos,before);np.testing.assert_array_equal(self.d.qvel,velocity)
            self.assertFalse(self.d.qfrc_applied.any());self.assertFalse(self.d.xfrc_applied.any());self.assertEqual(self.m.neq,0)
if __name__=='__main__':unittest.main()
