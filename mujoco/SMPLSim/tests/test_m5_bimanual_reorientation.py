"""Independent rotation/contact experiment: FK, state integrity, axis limits."""
import unittest
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from examples.m5_bimanual_reorientation import RotationEnv,pose_ik,palm_basis,MODELS,ROOT
from examples.m5_upper_body_ik import joint_ids

class BimanualRotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.env=RotationEnv(condition='C',support_tilt=0,finger_flex=0,guide_force=1,orbit=.5)
    @classmethod
    def tearDownClass(cls):cls.env.close()
    def setUp(self):self.env.reset(seed=7)
    def test_target_and_original_controller_leave_ball_state_unchanged(self):
        e=self.env;p=e.d.qpos[e.ball_q:e.ball_q+7].copy();v=e.d.qvel[e.ball_v:e.ball_v+6].copy();q=e.adapter.target(.1,e.d);e.adapter.control(e.d,q)
        np.testing.assert_array_equal(p,e.d.qpos[e.ball_q:e.ball_q+7]);np.testing.assert_array_equal(v,e.d.qvel[e.ball_v:e.ball_v+6]);self.assertEqual(e.m.neq,0);self.assertFalse(e.d.xfrc_applied.any());self.assertFalse(e.d.qfrc_applied.any())
    def test_left_physical_contact_is_required_before_rotation(self):
        e=self.env;e.adapter.target(2.,e.d);self.assertIsNone(e.adapter.rotation_start)
    def test_fk_ik_recovers_actual_wrist_pose_within_limits(self):
        e=self.env;q=e.d.qpos[:e.bridge.model.nq].copy();m=e.bridge.model;probe=mujoco.MjData(m);probe.qpos[:]=q;mujoco.mj_kinematics(m,probe);p=probe.xpos[m.body('R_Wrist').id].copy();R=probe.xmat[m.body('R_Wrist').id].reshape(3,3).copy();got,z=pose_ik(m,q,'R',p,R)
        self.assertLess(z['position_error_mm'],.01);self.assertLess(z['orientation_error_deg'],.01)
        for name in [f'R_{b}_{a}' for b in ('Shoulder','Elbow','Wrist') for a in 'xyz']:
            j=m.joint(name);self.assertGreaterEqual(got[j.qposadr[0]],j.range[0]);self.assertLessEqual(got[j.qposadr[0]],j.range[1])
    def test_opposite_orientation_candidate_changes_actual_fk_finger(self):
        e=self.env;m=e.bridge.model;probe=mujoco.MjData(m);probe.qpos[:]=e.d.qpos[:m.nq];mujoco.mj_kinematics(m,probe);basis,finger,normal=palm_basis(m,probe,'R');W=probe.xmat[m.body('R_Wrist').id].reshape(3,3).copy();p=probe.xpos[m.body('R_Wrist').id].copy();q=probe.qpos.copy()
        # Continuous IK sequence, not a single hinge flip. Ball is not stepped.
        for a in np.linspace(0,150,31):q,z=pose_ik(m,q,'R',p,Rotation.from_rotvec(normal*np.radians(a)).as_matrix()@W,iterations=80)
        probe.qpos[:]=q;mujoco.mj_kinematics(m,probe);_,end,_=palm_basis(m,probe,'R');self.assertGreater(np.degrees(np.arccos(np.clip(finger@end,-1,1))),140);self.assertLess(z['position_error_mm'],1)
    def test_model_masses_geometries_limits_and_actuators_are_unchanged(self):
        e=self.env;original=mujoco.MjModel.from_xml_path(str(ROOT/'outputs/m5_1_upper_body/bimanual_contact.xml'))
        for key in ('body_mass','body_inertia','geom_size','geom_friction','jnt_range','actuator_gear'):np.testing.assert_array_equal(getattr(e.m,key),getattr(original,key))
    def test_target_velocity_and_acceleration_are_bounded(self):
        e=self.env;last=e.adapter.target(0,e.d)
        for t in np.arange(e.dt,.3,e.dt):
            q=e.adapter.target(t,e.d);self.assertLessEqual(np.max(abs(e.adapter.filtered_v)),1.5+1e-10);self.assertLessEqual(np.max(abs(q-last)),1.5*e.dt+1e-8);last=q


class WholeBodyKinematicTests(unittest.TestCase):
    def setUp(self):
        from examples.m5_bimanual_body_ik import calibration
        self.env=RotationEnv(condition='C');self.m=self.env.bridge.model;self.d=mujoco.MjData(self.m);self.d.qpos[:]=self.env.d.qpos[:self.m.nq];mujoco.mj_kinematics(self.m,self.d);self.cal=calibration(self.m,self.d)
    def tearDown(self):self.env.close()
    def test_wrist_component_is_not_forearm_rotation(self):
        from examples.m5_bimanual_body_ik import extension_angle
        baseline=extension_angle(self.m,self.d,self.cal)
        self.d.qpos[int(self.m.joint('R_Elbow_x').qposadr[0])]+=.2;mujoco.mj_kinematics(self.m,self.d)
        self.assertAlmostEqual(extension_angle(self.m,self.d,self.cal),baseline,places=6)
        changes=[]
        for axis in 'xyz':
            adr=int(self.m.joint('R_Wrist_'+axis).qposadr[0]);old=self.d.qpos[adr];self.d.qpos[adr]+=.2;mujoco.mj_kinematics(self.m,self.d);changes.append(abs(extension_angle(self.m,self.d,self.cal)-baseline));self.d.qpos[adr]=old
        self.assertGreater(max(changes),.1)
    def test_wholebody_ik_preserves_noop_pose_and_bounds(self):
        from examples.m5_bimanual_body_ik import whole_body_ik,trunk_axis
        w=self.m.body('R_Wrist').id;got,z=whole_body_ik(self.m,self.d.qpos,self.d.xpos[w].copy(),self.d.xmat[w].reshape(3,3).copy(),trunk_axis(self.m,self.d),self.cal,0)
        self.assertLess(z['position_error_mm'],.01);self.assertLess(z['extension_error_deg'],.01);self.assertLess(z['trunk_error_deg'],.01)
    def test_head_ik_aligns_to_actual_scene_rim(self):
        from examples.m5_bimanual_body_ik import head_ik
        target=self.env.ref['rim_center'];q=head_ik(self.m,self.d.qpos,target);self.d.qpos[:]=q;mujoco.mj_kinematics(self.m,self.d);h=self.m.body('Head').id;forward=self.d.xmat[h].reshape(3,3)[:,2];direction=target-self.d.xpos[h];direction/=np.linalg.norm(direction)
        self.assertGreater(forward@direction,.99)
    def test_bimanual_ik_uses_the_same_planned_trunk(self):
        e=self.env;e.adapter.exp['trunk_extension_deg']=8;e.adapter.exp['wrist_extension_deg']=15;e.adapter.target(.1,e.d);e.adapter.rotation_start=0;e.adapter.target(2,e.d);a=e.adapter;probe=mujoco.MjData(e.m);probe.qpos[:]=e.d.qpos;probe.qpos[a.trunk_ids]=a.unfiltered[a.trunk_ids];probe.qpos[a.left_ids]=a.unfiltered[a.left_ids];mujoco.mj_kinematics(e.m,probe);actual_error=np.linalg.norm(probe.xpos[e.m.body('L_Wrist').id]-a.cartesian_left)*1000;self.assertAlmostEqual(actual_error,a.jac_reports['L']['position_error_mm'],places=5)
    def test_wrist_component_has_principal_quaternion_branch(self):
        from examples.m5_bimanual_body_ik import extension_angle
        for angle in np.linspace(-3,3,13):
            self.d.qpos[int(self.m.joint('R_Wrist_x').qposadr[0])]=angle;mujoco.mj_kinematics(self.m,self.d);value=extension_angle(self.m,self.d,self.cal);self.assertLessEqual(abs(value),np.pi+1e-10)
    def test_proximity_guide_changes_only_targets(self):
        e=self.env;e.adapter.exp['proximity_guide']=True;before=e.d.qpos.copy();velocity=e.d.qvel.copy();e.adapter.target(2.1,e.d);np.testing.assert_array_equal(e.d.qpos,before);np.testing.assert_array_equal(e.d.qvel,velocity);self.assertLessEqual(e.adapter.guide_depth,.035);self.assertEqual(e.m.neq,0)
    def test_reference_interpolation_quaternion_and_endpoints(self):
        import tempfile
        from examples.m5_bimanual_reference import ReferenceSequence
        q=self.d.qpos.copy();q2=q.copy();q2[3:7]=Rotation.from_rotvec([0,0,.4]).as_quat()[[3,0,1,2]]
        with tempfile.TemporaryDirectory() as folder:
            path=folder+'/r.npz';np.savez(path,time_s=[0.,1.],target_qpos=[q,q2],joint_names=['example']);r=ReferenceSequence(path);self.assertAlmostEqual(np.linalg.norm(r.target(.5)[3:7]),1);np.testing.assert_allclose(r.target(-1),q,atol=1e-12);np.testing.assert_allclose(r.target(2),q2,atol=1e-12)

if __name__=='__main__':unittest.main()
