"""Independent reference/readiness regressions; no training."""
import json,unittest
from pathlib import Path
import mujoco,numpy as np
from examples.m5_coordinated_reference import CoordinatedEnv,OUT,scene_rim,reserve,minimum_jerk
from examples.m5_coordinated_eval import readiness,audit

class CoordinationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.env=CoordinatedEnv(guide='phase',record=False)
    @classmethod
    def tearDownClass(cls):cls.env.close()
    def setUp(self):self.env.reset(seed=7)
    def test_reset_exact_old_ball(self):
        from examples.m5_residual_ppo_env import BASE
        old=json.loads((BASE/'C/initial_state.json').read_text());np.testing.assert_allclose(self.env.d.qpos[self.env.ball_q:self.env.ball_q+7],old['ball_qpos'],atol=1e-10,rtol=0)
    def test_scene_rim_from_geometry(self):
        from smpl_sim.physics.basketball import RIM_CENTER
        np.testing.assert_allclose(scene_rim(self.env.m,self.env.d),RIM_CENTER,atol=1e-12)
    def test_ball_unchanged_by_target_and_controller(self):
        e=self.env;q=e.d.qpos[e.ball_q:e.ball_q+7].copy();v=e.d.qvel[e.ball_v:e.ball_v+6].copy();target=e.adapter.target(.1,e.d);e.adapter.control(e.d,target);np.testing.assert_array_equal(q,e.d.qpos[e.ball_q:e.ball_q+7]);np.testing.assert_array_equal(v,e.d.qvel[e.ball_v:e.ball_v+6]);self.assertEqual(e.m.neq,0);self.assertFalse(e.d.xfrc_applied.any());self.assertFalse(e.d.qfrc_applied.any())
    def test_parameter_preservation(self):
        original=mujoco.MjModel.from_xml_path(str(OUT.parent/'m5_1_upper_body/bimanual_contact.xml'))
        for field in ('body_mass','body_inertia','geom_size','geom_friction','jnt_range','actuator_gear'):np.testing.assert_array_equal(getattr(self.env.m,field),getattr(original,field))
    def test_reference_reserve_is_finite(self):
        e=self.env;r=reserve(e.m,e.d,e.ref['rel0']);self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in r.values()));self.assertGreaterEqual(r['elbow_extension_reserve_deg'],0);self.assertGreaterEqual(r['wrist_rim_rotation_reserve_rad'],0)
    def test_preparation_head_raises_without_long_pause(self):
        e=self.env;a=e.adapter.target(0,e.d);b=e.adapter.target(.1,e.d);self.assertGreater(np.linalg.norm(b[e.head_ids]-a[e.head_ids]),0);self.assertEqual(e.adapter.phase,'A')
    def test_phase_boundaries_cartesian_continuity(self):
        e=self.env
        for boundary in (1.5,5.5,9.):
            e.adapter.target(boundary-1e-6,e.d);p=e.adapter.cartesian_right.copy();e.adapter.target(boundary+1e-6,e.d);self.assertLess(np.linalg.norm(e.adapter.cartesian_right-p),1e-5)
    def test_target_velocity_acceleration_filter(self):
        e=self.env
        for t in np.arange(0,.5,e.dt):
            prev=e.adapter.filtered_v.copy();e.adapter.target(float(t),e.d);self.assertLessEqual(np.max(abs(e.adapter.filtered_v)),2+1e-10);self.assertLessEqual(np.max(abs(e.adapter.filtered_v-prev)),8*e.dt+1e-10)
    def test_optional_guide_not_mandatory_in_gate(self):
        z=dict(finite=True,fallen=False,warnings=[0],contacts=2,drift_m=.005,goal_distance_m=.01,head_error_deg=5,both_feet=1,com_margin_m=.05,left_normal=0)
        self.assertTrue(self.env.stage_success(z));z['left_normal']=4;self.assertFalse(self.env.stage_success(z))
    def test_spaces_finite_and_rewards(self):
        e=self.env;o,r,t,tr,info=e.step(np.zeros(23));self.assertEqual(o.shape,(193,));self.assertTrue(np.isfinite(o).all());self.assertTrue(np.isfinite(r));self.assertEqual(info['ball_state_writes_after_reset'],0)
    def test_readiness_measures_ball_safety_and_force_direction(self):
        r=readiness(self.env)
        for key in ('palm_rim_dot','head_chest_relative_rotation_deg','body_ball_collision_force_N','left_forward_force_N','foot_slip_rms_m_s'):self.assertTrue(np.isfinite(r[key]))
    def test_hash_preservation(self):self.assertEqual(audit()['changed_scientific_files'],[])
    def test_short_contact_gap_uses_existing_loss_tolerance(self):
        e=self.env;z=dict(time_s=.01,finite=True,fallen=False,warnings=[0],contacts=0,drift_m=.005,goal_distance_m=.01,head_error_deg=5,both_feet=1,com_margin_m=.05,left_normal=0)
        e.seen=True;e.loss_start=.008;self.assertTrue(e.stage_success(z));z['time_s']=.07;self.assertFalse(e.stage_success(z))
    def test_bounded_ball_feedback_never_writes_ball(self):
        e=self.env;q=e.d.qpos[e.ball_q:e.ball_q+7].copy();e.config['ball_feedback']=True
        try:
            e.adapter.target(7,e.d);self.assertLessEqual(np.max(abs(e.adapter.feedback_offset)),.025+1e-12);np.testing.assert_array_equal(q,e.d.qpos[e.ball_q:e.ball_q+7])
        finally:e.config['ball_feedback']=False
    def test_head_reference_and_residual_variants(self):
        e=CoordinatedEnv(condition='D',record=False)
        try:
            self.assertEqual(e.action_space.shape,(28,));self.assertEqual(e.observation_space.shape,(203,));a=np.zeros(28);a[-2:]=.2;ref=e.adapter.target(0,e.d);target=e.residual_target(ref,a);self.assertGreater(np.linalg.norm(target[e.head_ids]-ref[e.head_ids]),0)
        finally:e.close()
    def test_candidate_limits(self):
        for kwargs in ({'tilt_deg':30},{'duration_scale':0},{'guide':'weld'}):
            with self.assertRaises(ValueError):CoordinatedEnv(**kwargs)
    def test_kinematic_not_physics_success(self):self.assertFalse(json.loads((OUT/'reference/summary.json').read_text())['ik_physics_success'])

if __name__=='__main__':unittest.main()
