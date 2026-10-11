"""M5 API, zero-residual provenance, physics constraints and saved PPO smoke."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from examples.m5_residual_ppo_env import BasketballResidualEnv,OUT,BASE
from examples.m5_residual_ppo_eval import require_gate
class ResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.env=BasketballResidualEnv(use_snapshot=True)
    @classmethod
    def tearDownClass(cls):cls.env.close()
    def setUp(self):self.env.max_seconds=6.5;self.env.success_enabled=False;self.obs,self.info=self.env.reset(seed=123)
    def test_gymnasium_api(self):
        from gymnasium.utils.env_checker import check_env
        from stable_baselines3.common.env_checker import check_env as sb_check
        check_env(self.env,skip_render_check=True);sb_check(self.env,warn=True,skip_render_check=True)
    def test_observation_space_and_finite(self):
        self.assertEqual(self.obs.shape,(74,));self.assertEqual(self.obs.dtype,np.float32);self.assertTrue(self.env.observation_space.contains(self.obs));self.assertTrue(np.isfinite(self.obs).all())
    def test_action_space(self):
        self.assertEqual(self.env.action_space.shape,(12,));self.assertTrue(self.env.action_space.contains(np.zeros(12,np.float32)))
    def test_zero_baseline_matches_existing_lifting_quantitatively(self):
        gate=require_gate();self.assertTrue(gate['passed']);self.assertEqual(gate['max_recorded_qpos_difference'],0.);self.assertEqual(gate['snapshot_max_qpos_difference'],0.)
        old=json.loads((BASE/'C/summary.json').read_text())
        for key in gate['metric_differences']:self.assertAlmostEqual(gate['baseline'][key],old[key],places=12)
    def test_residual_changes_targets_without_direct_ball_state_writes(self):
        self.env.record=True;self.env.step(np.ones(12,np.float32));self.assertGreater(np.linalg.norm(self.env.last_target[self.env.ids]-self.env.last_reference[self.env.ids]),0);self.assertEqual(self.env.ball_state_writes_after_reset,0);self.assertTrue(self.env.goals);self.env.record=False
    def test_action_joint_and_residual_bounds(self):
        self.env.step(np.full(12,100.));np.testing.assert_array_equal(self.env.action,np.ones(12));delta=self.env.last_target[self.env.ids]-self.env.last_reference[self.env.ids];self.assertTrue(np.all(abs(delta)<=np.sum(abs(self.env.synergy),axis=1)+1e-10));self.assertTrue(np.all(self.env.last_target[self.env.ids]>=self.env.bounds[:,0]));self.assertTrue(np.all(self.env.last_target[self.env.ids]<=self.env.bounds[:,1]))
    def test_seed_reproducibility(self):
        a,_=self.env.reset(seed=43);self.env.step(np.full(12,.2));state=self.env.d.qpos.copy();b,_=self.env.reset(seed=43);self.env.step(np.full(12,.2));np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(state,self.env.d.qpos)
    def test_snapshot_disk_round_trip_and_initial_controller_state(self):
        snapshot=copy.deepcopy(self.env.snapshot)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'reset.npz';self.env.save_snapshot(p);self.env.load_snapshot(p)
        for key in ['physics','bridge','kp','kd','jac','last_correction','limit']:np.testing.assert_array_equal(snapshot[key],self.env.snapshot[key])
    def test_no_attachment_force_or_model_parameter_changes(self):
        m=self.env.m;mass=m.body_mass.copy();inertia=m.body_inertia.copy();friction=m.geom_friction.copy();self.env.step(np.zeros(12));self.assertEqual(m.neq,0);self.assertFalse(self.env.d.qfrc_applied.any());self.assertFalse(self.env.d.xfrc_applied.any());self.assertEqual(int(m.joint('Pelvis').type[0]),0);self.assertAlmostEqual(m.body('basketball').mass[0],.6)
        for a,b in [(mass,m.body_mass),(inertia,m.body_inertia),(friction,m.geom_friction)]:np.testing.assert_array_equal(a,b)
    def test_contact_loss_50ms_not_microbounce(self):
        z=copy.deepcopy(self.env.last_metrics);z.update(contacts=0,drift_m=0.,fallen=False,time_s=.01);self.env.seen=True;self.assertIsNone(self.env.termination(z));z['time_s']=.049;self.assertIsNone(self.env.termination(z));z['time_s']=.061;self.assertEqual(self.env.termination(z),'contact_loss')
    def test_contact_can_exist_but_drift_is_failure(self):
        z=copy.deepcopy(self.env.last_metrics);z.update(contacts=3,drift_m=.051,fallen=False);self.assertEqual(self.env.termination(z),'ball_relative_drift')
    def test_time_limit_is_truncation(self):
        self.env.max_seconds=.04;self.env.step(np.zeros(12));_,_,term,trunc,info=self.env.step(np.zeros(12));self.assertFalse(term);self.assertTrue(trunc);self.assertEqual(info['termination_reason'],'time_limit')
    def test_reward_target_better_than_static_and_detached_has_no_positive_terms(self):
        z=copy.deepcopy(self.env.last_metrics);z.update(contacts=2,drift_m=0.,goal_distance_m=.25,fallen=False,slip_ms=0.);hold=self.env.reward_terms(z,np.zeros(12));z['goal_distance_m']=0;goal=self.env.reward_terms(z,np.zeros(12));self.assertGreater(goal['ball_target'],hold['ball_target']);z.update(contacts=0,drift_m=.1);drop=self.env.reward_terms(z,np.zeros(12));self.assertEqual(sum(drop[k] for k in ['ball_target','grasp','motion','balance']),0.)
    def test_invalid_action_rejected_and_nonfinite_state_terminates(self):
        for value in [np.nan,np.inf]:
            with self.assertRaises(ValueError):self.env.step(np.full(12,value))
        self.env.d.qvel[0]=np.nan;obs,reward,term,trunc,info=self.env.step(np.zeros(12));self.assertTrue(term);self.assertFalse(trunc);self.assertTrue(np.isfinite(obs).all());self.assertTrue(np.isfinite(reward));self.assertEqual(info['termination_reason'],'nonfinite_or_physics_warning')
    def test_goal_success_hold_termination(self):
        self.env.success_enabled=True;z=copy.deepcopy(self.env.last_metrics);z.update(contacts=1,drift_m=.001,goal_distance_m=.001,both_feet=1,com_margin_m=.1,fallen=False,time_s=6.);self.assertIsNone(self.env.termination(z));z['time_s']=6.51;self.assertEqual(self.env.termination(z),'goal_stable')
    def test_ppo_save_load(self):
        from stable_baselines3 import PPO
        p=OUT/'smoke/ppo_smoke.zip';self.assertTrue(p.exists());a=PPO.load(p,device='cpu');b=PPO.load(p,device='cpu');np.testing.assert_array_equal(a.predict(self.obs,deterministic=True)[0],b.predict(self.obs,deterministic=True)[0]);s=json.loads((OUT/'smoke/summary.json').read_text());self.assertTrue(s['save_load_action_exact']);self.assertTrue(s['finite_rewards'])
if __name__=='__main__':unittest.main()
