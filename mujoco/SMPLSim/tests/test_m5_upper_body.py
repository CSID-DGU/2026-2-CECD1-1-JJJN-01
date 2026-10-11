"""M5-1 API/physical invariants and measured feasibility, without new training."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from examples.m5_upper_body_ppo_env import UpperBodyResidualEnv,OUT
from examples.m5_residual_ppo_env import BASE
from examples.m5_residual_ppo_eval import require_gate
from examples.m5_bimanual_contact import derived_scene,side_contacts
from smpl_sim.physics.basketball import RIM_CENTER
REFERENCE=OUT/'references/guide_safe/reference.npz'

class UpperBodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.env=UpperBodyResidualEnv(condition='B',stage=1,reference=REFERENCE)
    @classmethod
    def tearDownClass(cls):cls.env.close()
    def setUp(self):self.env.max_seconds=7.;self.env.success_enabled=False;self.obs,self.info=self.env.reset(seed=123)
    def test_original_m5_gate(self):self.assertTrue(require_gate()['passed'])
    def test_existing_lifting_trace_preserved(self):
        measured=np.load(OUT/'final_baseline/states.npz')['qpos'];original=np.load(BASE/'C/states.npz')['qpos'];np.testing.assert_array_equal(measured,original);self.assertTrue(json.loads((OUT/'baseline_gate.json').read_text())['passed'])
    def test_gymnasium_api(self):
        from gymnasium.utils.env_checker import check_env
        from stable_baselines3.common.env_checker import check_env as sb_check
        check_env(self.env,skip_render_check=True);sb_check(self.env,warn=True)
    def test_space_and_finite(self):
        self.assertEqual(self.obs.shape,(193,));self.assertEqual(self.env.action_space.shape,(23,));self.assertTrue(self.env.observation_space.contains(self.obs));self.assertTrue(np.isfinite(self.obs).all())
    def test_collision_masks_enable_left_and_face_without_disabling_right(self):
        m=self.env.m;ball=m.geom('basketball_geom').id
        for side in ('R','L'):
            for n in [side+'_Wrist',side+'_Index1']:
                self.assertTrue(m.geom_contype[m.geom(n).id]&m.geom_conaffinity[ball])
        self.assertTrue(m.geom_contype[m.geom('Head').id]&m.geom_conaffinity[ball])
    def test_model_parameters_unchanged(self):
        original=mujoco.MjModel.from_xml_path(str((BASE/'contact.xml').resolve()));m=self.env.m
        for key in ['body_mass','body_inertia','geom_size','geom_pos','geom_quat','geom_friction','geom_solref','geom_solimp','jnt_range','actuator_gear']:np.testing.assert_array_equal(getattr(m,key),getattr(original,key))
        self.assertEqual(m.nq,original.nq);self.assertEqual(m.nu,153);self.assertEqual(m.neq,0);self.assertAlmostEqual(m.body('basketball').mass[0],.6)
    def test_measured_left_contact_force(self):
        x=np.genfromtxt(OUT/'feasibility/guide_safe/physics.csv',delimiter=',',names=True)
        self.assertGreater(x['left_contacts'].max(),0);self.assertGreater(x['left_normal'].max(),.01);self.assertTrue(np.isfinite(x['left_normal']).all())
        rows=list(__import__('csv').DictReader((OUT/'feasibility/guide_safe/contacts.csv').open()));self.assertTrue(any(r['side']=='L' and float(r['normal_force_N'])>.01 for r in rows))
    def test_no_direct_ball_state_or_external_force(self):
        self.env.step(np.zeros(23));self.assertEqual(self.env.ball_state_writes_after_reset,0);self.assertFalse(self.env.d.xfrc_applied.any());self.assertFalse(self.env.d.qfrc_applied.any());self.assertEqual(self.env.m.neq,0)
    def test_residual_and_bounds(self):
        self.env.step(np.full(23,100.));delta=self.env.last_target[self.env.ids]-self.env.last_reference[self.env.ids];self.assertGreater(np.linalg.norm(delta),0);self.assertTrue(np.all(abs(delta)<=np.sum(abs(self.env.synergy),axis=1)+1e-10));self.assertTrue(np.all(self.env.last_target[self.env.ids]>=self.env.bounds[:,0]));self.assertTrue(np.all(self.env.last_target[self.env.ids]<=self.env.bounds[:,1]))
    def test_seed_and_snapshot_reproducibility(self):
        a,_=self.env.reset(seed=9);self.env.step(np.full(23,.1));q=self.env.d.qpos.copy();b,_=self.env.reset(seed=9);self.env.step(np.full(23,.1));np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(q,self.env.d.qpos)
    def test_reference_head_and_trunk_fk(self):
        r=dict(np.load(REFERENCE));d=mujoco.MjData(self.env.m);d.qpos[:]=r['aligned'];mujoco.mj_forward(self.env.m,d);head=self.env.m.body('Head').id;desired=RIM_CENTER-d.xpos[head];desired/=np.linalg.norm(desired);error=np.degrees(np.arccos(np.clip(np.dot(desired,d.xmat[head].reshape(3,3)[:,2]),-1,1)));self.assertLess(error,5.)
        np.testing.assert_allclose(r['aligned'][r['trunk_ids']],.8*r['approached'][r['trunk_ids']])
    def test_wrist_tracking_metrics_are_separate_and_finite(self):
        self.env.step(np.zeros(23));z=self.env.last_metrics
        for side in ('R','L'):
            for k in ('ref_target_mm','target_actual_mm','ref_actual_mm','orientation_error_deg'):self.assertTrue(np.isfinite(z[side+'_'+k]));self.assertGreaterEqual(z[side+'_'+k],0)
    def test_goal_joint_filter_velocity_and_acceleration_limits(self):
        e=self.env;a=e.adapter;previous=None;velocity=None
        for t in np.arange(0,1.1,.0022222222222222222):
            q=a.target(float(t),e.d)
            if previous is not None:
                v=(q-previous)/e.dt;self.assertLessEqual(np.max(abs(v)),2.000001)
                if velocity is not None:self.assertLessEqual(np.max(abs(v-velocity)/e.dt),8.00001)
                velocity=v
            previous=q.copy()
    def test_later_reference_phases_have_finite_targets(self):
        adapter=self.env.adapter;adapter.stage=4
        for t in (6.,10.,14.,15.,20.,23.,25.):
            target=adapter.target(t,self.env.d);self.assertEqual(target.shape,(160,));self.assertTrue(np.isfinite(target).all())
    def test_contact_loss_microbounce_and_drift(self):
        z=copy.deepcopy(self.env.last_metrics);z.update(contacts=0,drift_m=0.,fallen=False,time_s=.01);self.assertIsNone(self.env.termination(z));z['time_s']=.061;self.assertEqual(self.env.termination(z),'contact_loss');z.update(contacts=2,drift_m=.051);self.assertEqual(self.env.termination(z),'ball_relative_drift')
    def test_time_limit_truncation(self):
        self.env.max_seconds=.04;self.env.step(np.zeros(23));_,reward,term,trunc,info=self.env.step(np.zeros(23));self.assertFalse(term);self.assertTrue(trunc);self.assertEqual(info['termination_reason'],'time_limit');self.assertTrue(np.isfinite(reward))
    def test_reward_shaping_and_no_detached_positive_reward(self):
        z=copy.deepcopy(self.env.last_metrics);z.update(contacts=2,drift_m=.001,fallen=False,goal_distance_m=1.);self.env.prev_goal_distance=1.;far=self.env.reward_terms(z,np.zeros(23));self.assertGreater(far['ball_progress'],0);z['goal_distance_m']=.9;near=self.env.reward_terms(z,np.zeros(23));self.assertGreater(near['ball_progress'],far['ball_progress']);z['contacts']=0;drop=self.env.reward_terms(z,np.zeros(23));self.assertEqual(sum(drop[k] for k in ('ball_progress','grasp_stability','bimanual_coordination','shooting_pose','head_alignment','balance','reference_tracking')),0.)
    def test_excessive_guide_force_is_penalized(self):
        z=copy.deepcopy(self.env.last_metrics);z.update(left_contacts=3,left_normal=25.,left_max_normal=25.,left_force=np.array([0.,-25.,0.]),contacts=2,drift_m=.001);r=self.env.reward_terms(z,np.zeros(23));self.assertEqual(r['bimanual_coordination'],0);self.assertGreater(r['guide_push'],0.)
    def test_nonfinite_state_terminates_with_finite_output(self):
        self.env.d.qvel[0]=np.nan;obs,r,term,trunc,info=self.env.step(np.zeros(23));self.assertTrue(term);self.assertFalse(trunc);self.assertTrue(np.isfinite(obs).all());self.assertTrue(np.isfinite(r));self.assertTrue(all(np.isfinite(v) for v in info['reward_terms'].values()))
    def test_gui_headless_trace(self):
        a=np.load(OUT/'gui_guide/states.npz')['qpos'];b=np.load(OUT/'feasibility/guide_safe/states.npz')['qpos'];np.testing.assert_array_equal(a,b)
    def test_invalid_actions(self):
        for x in [np.full(23,np.nan),np.full(23,np.inf),np.zeros(12)]:
            with self.assertRaises(ValueError):self.env.step(x)
    def test_ppo_save_load_and_condition_dimensions(self):
        from stable_baselines3 import PPO
        for condition,dim in [('A',12),('B',23),('C',26),('D',28)]:
            folder=OUT/'smoke'/f'{condition}_stage1_seed7';summary=json.loads((folder/'summary.json').read_text());self.assertTrue(summary['finite_rewards']);self.assertTrue(summary['save_load_action_exact']);self.assertFalse(summary['shooting_learned']);a=PPO.load(folder/'ppo_smoke.zip',device='cpu');b=PPO.load(folder/'ppo_smoke.zip',device='cpu');obs=np.zeros(a.observation_space.shape,np.float32);np.testing.assert_array_equal(a.predict(obs,deterministic=True)[0],b.predict(obs,deterministic=True)[0]);self.assertEqual(a.action_space.shape,(dim,));self.assertEqual(a.observation_space.shape,(147+2*dim,))
    def test_protected_files_hashes(self):
        before=json.loads((OUT/'preservation_before.json').read_text())
        for name,digest in before.items():
            if Path(name).name=='.DS_Store':continue # Finder metadata is reported separately in the audit.
            self.assertEqual(hashlib.sha256(Path(name).read_bytes()).hexdigest(),digest,name)
if __name__=='__main__':unittest.main()
