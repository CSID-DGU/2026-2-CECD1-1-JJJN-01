import unittest
import numpy as np
from examples.m6.core import ScoringDetector,WeldShootingEnv,RIM_CENTER,BALL_RADIUS
class DetectorTests(unittest.TestCase):
 def trajectory(self,xy=0,up=False,bounce=False,stationary=False,released=True):
  s=ScoringDetector();z=np.linspace(.3,-.3,61)
  if up:z=z[::-1]
  if bounce:z=np.r_[np.linspace(.3,.03,20),np.linspace(.03,.4,20)]
  if stationary:z=np.zeros(20)
  events=[s.update(RIM_CENTER+[xy,0,h],released) for h in z];return sum(events)
 def test_center(self):self.assertEqual(self.trajectory(),1)
 def test_outside(self):self.assertEqual(self.trajectory(.3),0)
 def test_bounce_out(self):self.assertEqual(self.trajectory(bounce=True),0)
 def test_upward(self):self.assertEqual(self.trajectory(up=True),0)
 def test_stationary(self):self.assertEqual(self.trajectory(stationary=True),0)
 def test_not_released(self):self.assertEqual(self.trajectory(released=False),0)
 def test_backboard_then_entry(self):
  # Detector-only fixture: earlier wall rebound, then descending center pass.
  s=ScoringDetector();s.update(RIM_CENTER+[.35,0,.7],True);s.update(RIM_CENTER+[.1,0,.4],True)
  self.assertTrue(any(s.update(RIM_CENTER+[0,0,z],True) for z in np.linspace(.3,-.3,61)))
 def test_fast_full_crossing(self):
  s=ScoringDetector();s.update(RIM_CENTER+[0,0,.3],True);self.assertTrue(s.update(RIM_CENTER+[0,0,-.3],True))
 def test_oblique_exit(self):
  s=ScoringDetector();self.assertTrue(any(s.update(RIM_CENTER+[.25*(.3-z),0,z],True) for z in np.linspace(.3,-.3,121)))
 def test_no_repeat(self):
  s=ScoringDetector();self.assertEqual(sum(s.update(RIM_CENTER+[0,0,z],True) for z in list(np.linspace(.3,-.3,61))*2),1)
class ReferenceTests(unittest.TestCase):
 def test_reference_continuity(self):
  from examples.m6.core import Reference,generate_reference
  r=Reference(generate_reference('shoot'));samples=np.array([r.target(t) for t in np.linspace(0,r.duration,500)])
  self.assertTrue(np.isfinite(samples).all());self.assertLess(np.max(abs(np.diff(samples[:,7:],axis=0))),.2)
  np.testing.assert_allclose(np.linalg.norm(samples[:,3:7],axis=1),1,atol=1e-9)
 def test_ppo_save_load_checkpoint(self):
  from stable_baselines3 import PPO
  from examples.m6.core import OUT
  a=PPO.load(OUT/'smoke_verified/final.zip',device='cpu');b=PPO.load(OUT/'smoke_verified/checkpoints/m6_256_steps.zip',device='cpu')
  e=WeldShootingEnv();o,_=e.reset(seed=7);action=a.predict(o,deterministic=True)[0];self.assertEqual(action.shape,(30,));self.assertTrue(np.isfinite(b.predict(o,deterministic=True)[0]).all())
  a.save(OUT/'save_load_roundtrip');c=PPO.load(OUT/'save_load_roundtrip.zip',device='cpu');np.testing.assert_array_equal(action,c.predict(o,deterministic=True)[0]);e.close()
 def test_resume_counter(self):
  import json
  from examples.m6.core import OUT
  z=json.loads((OUT/'resume_verified/training.json').read_text());self.assertEqual(z['initial_steps'],256);self.assertEqual(z['final_steps'],384)
class PhysicsTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.e=WeldShootingEnv(mode='A',stage=0)
 @classmethod
 def tearDownClass(cls):cls.e.close()
 def test_spaces_seed(self):
  e=self.e;o,_=e.reset(seed=42);self.assertTrue(e.observation_space.contains(o));a=e.action_space.sample();o1,r1,*_=e.step(a);e.reset(seed=42);o2,r2,*_=e.step(a);np.testing.assert_array_equal(o1,o2);self.assertEqual(r1,r2);self.assertTrue(np.isfinite(r1))
 def test_release_state_preserved(self):
  e=self.e;e.reset(seed=1)
  while not e.grip.released:e.step(np.zeros(e.action_space.shape))
  ev=e.release_event;np.testing.assert_array_equal(ev['before_qpos'],ev['after_qpos']);np.testing.assert_array_equal(ev['before_qvel'],ev['after_qvel']);self.assertEqual(ev['angular_velocity_jump_rad_s'],0);self.assertEqual(e.grip.state_write_count,1);self.assertEqual(e.d.eq_active[e.grip.equality],0)
 def test_invalid_action(self):
  self.e.reset();self.assertRaises(ValueError,self.e.step,np.full(self.e.action_space.shape,np.nan))
 def test_gym_checker(self):
  from stable_baselines3.common.env_checker import check_env
  check_env(self.e,warn=True)
if __name__=='__main__':unittest.main(verbosity=2)
