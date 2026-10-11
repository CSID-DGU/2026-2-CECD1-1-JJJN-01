import unittest,json
import numpy as np
from examples.m6_2.experiment import OUT
from examples.m6_2.refine import StableVelocityEnv

class TrackingTests(unittest.TestCase):
 def test_velocity_control_keeps_ball_state_and_external_force_unchanged(self):
  e=StableVelocityEnv(reference=OUT/'references/G_stable_0.14_0.8_0.65.npz',velocity_ff=.8)
  try:
   e.reset(seed=7);q=e.d.qpos[e.qa:e.qa+7].copy();v=e.d.qvel[e.va:e.va+6].copy();e.bridge.control(e.d);np.testing.assert_array_equal(q,e.d.qpos[e.qa:e.qa+7]);np.testing.assert_array_equal(v,e.d.qvel[e.va:e.va+6]);self.assertFalse(e.d.xfrc_applied.any());self.assertFalse(e.d.qfrc_applied.any());self.assertEqual(e.action_space.shape,(30,));self.assertEqual(e.observation_space.shape,(169,))
  finally:e.close()
 def test_selected_actual_release_and_stability(self):
  z=json.loads((OUT/'runs/G_stable_0.14_0.8_0.65/summary.json').read_text());r=z['release'];np.testing.assert_array_equal(r['before_qpos'],r['after_qpos']);np.testing.assert_array_equal(r['before_qvel'],r['after_qvel']);self.assertGreater(z['forward_m_s'],0);self.assertGreater(z['vertical_m_s'],0);self.assertGreater(z['release_speed_m_s'],1.92);self.assertEqual((z['falls'],z['warnings']),(0,0));self.assertEqual(z['both_feet_contact_rate'],1);self.assertFalse(z['score'])
 def test_invalid_fast_controller_is_not_recommended(self):
  bad=json.loads((OUT/'runs/controller_ff_0.7/summary.json').read_text());self.assertGreater(bad['warnings'],0);r=json.loads((OUT/'recommended.json').read_text());self.assertNotEqual(r['condition'],'controller_ff_0.7');self.assertFalse(r['success_all_criteria'])

if __name__=='__main__':unittest.main(verbosity=2)
