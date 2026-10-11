"""Real online rollout and portable physics baseline verification."""
import sys,json,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from free_throw_env import FreeThrowEnv,quintic


class OnlineEnvironmentTests(unittest.TestCase):
    def test_zero_action_reproduces_contact_baseline_and_finishes_hold(self):
        env=FreeThrowEnv();obs,_=env.reset(seed=42)
        self.assertTrue(env.observation_space.contains(obs))
        for invalid in [np.zeros(4),np.full(5,np.nan),np.full(5,2)]:
            with self.assertRaises(ValueError):env.step(invalid)
        done=False;steps=0
        while not done:
            obs,reward,done,truncated,info=env.step(np.zeros(5));steps+=1
            self.assertTrue(np.isfinite(obs).all());self.assertFalse(truncated)
        metrics=info['metrics'];baseline=json.loads((ROOT/'runs/spl_residual_ready/shooting_report.json').read_text())
        self.assertEqual(steps,88);self.assertTrue(metrics['accepted']);self.assertFalse(metrics['scored'])
        np.testing.assert_allclose(metrics['release_velocity_m_s'],baseline['release']['linear_velocity_m_s'],atol=1e-8)
        self.assertEqual(metrics['external_force'],0)
        self.assertLess(metrics['torque_fraction'],1.000001)
        with self.assertRaises(RuntimeError):env.step(np.zeros(5))
        print('ONLINE_BASELINE',json.dumps(metrics),flush=True)

    def test_envelope_has_zero_boundary_derivatives(self):
        np.testing.assert_array_equal(quintic(0),[0,0,0]);np.testing.assert_array_equal(quintic(1),[1,0,0])


if __name__=='__main__':unittest.main()
