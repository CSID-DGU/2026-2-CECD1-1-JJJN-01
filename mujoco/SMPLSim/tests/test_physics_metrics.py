"""Contact material velocity and quaternion double-cover regression tests."""
import unittest
import numpy as np
import mujoco
from smpl_sim.motion.physics_metrics import foot_contact_slip, quaternion_error_deg


class PhysicsMetricsTest(unittest.TestCase):
    def test_quaternion_sign_and_known_rotation(self):
        self.assertAlmostEqual(quaternion_error_deg([1,0,0,0], [-1,0,0,0]), 0)
        self.assertAlmostEqual(quaternion_error_deg([1,0,0,0], [np.sqrt(.5),0,0,np.sqrt(.5)]), 90)

    def test_loaded_contact_translation_is_slip(self):
        m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
        <geom name="floor" type="plane" size="2 2 .1"/>
        <body name="L_Ankle" pos="0 0 .09"><freejoint/>
        <geom type="sphere" size=".1" mass="1"/>
        </body></worldbody></mujoco>''')
        d = mujoco.MjData(m)
        d.qvel[0] = .2
        mujoco.mj_forward(m, d)
        slip, points = foot_contact_slip(m, d)
        self.assertGreater(slip['L']['force'], .1)
        self.assertAlmostEqual(slip['L']['rms_speed'], .2, places=8)
        self.assertEqual(slip['R']['rms_speed'], 0)
        self.assertTrue(points)
        # Normal motion and a moving contact location alone do not count as sliding.
        d.qvel[:] = 0
        d.qvel[2] = -.2
        mujoco.mj_forward(m, d)
        slip, _ = foot_contact_slip(m, d)
        self.assertAlmostEqual(slip['L']['rms_speed'], 0, places=8)

    def test_airborne_has_no_loaded_contacts(self):
        m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
        <geom name="floor" type="plane" size="2 2 .1"/>
        <body name="R_Ankle" pos="0 0 1"><freejoint/>
        <geom type="sphere" size=".1" mass="1"/>
        </body></worldbody></mujoco>''')
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        slip, points = foot_contact_slip(m, d)
        self.assertEqual(points, [])
        self.assertEqual(slip['R']['force'], 0)


if __name__ == '__main__':
    unittest.main()
