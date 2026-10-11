"""Integration tests of the generated physics model; no policy training."""
from pathlib import Path
import json
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from humanoid_model import MODEL_PATH, load_model, reset_pose
from check_humanoid import run_checks


class HumanoidPhysicsTests(unittest.TestCase):
    def setUp(self):
        self.model = load_model()
        self.data = mujoco.MjData(self.model)
        reset_pose(self.model, self.data, 't_pose')

    def test_rest_joint_positions_match_smpl(self):
        metadata = json.loads(MODEL_PATH.with_suffix('.json').read_text(encoding='utf-8'))
        for record in metadata['joint_mapping']:
            np.testing.assert_allclose(self.data.body(record['body']).xpos,
                                       record['rest_joint_world_m'], atol=2e-7)
        self.assertAlmostEqual(float(self.model.body_mass.sum()), 75., places=5)
        self.assertEqual(self.model.nv - self.model.nu, 6)
        self.assertTrue(np.all(self.model.jnt_limited[1:]))
        self.assertTrue(np.all(self.model.actuator_forcelimited))

    def test_unactuated_root_falls_under_gravity(self):
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        self.data.qpos[2] += 3.
        mujoco.mj_forward(self.model, self.data)
        start = self.data.subtree_com[1].copy()
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        displacement = self.data.subtree_com[1] - start
        self.assertAlmostEqual(displacement[2], -.5 * 9.81 * self.data.time ** 2, delta=.001)
        np.testing.assert_allclose(displacement[:2], 0, atol=.0001)
        self.assertEqual(self.model.neq, 0)
        self.assertFalse(np.any(self.model.body_gravcomp))

    def test_leg_arm_wrist_and_each_hand_respond_to_pd_targets(self):
        self.model.opt.gravity[:] = 0
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        targets = {'left_knee_y': .3, 'left_elbow_z': -.4,
                   'right_wrist_y': .25, 'left_index1_x': .4, 'right_index1_x': -.4}
        for name, target in targets.items():
            with self.subTest(joint=name):
                reset_pose(self.model, self.data, 't_pose')
                self.data.ctrl[self.model.actuator(name + '_servo').id] = target
                for _ in range(500):
                    mujoco.mj_step(self.model, self.data)
                position = self.data.qpos[self.model.joint(name).qposadr[0]]
                self.assertLess(abs(position - target), abs(target) * .25)
                self.assertFalse(np.any(self.data.warning.number))

    def test_both_palms_can_contact_external_objects(self):
        for side in ['left', 'right']:
            with self.subTest(side=side):
                palm = self.data.site(side + '_palm').xpos.copy()
                palm[2] += .035
                xml = ET.fromstring(MODEL_PATH.read_text(encoding='utf-8'))
                # Added DOFs change keyframe dimensions, so omit keyframes in this fixture.
                xml.remove(xml.find('keyframe'))
                body = ET.SubElement(xml.find('worldbody'), 'body', name='test_ball',
                                     pos=' '.join(map(str, palm)))
                ET.SubElement(body, 'freejoint')
                ET.SubElement(body, 'geom', name='test_ball_geom', type='sphere', size='.03', mass='.1')
                model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)
                ball_id, palm_id = model.geom('test_ball_geom').id, model.geom(side + '_wrist_geom').id
                pairs = [{contact.geom1, contact.geom2} for contact in data.contact]
                self.assertIn({ball_id, palm_id}, pairs)

    def test_contact_rollout_has_no_numerical_warnings(self):
        report = run_checks(seconds=2.)
        self.assertGreater(report['peak_individual_foot_contact_force_n'], 0)
        self.assertLess(report['deepest_contact_m'], .02)

    def test_half_timestep_remains_finite_after_fall(self):
        self.model.opt.timestep *= .5
        reset_pose(self.model, self.data)
        for _ in range(4000):
            mujoco.mj_step(self.model, self.data)
        self.assertTrue(np.isfinite(self.data.qpos).all())
        self.assertTrue(np.isfinite(self.data.qvel).all())
        self.assertFalse(np.any(self.data.warning.number))
        self.assertAlmostEqual(self.data.time, 2., places=7)


if __name__ == '__main__':
    unittest.main()
