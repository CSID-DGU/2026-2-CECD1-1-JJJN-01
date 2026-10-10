"""Synthetic trajectory contract: duration, phases, continuity and SMPLH NPZ."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from smpl_sim.motion import SMPLHMotion
from smpl_sim.motion.synthetic_free_throw import generate, stage_at, transition
from smpl_sim.smpllib.smpl_joint_names import SMPLH_BONE_ORDER_NAMES as NAMES


class SyntheticMotionTests(unittest.TestCase):
    def setUp(self):
        self.motion = generate([0., .975, 0.])
        self.euler = Rotation.from_rotvec(self.motion.poses.reshape(-1, 3)).as_euler('XYZ').reshape(91, 52, 3)

    def angle(self, t, body, axis):
        return self.euler[round(t*30), NAMES.index(body), axis]

    def test_npz_contract_and_exact_duration(self):
        self.assertEqual(self.motion.poses.shape, (91,156))
        self.assertEqual(self.motion.fps,30)
        self.assertEqual(self.motion.duration,3.)
        np.testing.assert_array_equal(self.motion.root_position,np.tile([0.,.975,0.],(91,1)))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'synthetic.npz';self.motion.save(path)
            loaded=SMPLHMotion.load(path)
            np.testing.assert_array_equal(loaded.poses,self.motion.poses)
            self.assertEqual(loaded.coordinate_system,'smpl')

    def test_requested_order_and_returning_knee(self):
        self.assertAlmostEqual(self.angle(0,'L_Knee',0),.1)
        # The sampled peak surrounds 0.85 seconds, between two source samples.
        self.assertGreater(self.angle(.8333333333,'L_Knee',0),.349)
        self.assertAlmostEqual(self.angle(1.3,'L_Knee',0),.1)
        self.assertAlmostEqual(self.angle(1.3,'L_Shoulder',0),0.)
        self.assertAlmostEqual(self.angle(1.9666666667,'L_Elbow',1),-.9,places=3)
        self.assertAlmostEqual(self.angle(2.5,'L_Elbow',1),-.1)
        self.assertAlmostEqual(self.angle(2.5,'L_Wrist',0),-.3)
        self.assertGreater(self.angle(2.7666666667,'L_Wrist',0),.19)
        self.assertAlmostEqual(self.angle(3,'L_Wrist',0),.1)
        self.assertEqual(stage_at(.35),'knee_bend')
        self.assertEqual(stage_at(1.3),'arm_raise')
        self.assertEqual(stage_at(3),'wrist_flick')

    def test_smooth_boundaries_and_no_pose_jump(self):
        for start,end in [(0.35,.85),(.85,1.3),(1.3,1.95),(1.95,2.5),(2.5,2.75),(2.75,3)]:
            self.assertEqual(transition(start,start,end),0.)
            self.assertEqual(transition(end,start,end),1.)
            eps=1e-5
            self.assertLess(abs(float(transition(start+eps,start,end)))/eps,1e-6)
            self.assertLess(abs(1-float(transition(end-eps,start,end)))/eps,1e-6)
        self.assertLess(np.abs(np.diff(self.motion.poses,axis=0)).max(),.2)
        with self.assertRaises(ValueError):generate([0.,np.nan,0.])


if __name__=='__main__':unittest.main()
