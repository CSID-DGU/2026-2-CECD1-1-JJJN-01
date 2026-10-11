"""Trajectory properties independently of the physical contact hypothesis."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from examples.m5_shooting_trajectory import shooting_progress,so3_interpolate
class TestShootingTrajectory(unittest.TestCase):
    def test_simultaneous_vs_separated(self):
        linked=shooting_progress(1.5,3,0)
        self.assertAlmostEqual(linked['rise'],linked['rotation'])
        separate=shooting_progress(1.5,3,0,True)
        self.assertEqual(separate['rise'],0)
        self.assertGreater(separate['rotation'],0)
    def test_smooth_endpoint_and_lag(self):
        eps=1e-4
        self.assertLess(shooting_progress(eps,3)['rise']/eps,1e-6)
        self.assertEqual(shooting_progress(.1,3,.25)['rotation'],0)
        self.assertEqual(shooting_progress(4,3,.25)['rotation'],1)
    def test_so3_branch_and_orthogonality(self):
        a=Rotation.from_rotvec([0,0,3.1]).as_matrix();b=Rotation.from_rotvec([0,0,-3.1]).as_matrix()
        seq=[so3_interpolate(a,b,u) for u in np.linspace(0,1,100)]
        for r in seq:np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-12)
        self.assertLess(max(Rotation.from_matrix(y@x.T).magnitude() for x,y in zip(seq,seq[1:])),.001)
        np.testing.assert_allclose(seq[0],a,atol=1e-12);np.testing.assert_allclose(seq[-1],b,atol=1e-12)
    def test_duration_bounds(self):
        with self.assertRaises(ValueError):shooting_progress(0,0)
        with self.assertRaises(ValueError):shooting_progress(0,3,2)
