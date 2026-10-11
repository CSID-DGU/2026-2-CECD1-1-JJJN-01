"""SO(3) decomposition, observable-position null motion and adapter isolation."""
import unittest
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from examples.m4_forearm_rotation import ROOT,OUT,NAMES,apply_twist,transported_roll,CorrectedBridge,dyn
class RotationTests(unittest.TestCase):
    def test_pure_roll_and_pure_swing_separated(self):
        angles=np.linspace(0,4,100);R=Rotation.from_rotvec(np.c_[np.zeros((100,2)),angles]).as_matrix();a=np.tile([0,0,1.],(100,1));roll,_=transported_roll(R,a);np.testing.assert_allclose(roll,angles,atol=1e-12)
        R=Rotation.from_rotvec(np.c_[angles,np.zeros((100,2))]).as_matrix();a=R@np.array([0,0,1.]);roll,_=transported_roll(R,a);np.testing.assert_allclose(roll,0,atol=1e-12)
    def test_antiparallel_transport_rejected(self):
        with self.assertRaises(ValueError):transported_roll(np.tile(np.eye(3),(2,1,1)),np.array([[0,0,1.],[0,0,-1.]]))
class AssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT.parents[2]/'basketball-r1/SMPLSim/outputs/smplh_physics/body_01.xml';cls.m=mujoco.MjModel.from_xml_path(str(path));cls.b=dyn.prior.HumanoidControllerBridge(path,cls.m,ROOT/'outputs/m3_spl_fitting/spl_P0002_T0049.npz')
    def test_elbow_twist_preserves_centers_and_changes_orientation(self):
        d=mujoco.MjData(self.m)
        for t in [1.,1.5,1.85,2.,2.5]:
            q=self.b.target(t);d.qpos[:]=q;mujoco.mj_forward(self.m,d);p=np.array([d.xpos[self.m.body(n).id].copy() for n in NAMES]);R=d.xmat[self.m.body('R_Elbow').id].reshape(3,3).copy()
            for angle in [-.5,.5]:
                x=apply_twist(self.m,d,q,angle);np.testing.assert_allclose([d.xpos[self.m.body(n).id] for n in NAMES],p,atol=1e-12);self.assertAlmostEqual(Rotation.from_matrix(R.T@d.xmat[self.m.body('R_Elbow').id].reshape(3,3)).magnitude(),abs(angle),places=10)
    def test_corrected_elbow_targets_stay_within_xml_limits(self):
        ids=[int(self.m.joint('R_Elbow_'+k).qposadr[0]) for k in 'xyz'];bounds=np.array([self.m.jnt_range[self.m.joint('R_Elbow_'+k).id] for k in 'xyz'])
        for mode in ['B','C']:
            q=np.load(OUT/(mode+'_targets.npz'))['qpos'][:,ids];self.assertTrue(np.all(q>=bounds[:,0]));self.assertTrue(np.all(q<=bounds[:,1]))
    def test_interpolated_adapter_preserves_positions_inputs_and_zero_window(self):
        d=mujoco.MjData(self.m);pose=self.b.motion.poses.copy();original=self.b.data.qpos.copy()
        for mode in ['B','C']:
            proxy=CorrectedBridge(self.b,OUT/(mode+'_targets.npz'))
            for t in [.4,1.017,1.853,2.113,3.1]:
                q=self.b.target(t);x=proxy.target(t);d.qpos[:]=q;mujoco.mj_forward(self.m,d);p=np.array([d.xpos[self.m.body(n).id].copy() for n in NAMES]);d.qpos[:]=x;mujoco.mj_forward(self.m,d);np.testing.assert_allclose([d.xpos[self.m.body(n).id] for n in NAMES],p,atol=1e-12)
                if t<1 or t>2.75:np.testing.assert_array_equal(q,x)
        np.testing.assert_array_equal(pose,self.b.motion.poses);np.testing.assert_array_equal(original,self.b.data.qpos)
if __name__=='__main__':unittest.main()
