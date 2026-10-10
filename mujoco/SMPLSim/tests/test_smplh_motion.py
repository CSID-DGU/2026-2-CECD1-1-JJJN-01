"""Regression checks for rotation composition, input convention and interpolation."""
import tempfile
import unittest
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from smpl_sim.motion.smplh_motion import SMPLHMapping, SMPLHMotion, SMPL_TO_WORLD
from smpl_sim.smpllib.smpl_joint_names import SMPLH_BONE_ORDER_NAMES as NAMES

ROOT=Path(__file__).resolve().parents[1]


class MotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models=[mujoco.MjModel.from_xml_path(str(ROOT/f'outputs/smplh_physics/body_{i:02d}.xml')) for i in range(1,5)]

    def test_compound_fk_and_name_mapping_all_shapes(self):
        for m in self.models:
            mapping=SMPLHMapping(m)
            p=np.zeros((1,156))
            expected={}
            for name,angles in [('L_Shoulder',[.2,-.3,.4]),('R_Knee',[-.1,.2,.3]),('L_Index1',[.3,.1,-.2])]:
                i=NAMES.index(name)
                expected[name]=Rotation.from_euler('XYZ',angles).as_matrix()
                p[0,3*i:3*i+3]=Rotation.from_matrix(expected[name]).as_rotvec()
            p[0,:3]=[.2,.1,-.3]
            d=mujoco.MjData(m);d.qpos[:]=mapping.convert(p,[[.1,.2,.3]])[0]
            mujoco.mj_forward(m,d)
            np.testing.assert_allclose(d.xmat[1].reshape(3,3),(SMPL_TO_WORLD*Rotation.from_rotvec(p[0,:3])).as_matrix(),atol=1e-10)
            for name,rot in expected.items():
                b=m.body(name).id
                local=d.xmat[m.body_parentid[b]].reshape(3,3).T@d.xmat[b].reshape(3,3)
                np.testing.assert_allclose(local,rot,atol=1e-10)

    def test_common_angular_reference_all_shapes(self):
        from examples.m3_control import reference
        for kind in ('stand','arms','knees'):
            motions=[reference(kind,m) for m in self.models]
            for motion in motions[1:]:
                np.testing.assert_array_equal(motion.poses,motions[0].poses)
            # Shape-dependent collision clearance is explicit in root positions.
            self.assertGreater(np.ptp([motion.root_position[0,1] for motion in motions]),.1)

    def test_translation_conventions(self):
        m=self.models[0];mp=SMPLHMapping(m);p=np.zeros((1,156));t=np.array([[.1,.2,.3]])
        np.testing.assert_allclose(mp.convert(p,t)[0,:3],SMPL_TO_WORLD.apply(t)[0])
        np.testing.assert_allclose(mp.convert(p,t,'smpl','smpl_transl')[0,:3],SMPL_TO_WORLD.apply(t+m.body_pos[1])[0])
        np.testing.assert_allclose(mp.convert(p,t,'mujoco')[0,:3],t[0])
        np.testing.assert_allclose(mp.convert(p,t,'mujoco')[0,3:7],[1,0,0,0])

    def test_npz_sampling_and_quaternion_normalization(self):
        m=self.models[0]
        p=np.zeros((2,156));p[1,:3]=[0,.4,0];p[1,3*NAMES.index('R_Knee')]=.6
        motion=SMPLHMotion(p,np.array([[0,0,0],[1,2,3]]),2)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'sequence.npz';motion.save(path);loaded=SMPLHMotion.load(path)
            bound=loaded.bind(m);mid=bound.sample(.25)
            np.testing.assert_allclose(mid[:3],SMPL_TO_WORLD.apply([.5,1,1.5]))
            self.assertAlmostEqual(np.linalg.norm(mid[3:7]),1)
            self.assertAlmostEqual(mid[m.joint('R_Knee_x').qposadr[0]],.3)
            np.testing.assert_allclose(bound.sample(-1),bound.qpos[0])
            np.testing.assert_allclose(bound.sample(10),bound.qpos[-1])

    def test_nonfinite_wrong_shape_and_gimbal_rejected(self):
        with self.assertRaises(ValueError):SMPLHMotion(np.zeros((2,155)),np.zeros((2,3)),30)
        with self.assertRaises(ValueError):SMPLHMotion(np.zeros((2,156)),np.zeros((2,3)),0)
        bad=np.zeros((2,156));bad[1,3]=np.nan
        with self.assertRaises(ValueError):SMPLHMotion(bad,np.zeros((2,3)),30)
        p=np.zeros((1,156));p[0,3*NAMES.index('L_Shoulder')+1]=np.pi/2
        with self.assertRaises(ValueError):SMPLHMapping(self.models[0]).convert(p,np.zeros((1,3)))


if __name__=='__main__':unittest.main()
