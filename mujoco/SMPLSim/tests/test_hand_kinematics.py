"""FK/contact invariants, independent of claims of physical grasp success."""
import tempfile, unittest
from pathlib import Path
import mujoco
import numpy as np
from examples.m4_hand_kinematics import capsule,closest,tip,measure,GraspTargets
from examples.m4_smpl_contact import build_scene,ContactTargets
from smpl_sim.physics.basketball import HumanoidControllerBridge
ROOT=Path(__file__).resolve().parents[1]
MODELS=ROOT.parents[2]/'basketball-r1/SMPLSim/outputs/smplh_physics'
SOURCE=ROOT/'outputs/m3_spl_fitting/spl_P0002_T0049.npz'
@unittest.skipUnless((MODELS/'body_01.xml').exists() and SOURCE.exists(),'Local assets required')
class HandKinematicsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.out=Path(self.temp.name)
        self.m=mujoco.MjModel.from_xml_path(str(build_scene(MODELS/'body_01.xml',self.out/'scene.xml')))
        self.d=mujoco.MjData(self.m);self.b=HumanoidControllerBridge(MODELS/'body_01.xml',self.m,SOURCE)
        self.a=ContactTargets(self.b,'hold',0);self.b.initialize(self.m,self.d)
        self.q=self.a.palm_up(self.a.base.copy());self.d.qpos[:len(self.q)]=self.q;mujoco.mj_forward(self.m,self.d)
    def tearDown(self):self.temp.cleanup()
    def test_fk_central_difference_matches_spatial_hinge_axis_all_45(self):
        from examples.m3_control import FINGERS
        for f in FINGERS:
            p=tip(self.m,self.d,f).copy()
            for part in (1,2,3):
                for ax in 'xyz':
                    j=self.m.joint(f'R_{f}{part}_{ax}').id;adr=int(self.m.jnt_qposadr[j]);expected=np.cross(self.d.xaxis[j],p-self.d.xanchor[j]);points=[]
                    for eps in [-1e-6,1e-6]:
                        self.d.qpos[:len(self.q)]=self.q;self.d.qpos[adr]+=eps;mujoco.mj_forward(self.m,self.d);points.append(tip(self.m,self.d,f).copy())
                    np.testing.assert_allclose((points[1]-points[0])/2e-6,expected,atol=2e-9)
                    self.d.qpos[:len(self.q)]=self.q;mujoco.mj_forward(self.m,self.d)
    def test_capsule_endpoints_and_segment_clamp(self):
        a,b,r=capsule(self.m,self.d,self.m.geom('R_Index1').id)
        self.assertGreater(r,0);np.testing.assert_allclose(closest(a-(b-a),a,b),a);np.testing.assert_allclose(closest(b+(b-a),a,b),b)
    def test_measured_mapping_is_not_uniform_z_and_restores_state(self):
        mapping,joints=measure(self.m,self.d,self.q,self.out)
        self.assertEqual(len(set(joints)),45);self.assertEqual(len(mapping),45)
        self.assertEqual(next(r['axis'] for r in mapping if r['segment']=='R_Ring3' and r['role']=='flexion'),'x')
        np.testing.assert_array_equal(self.d.qpos[:len(self.q)],self.q)
if __name__=='__main__':unittest.main()
