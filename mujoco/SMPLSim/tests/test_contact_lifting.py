"""Arm FK, orientation goals and static gate; simulation evidence lives in CSV."""
import argparse,json,tempfile,unittest
from pathlib import Path
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from examples.m4_contact_lifting import ROOT,PREVIOUS,ARM,hand,arm_path,LiftingTargets,gate,input_signature
@unittest.skipUnless((PREVIOUS/'candidates.json').exists(),'D1 local grasp required')
class ContactLiftingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.out=Path(self.temp.name)
        self.args=argparse.Namespace(output=self.out,models=ROOT.parents[2]/'basketball-r1/SMPLSim/outputs/smplh_physics',source=ROOT/'outputs/m3_spl_fitting/spl_P0002_T0049.npz')
        m,d,self.b,a,q=hand.setup(self.args);self.m=self.b.model;self.d=mujoco.MjData(self.m)
        self.c=next(c for c in json.loads((PREVIOUS/'candidates.json').read_text()) if c['name']=='fk_fit_2')
        self.q=hand.GraspTargets(self.b,self.c).target(0);self.d.qpos[:]=self.q;mujoco.mj_forward(self.m,self.d)
    def tearDown(self):self.temp.cleanup()
    def test_arm_jacobian_matches_fk_position_and_rotation(self):
        w=self.m.body('R_Wrist').id;jp,jr=np.zeros((3,self.m.nv)),np.zeros((3,self.m.nv));mujoco.mj_jacBody(self.m,self.d,jp,jr,w)
        columns=[]
        for name in ARM:
            j=self.m.joint(name);adr=int(j.qposadr[0]);dof=int(j.dofadr[0]);ps=[];Rs=[]
            for eps in [-1e-6,1e-6]:
                self.d.qpos[:]=self.q;self.d.qpos[adr]+=eps;mujoco.mj_forward(self.m,self.d);ps.append(self.d.xpos[w].copy());Rs.append(self.d.xmat[w].reshape(3,3).copy())
            np.testing.assert_allclose((ps[1]-ps[0])/2e-6,jp[:,dof],atol=2e-9)
            np.testing.assert_allclose(Rotation.from_matrix(Rs[1]@Rs[0].T).as_rotvec()/2e-6,jr[:,dof],atol=2e-9)
            columns.append(np.r_[jp[:,dof],jr[:,dof]])
        self.assertEqual(np.linalg.matrix_rank(np.array(columns).T),6)
    def test_ik_raises_wrist_and_compensated_target_preserves_orientation(self):
        path,R,adrs=arm_path(self.m,self.q,.25,self.out);w=self.m.body('R_Wrist').id
        p0=self.d.xpos[w].copy();self.d.qpos[:]=path[-1];mujoco.mj_forward(self.m,self.d)
        self.assertAlmostEqual(self.d.xpos[w,2]-p0[2],.25,places=4)
        np.testing.assert_allclose(self.d.xmat[w].reshape(3,3),R,atol=1e-12)
        a=LiftingTargets(self.b,self.c,path,R,adrs,'C',1,4);target=a.target(5,self.d)
        self.d.qpos[:]=target;mujoco.mj_forward(self.m,self.d);np.testing.assert_allclose(self.d.xmat[w].reshape(3,3),R,atol=1e-12)
        self.assertGreater(np.linalg.norm(path[-1][adrs[:3]]-path[0][adrs[:3]]),.05)
        self.assertGreater(np.linalg.norm(path[-1][adrs[3:6]]-path[0][adrs[3:6]]),.05)
    def test_gate_requires_success_and_matching_source_inputs(self):
        with self.assertRaises(RuntimeError):gate(self.args)
        folder=self.out/'trials/fk_fit_2';folder.mkdir(parents=True)
        (folder/'summary.json').write_text(json.dumps(dict(held_3s=False,max_relative_drift_m=.008,warnings=[0],fallen=False,both_feet_ratio=1.,min_com_margin_m=.1)))
        (self.out/'static_inputs.json').write_text(json.dumps(input_signature(self.args)))
        with self.assertRaises(RuntimeError):gate(self.args)
        (folder/'summary.json').write_text(json.dumps(dict(held_3s=True,max_relative_drift_m=.008,warnings=[0],fallen=False,both_feet_ratio=1.,min_com_margin_m=.1)))
        gate(self.args)
        (self.out/'static_inputs.json').write_text('{}')
        with self.assertRaises(RuntimeError):gate(self.args)
if __name__=='__main__':unittest.main()
