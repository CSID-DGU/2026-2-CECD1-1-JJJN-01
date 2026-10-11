import json,tempfile,unittest
from pathlib import Path
import numpy as np,mujoco
from examples.m4_motion_pose_editor import PoseEdits,OUT
class EditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        f=OUT/'A/C_1';cls.m=mujoco.MjModel.from_xml_path((f/'scene_path.txt').read_text());cls.z=np.load(f/'states.npz')
    def test_interpolation_has_unit_quaternions_and_does_not_change_records(self):
        e=PoseEdits(self.m,self.z['qpos'],self.z['time_s']);before=e.states.copy()
        for frame in [0,90,108,112,239]:
            q=e.base(frame);self.assertTrue(np.isfinite(q).all());self.assertAlmostEqual(np.linalg.norm(q[3:7]),1,places=12);self.assertAlmostEqual(np.linalg.norm(q[-4:]),1,places=12)
        np.testing.assert_array_equal(e.states,before)
    def test_edits_are_per_frame_and_preserve_root_ball_and_other_joints(self):
        e=PoseEdits(self.m,self.z['qpos'],self.z['time_s']);q=e.pose();q[e.ids[0]]+=.1;e.remember(q);np.testing.assert_allclose(e.pose(),q,atol=1e-12);e.frame+=1;np.testing.assert_array_equal(e.pose(),e.base(e.frame))
    def test_saves_separate_offsets_and_never_overwrites(self):
        e=PoseEdits(self.m,self.z['qpos'],self.z['time_s']);e.remember(e.pose())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'edits.json';a=e.save(p,{'condition':'A'});b=e.save(p,{'condition':'A'});self.assertNotEqual(a,b);self.assertFalse(p.exists());self.assertFalse(json.loads(a.read_text())['physics_verified'])
if __name__=='__main__':unittest.main()
