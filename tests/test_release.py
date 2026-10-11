"""Source-driven hand-frame fitting, form bounds, and loaded physical tracking."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,contextlib,io,tempfile,unittest
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from shoot_spl import REFERENCE,motor_model,prepare_reference
from release_reference import hand_rotations,envelope,corrected_reference
from calibrate_release import source_release,run
from track_spl import reference_at


class PalmFrameTests(unittest.TestCase):
    def test_recovers_proper_rotation_independent_of_landmark_lengths(self):
        m=motor_model();d=mujoco.MjData(m);d.qpos[:]=m.key('standing').qpos;mujoco.mj_forward(m,d)
        wrist=d.body('right_wrist');rot=Rotation.from_rotvec([.2,-.4,.8]).as_matrix()
        points={'RIGHT_WRIST':np.array([[2.,3.,4.]])}
        for scale,body,key in [(1.1,'right_index1','RIGHT_SECOND_FINGER_MCP'),
                    (.9,'right_middle1','RIGHT_THIRD_FINGER_MCP'),(1.3,'right_pinky1','RIGHT_FIFTH_FINGER_MCP')]:
            local=wrist.xmat.reshape(3,3).T@(d.body(body).xpos-wrist.xpos)
            points[key]=points['RIGHT_WRIST']+(rot@local)*scale
        fit=hand_rotations(points,np.eye(3),m)[0]
        np.testing.assert_allclose(fit,rot,atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(fit),1.)
        points['RIGHT_SECOND_FINGER_MCP']=points['RIGHT_WRIST'].copy()
        with self.assertRaisesRegex(ValueError,'Degenerate'):hand_rotations(points,np.eye(3),m)

    def test_window_returns_to_original_with_zero_edge_derivatives(self):
        start=.5;release=1.;end=1.5;h=1e-5
        self.assertEqual(envelope(start-.1,start,release,end),0.)
        self.assertEqual(envelope(end+.1,start,release,end),0.)
        self.assertEqual(envelope(release,start,release,end),1.)
        for t in [start,end]:
            f=lambda x:envelope(x,start,release,end)
            self.assertLess(abs((f(t+h)-f(t-h))/(2*h)),1e-6)
            self.assertLess(abs((f(t+h)-2*f(t)+f(t-h))/h**2),.03)


@unittest.skipUnless(REFERENCE.exists(),'Local SPL reference required')
class ReleaseReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.info=json.loads(REFERENCE.with_suffix('.json').read_text(encoding='utf-8'))
        cls.diagnostic,cls.source=source_release(cls.info)
        cls.release=cls.diagnostic['release_motion_time_s']
        cls.base,_,_,_=prepare_reference(REFERENCE,release_motion=cls.release)
        cls.ref,task,cls.meta=corrected_reference(cls.base,cls.release,cls.source['motion_times'],cls.source['palm_rotation'])
        cls.task=staticmethod(task)

    def test_source_hash_change_is_rejected(self):
        info=json.loads(json.dumps(self.info));info['selection']['source_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'Source changed'):source_release(info)

    def test_source_units_and_ball_wrist_targets_are_distinct(self):
        self.assertEqual(self.diagnostic['estimated_release_frame'],149)
        self.assertGreater(self.diagnostic['aligned_target_speed_m_s'],6.)
        self.assertLess(self.diagnostic['aligned_target_speed_m_s'],7.)
        self.assertLess(self.diagnostic['source_wrist_speed_m_s'],1.5)
        self.assertTrue(self.meta['ball_speed_is_not_an_arm_target'])
        self.assertLess(np.linalg.norm(self.meta['source_adapted_wrist_velocity_m_s']),1.)
        for rotation in self.source['palm_rotation'][::30]:
            np.testing.assert_allclose(rotation.T@rotation,np.eye(3),atol=1e-10)

    def test_correction_preserves_other_joints_and_bounded_form(self):
        m=motor_model();right=[j for j in range(m.njnt) if m.joint(j).name.startswith(('right_shoulder_','right_elbow_','right_wrist_'))]
        adr=m.jnt_qposadr[right];keep=np.setdiff1d(np.arange(m.nq),adr)
        reach=np.array(self.meta['right_arm_delta_bounds_rad'])
        for t,q in zip(self.ref['times'],self.ref['qpos']):
            base=reference_at(self.base,t)[0]
            np.testing.assert_allclose(q[keep],base[keep],atol=1e-12)
            self.assertTrue(np.all(abs(q[adr]-base[adr])<=reach+1e-10))
            lo,hi=self.meta['correction_window_s']
            if t<lo or t>hi:np.testing.assert_allclose(q,base,atol=1e-12)
        self.assertLess(self.meta['maximum_reference_joint_clip_rad'],1e-8)
        # Final task holds the original follow-through rather than extrapolating.
        task=self.task(float(self.ref['times'][-1])+2.)
        np.testing.assert_array_equal(task['velocity'],np.zeros(3))
        np.testing.assert_array_equal(task['acceleration'],np.zeros(3))

    def test_physical_correction_passes_tracking_and_continuous_release(self):
        with tempfile.TemporaryDirectory() as temp:
            with contextlib.redirect_stdout(io.StringIO()):report=run(Path(temp))
            self.assertTrue(report['release_diagnostics']['tracking_accepted'],report['release_diagnostics']['checks'])
            self.assertLess(report['release_diagnostics']['reference_wrist_velocity_error_m_s'],.3)
            self.assertFalse(report['release_diagnostics']['source_speed_reproduced_within_10pct'])
            with np.load(Path(temp)/'shooting.npz') as z:
                release=report['release']['time_s'];mask=(z['ball_times']>release+.002)&(z['ball_times']<release+.12)
                dt=z['ball_times'][mask]-release
                expected=np.array(report['release']['linear_velocity_m_s'])+dt[:,None]*[0,0,-9.81]
                np.testing.assert_allclose(z['ball_velocities'][mask],expected,atol=1e-7)


if __name__=='__main__':unittest.main()
