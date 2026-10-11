"""Overhand preparation geometry, unchanged body reference, and physical rollout."""
import sys,json,unittest
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from free_throw_env import FreeThrowEnv
from shooting_model import contact_motor_model


class OverhandPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        choices=[]
        for name in ['spl_overhand_v2','spl_overhand_v3']:
            path=ROOT/'runs'/name;review=path/'preparation_review.json'
            if review.exists():
                report=json.loads(review.read_text(encoding='utf-8'))
                if report['physics_ready']:choices.append((report['apex_m'],path))
        if not choices:raise unittest.SkipTest('No physically accepted overhand candidate')
        cls.path=max(choices,key=lambda pair:pair[0])[1];cls.ref=dict(np.load(cls.path/'nominal_reference.npz'))

    def test_starts_above_ball_and_removes_late_heading_spin(self):
        env=FreeThrowEnv(self.path)
        self.assertGreater(env.d.site('right_palm').xpos[2]-env.d.qpos[env.nq+2],.10)
        self.assertLess(env.d.body('right_wrist').xmat.reshape(3,3)[2,2],-.90)
        baseline=np.load(ROOT/'runs/spl_residual_ready/nominal_reference.npz');m=contact_motor_model();d=mujoco.MjData(m)
        def late_spin(ref):
            mask=(ref['times']>=.6)&(ref['times']<=.88);rot=[]
            for q in ref['qpos'][mask]:
                d.qpos[:]=q;mujoco.mj_kinematics(m,d);rot.append(d.body('right_wrist').xmat.reshape(3,3).copy())
            rot=np.array(rot);angular=Rotation.from_matrix(rot[1:]@rot[:-1].transpose(0,2,1)).as_rotvec()/np.diff(ref['times'][mask])[:,None]
            return float(np.max(abs(angular[:,2])))
        old=late_spin(baseline);new=late_spin(self.ref)
        self.assertLess(new,old*.25)
        print('LATE_HEADING_SPIN_RAD_S',old,new,flush=True)

    def test_torso_legs_and_finger_targets_are_preserved(self):
        old=np.load(ROOT/'runs/spl_residual_ready/nominal_reference.npz');m=contact_motor_model();protected=list(range(7))
        for j in range(1,m.njnt):
            if not any(part in m.joint(j).name for part in ['shoulder_','elbow_','wrist_','forearm_']):protected.append(m.jnt_qposadr[j])
        np.testing.assert_allclose(self.ref['qpos'][:,protected],old['qpos'][:,protected],atol=1e-12)

    def test_online_baseline_passes_physics_gates(self):
        env=FreeThrowEnv(self.path);done=False
        while not done:_,_,done,_,info=env.step(np.zeros(5))
        self.assertTrue(info['metrics']['accepted'],info['metrics'])
        baseline=json.loads((self.path/'shooting_report.json').read_text(encoding='utf-8'))
        np.testing.assert_allclose(info['metrics']['release_velocity_m_s'],baseline['release']['linear_velocity_m_s'],atol=1e-8)


if __name__=='__main__':unittest.main()
