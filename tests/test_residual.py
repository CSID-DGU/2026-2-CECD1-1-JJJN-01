"""Derived axes, physical contact, and constrained residual action validation."""
import sys,json,unittest,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from shooting_model import contact_motor_model,expand_reference
from track_spl import motor_model
from contact_shot import contact_scene,ball_reaction
from contact_reference import GRIP_OFFSET
from prepare_residual import residual_reference,gates


class ShootingModelTests(unittest.TestCase):
    def test_roll_axis_preserves_segment_endpoints_and_mass(self):
        old=motor_model();m=contact_motor_model();d=mujoco.MjData(m)
        self.assertEqual((m.nq,m.nv,m.nu),(102,101,95))
        self.assertAlmostEqual(sum(m.body_mass),sum(old.body_mass),places=8)
        self.assertTrue(np.all(m.jnt_dofadr[m.actuator_trnid[:,0]]>=6))
        d.qpos[:3]=[0,0,1];d.qpos[3]=1;mujoco.mj_forward(m,d)
        p=d.body('right_wrist').xpos.copy();e=d.body('right_elbow').xpos.copy();rot=d.body('right_wrist').xmat.copy()
        j=m.joint('right_forearm_roll').id;d.qpos[m.jnt_qposadr[j]]=.5;mujoco.mj_forward(m,d)
        np.testing.assert_allclose(d.body('right_wrist').xpos,p,atol=1e-10)
        np.testing.assert_allclose(d.body('right_elbow').xpos,e,atol=1e-10)
        self.assertGreater(np.linalg.norm(d.body('right_wrist').xmat-rot),.1)
        a=m.actuator('right_forearm_roll_motor').id
        np.testing.assert_array_equal(m.actuator_forcerange[a],[-8,8])

    def test_reference_transfer_preserves_forward_kinematics(self):
        ref=dict(np.load(ROOT/'data/motions/processed/free_throw_reference.npz'))
        old=motor_model();m=contact_motor_model();new=expand_reference(ref,m)
        a=mujoco.MjData(old);b=mujoco.MjData(m)
        for k in [0,50,100]:
            a.qpos[:]=ref['qpos'][k];b.qpos[:]=new['qpos'][k];mujoco.mj_forward(old,a);mujoco.mj_forward(m,b)
            for name in ['right_wrist','left_wrist','left_ankle','right_ankle']:
                np.testing.assert_allclose(a.body(name).xpos,b.body(name).xpos,atol=1e-10)

    def test_hands_retain_ball_collision_pairs(self):
        m=mujoco.MjModel.from_xml_string(contact_scene());ball=m.geom('ball_geom').id
        hand=[g for g in range(m.ngeom) if m.geom_contype[g]==4]
        self.assertGreater(len(hand),30)
        for g in hand:
            self.assertTrue((m.geom_contype[g]&m.geom_conaffinity[ball]) or (m.geom_contype[ball]&m.geom_conaffinity[g]))
            self.assertTrue(any(set(pair)=={g,ball} for pair in zip(m.pair_geom1,m.pair_geom2)))
        self.assertFalse(any(m.body(m.jnt_bodyid[j]).name=='basketball' for j in m.actuator_trnid[:,0]))

    def test_contact_reaction_matches_solver_without_applying_forces(self):
        tree=ET.fromstring(contact_scene())
        # Keep only explicit hand-ball pairs, isolating their generalized load.
        for g in tree.iter('geom'):g.set('contype','0');g.set('conaffinity','0')
        m=mujoco.MjModel.from_xml_string(ET.tostring(tree,encoding='unicode'));d=mujoco.MjData(m)
        ref=np.load(ROOT/'runs/spl_residual_ready/nominal_reference.npz');q=ref['qpos'][0];nq=len(q);nv=m.nv-6
        d.qpos[:nq]=q;d.eq_active[:]=False;mujoco.mj_forward(m,d)
        d.qpos[nq:nq+3]=d.body('right_wrist').xpos+d.body('right_wrist').xmat.reshape(3,3)@GRIP_OFFSET
        d.qpos[nq+3:nq+7]=d.body('right_wrist').xquat;mujoco.mj_forward(m,d)
        reaction=ball_reaction(m,d,m.geom('ball_geom').id)
        ef=np.where(d.efc_type>=int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS),d.efc_force,0.)
        expected=np.zeros(m.nv);mujoco.mj_mulJacTVec(m,d,expected,ef)
        self.assertGreater(np.linalg.norm(reaction[:nv]),.01)
        np.testing.assert_allclose(reaction[:nv],expected[:nv],atol=1e-8)
        np.testing.assert_array_equal(d.qfrc_applied,np.zeros(m.nv))
        np.testing.assert_array_equal(d.xfrc_applied,np.zeros_like(d.xfrc_applied))


class ResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out=ROOT/'runs/spl_residual_ready'
        cls.ref=dict(np.load(cls.out/'nominal_reference.npz'))
        cls.meta=json.loads((cls.out/'nominal_reference.json').read_text())['adaptation']

    def test_zero_action_is_exact_and_invalid_actions_rejected(self):
        ref,_,detach=residual_reference(self.ref,self.meta,np.zeros(5))
        for key in self.ref:np.testing.assert_array_equal(ref[key],self.ref[key])
        self.assertEqual(detach,.99)
        for action in [[0]*4,[0,0,np.nan,0,0],[0,0,0,0,1.1]]:
            with self.assertRaises(ValueError):residual_reference(self.ref,self.meta,action)

    def test_residual_cannot_change_root_legs_or_followthrough(self):
        ref,_,detach=residual_reference(self.ref,self.meta,np.ones(5))
        self.assertAlmostEqual(detach,1.002)
        m=contact_motor_model();allowed=['right_elbow_z','right_wrist_x','right_forearm_roll']
        for j in range(m.njnt):
            name=m.joint(j).name
            if name in allowed or (name.startswith('right_') and name.endswith('_x') and any(f in name for f in ['index','middle','ring','pinky'])):continue
            qa,va=m.jnt_qposadr[j],m.jnt_dofadr[j];nq,nv=(7,6) if j==0 else (1,1)
            for key,start,size in [('qpos',qa,nq),('qvel',va,nv),('qacc',va,nv)]:
                np.testing.assert_array_equal(ref[key][:,start:start+size],self.ref[key][:,start:start+size])
        np.testing.assert_allclose(ref['qpos'][-1],self.ref['qpos'][-1],atol=1e-12)
        for j in range(1,m.njnt):
            self.assertTrue(np.all(ref['qpos'][:,m.jnt_qposadr[j]]>=m.jnt_range[j,0]-1e-9))
            self.assertTrue(np.all(ref['qpos'][:,m.jnt_qposadr[j]]<=m.jnt_range[j,1]+1e-9))

    def test_rollouts_pass_gates_and_small_residual_changes_trajectory(self):
        summary=json.loads((self.out/'readiness.json').read_text())
        self.assertFalse(summary['training_started'])
        self.assertTrue(summary['physics_ready'])
        self.assertTrue(all(all(r['gates'].values()) for r in summary['runs']))
        self.assertTrue(summary['trajectory_sensitivity_observed'])
        report=json.loads((self.out/'shooting_report.json').read_text())
        self.assertTrue(all(gates(report).values()))

    def test_free_flight_matches_gravity_without_velocity_injection(self):
        report=json.loads((self.out/'shooting_report.json').read_text());run=np.load(self.out/'shooting.npz')
        t=run['ball_times'];mask=(t>report['release']['time_s']+.03)&(t<report['release']['time_s']+.20)
        dv=np.diff(run['ball_velocities'][mask],axis=0)/np.diff(t[mask])[:,None]
        np.testing.assert_allclose(dv.mean(axis=0),[0,0,-9.81],atol=.01)
        self.assertEqual(report['release']['qvel_change_at_release'],0.)


if __name__=='__main__':unittest.main()
