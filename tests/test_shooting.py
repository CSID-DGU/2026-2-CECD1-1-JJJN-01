"""Loaded dynamics, continuous release, ballistic flight, and score direction."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from shoot_spl import (REFERENCE,HOOP,RADIUS,MASS,RIM_INNER,ScoreDetector,scene_xml,
                       loaded_control_model,shoot,motor_model)


class ScoreTests(unittest.TestCase):
    def point(self,z,x=0.):return HOOP+np.array([x,0,z])

    def test_requires_downward_entry_and_full_clearance(self):
        score=ScoreDetector()
        score.update(self.point(.1),self.point(-.02),0.)
        self.assertFalse(score.scored)
        score.update(self.point(-.02),self.point(-RADIUS-.01),.1)
        self.assertTrue(score.scored)

    def test_upward_and_side_entry_do_not_score(self):
        for before,after in [(self.point(-.2),self.point(.2)),
                             (self.point(-.2,.4),self.point(-.2)),
                             (self.point(.2,.3),self.point(-.2,.3))]:
            score=ScoreDetector();score.update(before,after,0.)
            self.assertFalse(score.scored)

    def test_ball_radius_and_high_speed_crossing(self):
        for offset,expected in [(0.,True),(RIM_INNER-RADIUS+.005,False)]:
            score=ScoreDetector();score.update(self.point(.3,offset),self.point(-.3,offset),0.)
            self.assertEqual(score.scored,expected)


class ObjectPhysicsTests(unittest.TestCase):
    def test_ball_mass_inertia_and_grasp_are_internal(self):
        xml=scene_xml();m=mujoco.MjModel.from_xml_string(xml)
        loaded=loaded_control_model(xml);empty=motor_model()
        self.assertAlmostEqual(loaded.body_mass.sum()-empty.body_mass.sum(),MASS,places=8)
        self.assertEqual(loaded.nv,empty.nv)
        self.assertEqual(m.nv,empty.nv+6)
        self.assertTrue(np.all(m.body('basketball').inertia>0))
        self.assertEqual(m.eq_obj1id[0],m.body('right_wrist').id)
        self.assertEqual(m.eq_obj2id[0],m.body('basketball').id)
        self.assertFalse(np.any(m.body_gravcomp))
        self.assertNotIn(m.joint('ball_free').id,m.actuator_trnid[:,0])
        self.assertNotIn(0,m.actuator_trnid[:,0])

    def test_actual_rim_collision_and_clean_drop(self):
        for offset in [0.,.20]:
            m=mujoco.MjModel.from_xml_string(scene_xml());d=mujoco.MjData(m)
            d.eq_active[:]=False;qa=m.joint('ball_free').qposadr[0];va=m.joint('ball_free').dofadr[0]
            d.qpos[qa:qa+3]=HOOP+[offset,0,.35];d.qvel[va+2]=-1.
            mujoco.mj_forward(m,d);score=ScoreDetector();rim_contact=False
            for _ in range(350):
                previous=d.qpos[qa:qa+3].copy();mujoco.mj_step(m,d)
                score.update(previous,d.qpos[qa:qa+3].copy(),d.time)
                rim_contact|=any((m.geom(c.geom1).name or '').startswith('rim_') or
                                 (m.geom(c.geom2).name or '').startswith('rim_') for c in d.contact)
            if offset==0:
                self.assertTrue(score.scored);self.assertFalse(rim_contact)
            else:
                self.assertTrue(rim_contact)
                self.assertGreater(abs(d.qvel[va]),.05)


@unittest.skipUnless(REFERENCE.exists(),'Local retargeted reference required')
class LoadedRolloutTests(unittest.TestCase):
    def test_reference_shot_release_ballistics_and_balance(self):
        with tempfile.TemporaryDirectory() as temp:
            with contextlib.redirect_stdout(io.StringIO()):report=shoot(REFERENCE,Path(temp))
            with np.load(Path(temp)/'shooting.npz') as data:
                t=data['ball_times'];v=data['ball_velocities'];p=data['ball_positions']
                release=report['release']['time_s']
                # Before any impact, motion must be gravity-only, without launch impulses.
                airborne=(t>release+.002)&(t<release+.15)
                dt=t[airborne]-release
                expected=np.array(report['release']['linear_velocity_m_s'])+dt[:,None]*[0,0,-9.81]
                np.testing.assert_allclose(v[airborne],expected,atol=1e-7)
                # Implicit Euler's position error relative to continuous ballistics is O(dt).
                expected_p=np.array(report['release']['position_m'])+dt[:,None]*np.array(report['release']['linear_velocity_m_s'])+dt[:,None]**2*[0,0,-4.905]
                np.testing.assert_allclose(p[airborne],expected_p,atol=.002)
                self.assertTrue(np.isfinite(data['qpos']).all())
        self.assertFalse(report['fell'])
        self.assertEqual(report['release']['qpos_change_at_release'],0.)
        self.assertEqual(report['release']['qvel_change_at_release'],0.)
        self.assertEqual(report['max_external_applied_force'],0.)
        self.assertEqual(report['qp_fallback_count'],0)
        self.assertFalse(any(report['mujoco_warning_counts']))
        self.assertLess(report['maximum_ankle_horizontal_displacement_m'],.02)
        self.assertLessEqual(report['maximum_torque_limit_fraction'],1.+1e-8)
        self.assertLess(report['max_grasp_position_error_m'],.003)
        self.assertLess(report['maximum_excluded_hand_ball_overlap_m'],.006)
        self.assertLess(report['maximum_held_ball_contact_penetration_m'],.003)
        self.assertLess(report['peak_grasp_force_n'],100.)
        self.assertLess(report['rms_error_m']['right_wrist'],.06)
        self.assertLess(report['rms_error_m']['left_wrist'],.07)


if __name__=='__main__':unittest.main()
