"""SPL parsing, reference validity, and an independently rerun dynamics rollout."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json
import sys
import tempfile
import unittest
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from spl_motion import read_trial,KEYS
from humanoid_model import load_model
from retarget_spl import retarget
from track_spl import track,motor_model,reference_at

REF=ROOT/'data/motions/processed/free_throw_reference.npz'


class ParsingTests(unittest.TestCase):
    def fixture(self):
        return {'sampling_rate':60,'tracking':[{'frame':i,'time':int(i*1000/60),
                'data':{'ball':[1,2,3],'player':{k:[1,2,3] for k in KEYS}}} for i in range(35)]}

    def test_feet_to_metres_and_frame_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'clip.json';file.write_text(json.dumps(self.fixture()))
            _,fps,ids,t,p,missing=read_trial(file)
            np.testing.assert_allclose(p['LEFT_WRIST'][0],[.3048,.6096,.9144])
            self.assertAlmostEqual(t[1],1/60)
            self.assertEqual(fps,60)

    def test_missing_samples_interpolate_without_hiding_missingness(self):
        raw=self.fixture();raw['tracking'][10]['data']['player']['LEFT_WRIST']=[None,None,None]
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'clip.json';file.write_text(json.dumps(raw))
            *_,p,missing=read_trial(file)
            self.assertGreater(missing['LEFT_WRIST'],0)
            np.testing.assert_allclose(p['LEFT_WRIST'][10],[.3048,.6096,.9144])

    def test_frame_gaps_are_rejected_instead_of_silently_retimed(self):
        raw=self.fixture();raw['tracking'][10]['frame']=99
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'clip.json';file.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):read_trial(file)


@unittest.skipUnless(REF.exists(),'Generate the local reference first')
class ReferenceAndTrackingTests(unittest.TestCase):
    def test_reference_bounds_quaternions_and_duration(self):
        r=np.load(REF);m=load_model()
        self.assertEqual(r['qpos'].shape[1],m.nq)
        self.assertTrue(np.isfinite(r['qpos']).all())
        np.testing.assert_allclose(np.linalg.norm(r['qpos'][:,3:7],axis=1),1,atol=1e-8)
        for j in range(1,m.njnt):
            q=r['qpos'][:,m.jnt_qposadr[j]]
            self.assertTrue(np.all(q>=m.jnt_range[j,0]))
            self.assertTrue(np.all(q<=m.jnt_range[j,1]))
        info=json.loads(REF.with_suffix('.json').read_text())
        selection=info['selection'];expected=(selection['end_frame']-selection['start_frame'])/selection['fps']/info['playback_speed']
        self.assertAlmostEqual(r['times'][-1],expected)
        np.testing.assert_allclose(reference_at(r,-1)[0],r['qpos'][0])
        np.testing.assert_allclose(reference_at(r,99)[0],r['qpos'][-1])

    def test_invalid_crop_does_not_create_reference(self):
        selection=json.loads((ROOT/'runs/spl_tracking/selection.json').read_text())
        selection['start_frame']=selection['end_frame']
        with tempfile.TemporaryDirectory() as directory:
            selection_path=Path(directory)/'selection.json';selection_path.write_text(json.dumps(selection))
            with self.assertRaises(ValueError):retarget(selection_path,Path(directory)/'invalid.npz')
            self.assertFalse((Path(directory)/'invalid.npz').exists())

    def test_motor_model_retains_free_root_and_limits(self):
        m=motor_model()
        self.assertEqual(m.jnt_type[0],mujoco.mjtJoint.mjJNT_FREE)
        self.assertEqual(m.neq,0)
        self.assertFalse(np.any(m.body_gravcomp))
        self.assertFalse(np.any(m.actuator_trnid[:,0]==0))
        self.assertTrue(np.all(m.actuator_forcelimited))

    def test_physical_rollout_without_root_support(self):
        with tempfile.TemporaryDirectory() as directory:
            report=track(REF,Path(directory))
        self.assertFalse(report['fell'])
        self.assertEqual(report['max_external_root_or_body_force'],0)
        self.assertEqual(report['qp_fallback_count'],0)
        self.assertFalse(any(report['mujoco_warning_counts']))
        self.assertLess(report['maximum_ankle_horizontal_displacement_m'],.02)
        self.assertLessEqual(report['maximum_torque_limit_fraction'],1+1e-6)
        for side in ['left','right']:
            self.assertLess(report['rms_error_m'][side+'_wrist'],.06)


if __name__=='__main__':unittest.main()
