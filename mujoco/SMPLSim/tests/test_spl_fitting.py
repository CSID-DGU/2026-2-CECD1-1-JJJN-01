"""Read-only trial checks, missing-data rules, and actual SMPL forward parity."""
import unittest
import json,tempfile
from pathlib import Path
import numpy as np
import torch
from smpl_sim.motion.spl_data import load_trial,fill_short_gaps,mapping_table,BODY_MAP,FINGER_MAP
from smpl_sim.motion.spl_fitting import JointFK,observation_positions
from smpl_sim.smpllib.smpl_parser import SMPLH_Parser

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parents[2]
DATA=BASE/'SPL-Open-Data/basketball/freethrow/data/2025-12-18/P0002/BB_FT_P0002_T0049.json'
MODEL=BASE/'basketball-r1/SMPLSim/assets/body_models/smplh'

class SPLTests(unittest.TestCase):
    def test_speed_spike_is_masked_then_interpolated(self):
        player={k:[0.,0.,3.] for k in BODY_MAP}
        player['LEFT_HIP']=[0.,.5,3.];player['RIGHT_HIP']=[0.,-.5,3.]
        frames=[]
        for i in range(7):
            data={k:list(v) for k,v in player.items()}
            if i==3:data['RIGHT_WRIST']=[40.,0.,3.]
            frames.append({'frame':i,'time':int(i*1000/60),'data':{'player':data}})
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'synthetic.json';path.write_text(json.dumps({'sampling_rate':60,'tracking':frames}))
            trial=load_trial(path);j=trial.names.index('RIGHT_WRIST')
            self.assertTrue(trial.outliers[2:5,j].all());self.assertTrue(trial.filled[2:5,j].all())
            np.testing.assert_allclose(trial.weights[2:5,j],.25)
            np.testing.assert_allclose(trial.restore_world(trial.normalized_world_m)[2:5,j,0],0)

    def test_pelvis_observation_is_hip_midpoint_not_library_root(self):
        joints=torch.zeros(2,52,3);joints[:,1,0]=2;joints[:,2,0]=4;joints[:,0,0]=99
        self.assertEqual(float(observation_positions(joints,['Pelvis'])[0,0,0]),3)

    def test_gap_policy(self):
        points=np.arange(30,dtype=float).reshape(10,1,3)
        missing=points.copy();missing[0]=np.nan;missing[2:4]=np.nan;missing[5:9]=np.nan
        filled,mask=fill_short_gaps(missing,2)
        np.testing.assert_allclose(filled[2:4],points[2:4]);self.assertEqual(mask.sum(),2)
        self.assertTrue(np.isnan(filled[0]).all());self.assertTrue(np.isnan(filled[5:9]).all())

    @unittest.skipUnless(DATA.exists(),'Local licensed dataset not present')
    def test_actual_trial_coordinates_masks_and_handedness(self):
        trial=load_trial(DATA)
        self.assertEqual(trial.fps,60);self.assertEqual(len(trial.names),69)
        self.assertEqual(trial.smpl_m.shape,(240,69,3))
        np.testing.assert_allclose(np.linalg.det(trial.world_rotation),1,atol=1e-12)
        np.testing.assert_allclose(trial.restore_world(trial.normalized_world_m),trial.original_m,atol=1e-12,equal_nan=True)
        lateral=trial.normalized_world_m[:30,trial.names.index('LEFT_HIP')]-trial.normalized_world_m[:30,trial.names.index('RIGHT_HIP')]
        self.assertGreater(np.nanmedian(lateral[:,1]),0)
        table=mapping_table(trial.names);self.assertEqual(len(table),69)
        self.assertEqual(len(FINGER_MAP),30);self.assertEqual(len(BODY_MAP),14)
        keys,names,points,weights=trial.observations(False)
        self.assertEqual(points.shape,(240,14,3));self.assertTrue(np.isfinite(points).all())
        self.assertNotIn('L_Index1',names);self.assertTrue((weights>0).all())

    @unittest.skipUnless(MODEL.exists(),'Local SMPL+H models not present')
    def test_joint_fk_matches_full_forward_and_finite_gradients(self):
        torch.set_num_threads(1);torch.manual_seed(3)
        parser=SMPLH_Parser(str(MODEL),gender='male',num_betas=16,use_pca=False,flat_hand_mean=True,create_transl=False)
        fk=JointFK(parser);pose=(torch.randn(2,156)*.1).requires_grad_()
        beta=(torch.randn(16)*.2).requires_grad_();translation=torch.randn(2,3,requires_grad=True)
        predicted,_=fk(pose,beta,translation)
        _,reference=parser.get_joints_verts(pose,beta.expand(2,-1),translation)
        torch.testing.assert_close(predicted,reference[:,:52],atol=2e-6,rtol=1e-5)
        predicted.square().mean().backward()
        for param in (pose,beta,translation):
            self.assertIsNotNone(param.grad);self.assertTrue(torch.isfinite(param.grad).all());self.assertGreater(float(param.grad.abs().sum()),0)

if __name__=='__main__':unittest.main()
