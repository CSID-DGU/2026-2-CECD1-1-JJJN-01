"""Verify actual optimizer/normalization restoration and complete portable input."""
import sys,json,unittest
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from train_residual import validate_bundle


class CheckpointTests(unittest.TestCase):
    def test_resume_preserves_weights_optimizer_then_updates(self):
        original=ROOT/'runs/residual_smoke_v2';resumed=ROOT/'runs/residual_resume_test'
        old=original/json.loads((original/'latest.json').read_text())['checkpoint']
        starts=sorted((resumed/'checkpoints').glob('initial_*'));self.assertTrue(starts)
        before=PPO.load(old/'policy.zip',device='cpu');restored=PPO.load(starts[0]/'policy.zip',device='cpu')
        for key,value in before.policy.state_dict().items():torch.testing.assert_close(value,restored.policy.state_dict()[key],rtol=0,atol=0)
        a=before.policy.optimizer.state_dict();b=restored.policy.optimizer.state_dict()
        self.assertEqual(a['param_groups'],b['param_groups'])
        for key in a['state']:
            for name,value in a['state'][key].items():
                if torch.is_tensor(value):torch.testing.assert_close(value,b['state'][key][name],rtol=0,atol=0)
                else:self.assertEqual(value,b['state'][key][name])
        final=resumed/json.loads((resumed/'latest.json').read_text())['checkpoint'];after=PPO.load(final/'policy.zip',device='cpu')
        self.assertGreater(after.num_timesteps,before.num_timesteps)
        self.assertTrue(any(not torch.equal(v,after.policy.state_dict()[k]) for k,v in before.policy.state_dict().items()))
        import pickle
        with (old/'normalization.pkl').open('rb') as f:oldnorm=pickle.load(f)
        with (starts[0]/'normalization.pkl').open('rb') as f:startnorm=pickle.load(f)
        np.testing.assert_array_equal(oldnorm.obs_rms.mean,startnorm.obs_rms.mean)
        np.testing.assert_array_equal(oldnorm.obs_rms.var,startnorm.obs_rms.var)

    def test_checkpoint_is_complete_and_needs_no_raw_smpl_or_spl_data(self):
        output=ROOT/'runs/residual_resume_test';path=output/json.loads((output/'latest.json').read_text())['checkpoint']
        validate_bundle(path)
        for name in ['policy.zip','normalization.pkl','rng.pt','training_state.json','config.json','requirements.txt','src/train_residual.py',
                     'environment/scene.xml','environment/nominal_reference.npz','environment/nominal_reference.json']:
            self.assertTrue((path/name).exists(),name)
        self.assertFalse((path/'data/models/smplh/SMPLH_MALE.pkl').exists())
        state=json.loads((path/'training_state.json').read_text(encoding='utf-8'));self.assertEqual(state['episodes'],10)


if __name__=='__main__':unittest.main()
