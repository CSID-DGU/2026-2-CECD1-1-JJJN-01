"""Safe regressions and short PPO integration; all writable paths redirected."""
import json, runpy, unittest
from pathlib import Path
import numpy as np
from examples.m6_2.experiment import OUT,OLD,ROOT,ShootingEnv,audit

def smoke(reference,ff=0.):
 from examples.m6_2.refine import StableVelocityEnv
 import torch
 torch.set_num_threads(1)
 from stable_baselines3 import PPO
 from stable_baselines3.common.env_checker import check_env
 p=OUT/'ppo_smoke';p.mkdir(exist_ok=True)
 e=StableVelocityEnv(reference=reference,velocity_ff=ff);check_env(e,warn=True);o,_=e.reset(seed=7)
 a=PPO.load(OLD/'smoke_verified/final.zip',env=e,device='cpu');assert a.action_space==e.action_space and a.observation_space==e.observation_space
 action=a.predict(o,deterministic=True)[0];assert np.isfinite(action).all()
 a.learn(total_timesteps=128,reset_num_timesteps=False);a.save(p/'final');b=PPO.load(p/'final.zip',env=e,device='cpu');np.testing.assert_array_equal(a.predict(o,deterministic=True)[0],b.predict(o,deterministic=True)[0]);before=b.num_timesteps;b.learn(total_timesteps=128,reset_num_timesteps=False);b.save(p/'resumed')
 (p/'verification.json').write_text(json.dumps(dict(action_dim=e.action_space.shape[0],observation_dim=e.observation_space.shape[0],old_checkpoint_space_compatible=True,old_reference_changed=True,strict_original_resume_config_compatible=False,checkpoint_save_load_exact=True,resume_initial_steps=before,resume_final_steps=b.num_timesteps,learned_shooting_success=False,reference=str(reference),velocity_ff=ff),indent=2));e.close()

def regressions():
 from examples import m5_coordinated_eval
 target=OUT/'regression';target.mkdir(exist_ok=True);(target/'preservation_before.json').write_bytes((m5_coordinated_eval.OUT/'preservation_before.json').read_bytes());m5_coordinated_eval.OUT=target
 try:runpy.run_path(str(ROOT/'tests/run_shooting_grip_regressions.py'),run_name='__main__')
 except SystemExit as x:
  if x.code:raise
 # Old M6 test's archive writes target the new runtime OUT; sources/assets stay intact.
 for dirname in ['smoke_verified','resume_verified']:
  link=OUT/dirname
  if not link.exists():link.symlink_to(OLD/dirname,target_is_directory=True)
 from examples.m6 import test_core
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_core));assert result.wasSuccessful()
 audit()

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('command',choices=['smoke','regression']);p.add_argument('--reference');p.add_argument('--ff',type=float,default=0.);a=p.parse_args();smoke(a.reference,a.ff) if a.command=='smoke' else regressions()
