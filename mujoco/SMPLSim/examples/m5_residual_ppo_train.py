"""Gate-protected, short CPU PPO smoke test; not shooting training."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from examples.m5_residual_ppo_env import BasketballResidualEnv,OUT
from examples.m5_residual_ppo_eval import require_gate,episode,save_episode
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env
from examples.m3_control import write_csv

class SmokeLog(BaseCallback):
    def __init__(self):super().__init__();self.rows=[];self.endings=[]
    def _on_step(self):
        for info,reward,done in zip(self.locals['infos'],self.locals['rewards'],self.locals['dones']):
            row=dict(step=self.num_timesteps,reward=float(reward),**info['reward_terms']);self.rows.append(row)
            if done:self.endings.append(dict(step=self.num_timesteps,reason=info['termination_reason'],**info.get('episode',{})))
        return bool(np.isfinite(self.locals['rewards']).all())

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=256);p.add_argument('--output',type=Path,default=OUT);p.add_argument('--seed',type=int,default=7);a=p.parse_args()
    if a.steps<64 or a.steps>2048:raise ValueError('M5-0 smoke test limited to 64..2048 steps')
    require_gate(a.output);folder=a.output/'smoke';folder.mkdir(exist_ok=True,parents=True);torch.set_num_threads(1)
    env=BasketballResidualEnv(use_snapshot=True);env.load_snapshot(a.output/'reset_snapshot.npz');check_env(env,warn=True);model=PPO('MlpPolicy',Monitor(env,str(folder/'episodes')),n_steps=64,batch_size=32,n_epochs=2,learning_rate=1e-4,policy_kwargs=dict(net_arch=[32,32]),device='cpu',seed=a.seed,verbose=1)
    evaluation=BasketballResidualEnv(record=True,use_snapshot=True);evaluation.load_snapshot(a.output/'reset_snapshot.npz');before,terms=episode(evaluation,model,a.seed);save_episode(evaluation,folder/'before',before,terms)
    log=SmokeLog();model.set_logger(configure(str(folder/'training'),['stdout','csv']));model.learn(total_timesteps=a.steps,callback=log);checkpoint=folder/'ppo_smoke.zip';model.save(checkpoint);loaded=PPO.load(checkpoint,device='cpu');obs,_=evaluation.reset(seed=a.seed);first=model.predict(obs,deterministic=True)[0];second=loaded.predict(obs,deterministic=True)[0];np.testing.assert_array_equal(first,second)
    after,terms=episode(evaluation,loaded,a.seed);save_episode(evaluation,folder/'after',after,terms);write_csv(folder/'step_rewards.csv',log.rows)
    if log.endings:write_csv(folder/'terminations.csv',log.endings)
    summary=dict(requested_steps=a.steps,executed_steps=model.num_timesteps,finite_rewards=bool(all(np.isfinite(r['reward']) for r in log.rows)),save_load_action_exact=True,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),checkpoint_bytes=checkpoint.stat().st_size,device=str(model.device),before=before,after=after,shooting_learning_claimed=False);(folder/'summary.json').write_text(json.dumps(summary,indent=2));env.close();evaluation.close();print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
