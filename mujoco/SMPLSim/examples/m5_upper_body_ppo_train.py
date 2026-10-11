"""M5-1 short PPO smoke only; later curriculum requires physical stage gates."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from examples.m5_upper_body_ppo_env import UpperBodyResidualEnv, OUT
from examples.m5_upper_body_ppo_eval import evaluate,save
from examples.m5_residual_ppo_eval import require_gate
from examples.m5_residual_ppo_train import SmokeLog
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env
from examples.m3_control import write_csv
import torch
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',choices=list('ABCD'),default='B');p.add_argument('--stage',type=int,default=1);p.add_argument('--steps',type=int,default=64);p.add_argument('--seed',type=int,default=7);p.add_argument('--reference',type=Path);a=p.parse_args()
    require_gate()
    baseline=json.loads((OUT/'baseline_gate.json').read_text())
    current=hashlib.sha256((ROOT/'examples/m5_upper_body_ppo_env.py').read_bytes()).hexdigest()
    if not baseline.get('passed') or baseline.get('validated_env_sha256')!=current:
        raise RuntimeError('M5-1 Action=0 baseline Gate missing or stale')
    if a.steps<64 or a.steps>512:raise ValueError('M5-1 limited to 64..512 smoke steps')
    if a.stage>1:
        gate=OUT/'feasibility/final_gates.json'
        if not gate.exists() or not json.loads(gate.read_text()).get('shooting_reference_physics_passed',False):raise RuntimeError('Physical shooting reference Gate failed; stage>=2 PPO blocked')
    torch.set_num_threads(1);folder=OUT/'smoke'/f'{a.condition}_stage{a.stage}_seed{a.seed}';folder.mkdir(parents=True,exist_ok=True)
    env=UpperBodyResidualEnv(condition=a.condition,stage=a.stage,reference=a.reference);check_env(env,warn=True)
    policy=PPO('MlpPolicy',Monitor(env,str(folder/'episodes')),n_steps=64,batch_size=32,n_epochs=2,learning_rate=1e-4,policy_kwargs=dict(net_arch=[32,32]),device='cpu',seed=a.seed,verbose=1)
    evaluation=UpperBodyResidualEnv(condition=a.condition,stage=a.stage,reference=a.reference,record=True)
    before,terms=evaluate(evaluation,policy,a.seed);save(evaluation,folder/'before',before,terms)
    policy.set_logger(configure(str(folder/'training'),['stdout','csv']));log=SmokeLog();policy.learn(total_timesteps=a.steps,callback=log)
    checkpoint=folder/'ppo_smoke.zip';policy.save(checkpoint);loaded=PPO.load(checkpoint,device='cpu');obs,_=evaluation.reset(seed=a.seed);np.testing.assert_array_equal(policy.predict(obs,deterministic=True)[0],loaded.predict(obs,deterministic=True)[0])
    after,terms=evaluate(evaluation,loaded,a.seed);save(evaluation,folder/'after',after,terms);write_csv(folder/'step_rewards.csv',log.rows)
    if log.endings:write_csv(folder/'terminations.csv',log.endings)
    summary=dict(condition=a.condition,stage=a.stage,requested_steps=a.steps,executed_steps=policy.num_timesteps,finite_rewards=bool(all(np.isfinite(r['reward']) for r in log.rows)),save_load_action_exact=True,before=before,after=after,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),shooting_learned=False)
    (folder/'summary.json').write_text(json.dumps(summary,indent=2));env.close();evaluation.close();print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
