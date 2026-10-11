"""M5 zero-action gate, saved PPO deterministic evaluation and physical telemetry."""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from examples.m5_residual_ppo_env import BasketballResidualEnv,OUT,BASE
import numpy as np
from examples.m3_control import write_csv

def episode(env,policy=None,seed=7):
    obs,reset_info=env.reset(seed=seed);episode_return=0.;reward_rows=[]
    while True:
        action=np.zeros(12,dtype=np.float32) if policy is None else policy.predict(obs,deterministic=True)[0]
        obs,reward,terminated,truncated,info=env.step(action);episode_return+=reward;reward_rows.append(dict(time_s=float(env.d.time),reward=reward,**info['reward_terms']))
        if env.render_mode=='human':time.sleep(env.control_dt)
        if terminated or truncated:break
    rows=env.rows;prior=[r for r in rows if .9<=r['time_s']<1];end=[r for r in rows if r['time_s']>6];lift=[r for r in rows if r['time_s']>=1]
    gap=maxgap=0
    for r in rows:
        if r['time_s']<.2:continue
        gap=gap+1 if not r['contacts'] else 0;maxgap=max(gap,maxgap)
    s=dict(duration_s=float(env.d.time),termination_reason=info['termination_reason'],episode_return=episode_return,reset_method=reset_info['reset_method'],wrist_rise_m=np.mean([r['wrist_z_m'] for r in end])-np.mean([r['wrist_z_m'] for r in prior]) if prior and end else None,ball_rise_m=np.mean([r['ball_z_m'] for r in end])-np.mean([r['ball_z_m'] for r in prior]) if prior and end else None,max_relative_drift_m=max(r['drift_m'] for r in lift) if lift else max(r['drift_m'] for r in rows),max_episode_relative_drift_m=max(r['drift_m'] for r in rows),contact_ratio=float(np.mean([r['contacts']>0 for r in lift])) if lift else 0.,max_contact_gap_s=maxgap*env.dt,both_feet_ratio=float(np.mean([r['both_feet'] for r in rows])),min_com_margin_m=min(r['com_margin_m'] for r in rows),mean_vertical_force_N=np.mean([r['force_z_N'] for r in lift]) if lift else 0.,fallen=bool(any(r['fallen'] for r in rows)),warnings=[int(w.number) for w in env.d.warning],ball_state_writes_after_reset=env.ball_state_writes_after_reset,neq=env.m.neq,observation_dim=env.observation_space.shape[0],action_dim=12,control_hz=1/env.control_dt,physics_dt=env.dt)
    return s,reward_rows

def save_episode(env,folder,s,reward_rows):
    folder.mkdir(parents=True,exist_ok=True);write_csv(folder/'physics.csv',env.rows);write_csv(folder/'joint_targets_torques.csv',env.goals);write_csv(folder/'reward_terms.csv',reward_rows);np.savez_compressed(folder/'states.npz',qpos=env.states,time_s=env.state_times);(folder/'summary.json').write_text(json.dumps(s,indent=2))

def gate(output=OUT):
    output.mkdir(exist_ok=True,parents=True);env=BasketballResidualEnv(record=True);baseline,terms=episode(env);save_episode(env,output/'baseline_warmup',baseline,terms);env.save_snapshot(output/'reset_snapshot.npz');original=json.loads((BASE/'C/summary.json').read_text());keys=['wrist_rise_m','ball_rise_m','max_relative_drift_m','contact_ratio','max_contact_gap_s','both_feet_ratio','min_com_margin_m','mean_vertical_force_N']
    differences={k:abs(baseline[k]-original[k]) for k in keys};old=np.load(BASE/'C/states.npz');state_error=float(np.max(abs(np.array(env.states)-old['qpos']))) if np.array(env.states).shape==old['qpos'].shape else None
    first=np.array(env.states).copy();env.use_snapshot=True;snapshot,terms=episode(env);save_episode(env,output/'baseline_snapshot',snapshot,terms);snapshot_error=float(np.max(abs(np.array(env.states)-first)))
    passed=state_error is not None and state_error<1e-8 and max(differences.values())<1e-8 and snapshot_error<1e-8 and baseline['termination_reason']=='time_limit' and not baseline['fallen'] and not any(baseline['warnings']) and baseline['ball_rise_m']>.2 and baseline['max_relative_drift_m']<.05
    result=dict(passed=bool(passed),metric_differences=differences,max_recorded_qpos_difference=state_error,snapshot_max_qpos_difference=snapshot_error,input_hashes=env.input_hashes,env_code_sha256=hashlib.sha256((ROOT/'examples/m5_residual_ppo_env.py').read_bytes()).hexdigest(),baseline=baseline,snapshot=snapshot);(output/'baseline_gate.json').write_text(json.dumps(result,indent=2));env.close();print(json.dumps(result,indent=2),flush=True)
    if not passed:raise RuntimeError('Baseline gate failed: PPO must not run')
    return result

def require_gate(output=OUT):
    f=output/'baseline_gate.json'
    if not f.exists():raise RuntimeError('Run --baseline before PPO')
    gate=json.loads(f.read_text())
    if not gate['passed'] or hashlib.sha256((ROOT/'examples/m5_residual_ppo_env.py').read_bytes()).hexdigest()!=gate['env_code_sha256']:raise RuntimeError('Baseline gate failed or env code changed')
    for path,digest in gate['input_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise RuntimeError('Protected baseline input changed')
    return gate

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',action='store_true');p.add_argument('--model',type=Path);p.add_argument('--gui',action='store_true');p.add_argument('--output',type=Path,default=OUT);args=p.parse_args()
    if args.baseline and not args.gui:gate(args.output);return
    policy=None
    if args.model:
        require_gate(args.output)
        from stable_baselines3 import PPO
        policy=PPO.load(args.model,device='cpu')
    env=BasketballResidualEnv(record=True,use_snapshot=True,render_mode='human' if args.gui else None);snapshot=args.output/'reset_snapshot.npz'
    if snapshot.exists():env.load_snapshot(snapshot)
    s,terms=episode(env,policy);folder=args.output/('gui_evaluation' if args.gui else 'evaluation');save_episode(env,folder,s,terms);env.close();print(json.dumps(s,indent=2))
if __name__=='__main__':main()
