"""M5-1 headless/GUI evaluation and reproducible stage feasibility gates."""
import argparse
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from examples.m5_upper_body_ppo_env import UpperBodyResidualEnv, OUT, STAGES
from examples.m3_control import write_csv
import numpy as np


def evaluate(env,policy=None,seed=7,speed=1.):
    if not np.isfinite(speed) or speed<0:
        raise ValueError('GUI speed must be finite and >=0 (0 means unlimited)')
    obs,reset_info=env.reset(seed=seed);reward_rows=[];ret=0.
    while True:
        frame_start=time.perf_counter()
        action=np.zeros(env.action_space.shape,np.float32) if policy is None else policy.predict(obs,deterministic=True)[0]
        obs,reward,term,trunc,info=env.step(action);ret+=reward
        reward_rows.append(dict(time_s=float(env.d.time),reward=reward,**info['reward_terms']))
        if env.render_mode=='human' and speed>0:
            time.sleep(max(0.,env.control_dt/speed-(time.perf_counter()-frame_start)))
        if term or trunc:break
    rows=env.rows;late=[r for r in rows if r['time_s']>env.max_seconds-1]
    gap=maxgap=0
    for r in rows:
        if r['time_s']<.2:continue
        gap=gap+1 if r['contacts']==0 else 0;maxgap=max(maxgap,gap)
    left_ratio=float(np.mean([r['left_contacts']>0 for r in late])) if late else 0.
    # Gate on actual state/contact/balance and the final one-second window.
    success=bool(info['termination_reason']=='time_limit' and late and max(r['drift_m'] for r in rows)<.05 and not any(r['fallen'] for r in rows) and not any(w.number for w in env.d.warning) and min(r['com_margin_m'] for r in rows)>0 and np.mean([r['both_feet'] for r in rows])>=.99 and np.mean([r['goal_distance_m'] for r in late])<.04)
    if env.stage==0:success=bool(info['termination_reason']=='time_limit' and max(r['drift_m'] for r in rows)<.05 and not any(r['fallen'] for r in rows) and not any(w.number for w in env.d.warning))
    if env.stage>=1 and env.condition!='A':success=success and left_ratio>=.5 and max(r['left_normal'] for r in rows)<20
    if env.stage>=2:success=success and np.mean([r['head_error_deg'] for r in late])<15 and np.mean([r['trunk_error_deg'] for r in late])<5
    right_only_success=bool(info['termination_reason']=='time_limit' and max(r['drift_m'] for r in rows)<.05 and not any(r['fallen'] for r in rows))
    summary=dict(right_holding_success=right_only_success,bimanual_stage_success=bool(success and env.stage>0 and left_ratio>=.5),guide_start_s=env.guide_start,condition=env.condition,stage=env.stage,stage_name=STAGES[env.stage],seed=seed,success=bool(success),duration_s=float(env.d.time),termination_reason=info['termination_reason'],episode_return=ret,observation_dim=env.observation_space.shape[0],action_dim=env.action_space.shape[0],max_relative_drift_mm=max(r['drift_m'] for r in rows)*1000,max_contact_gap_ms=maxgap*env.dt*1000,left_final_contact_ratio=left_ratio,max_left_normal_force_N=max(r['left_normal'] for r in rows),mean_left_force_z_N=float(np.mean([r['L_force_z_N'] for r in rows])),mean_right_force_z_N=float(np.mean([r['R_force_z_N'] for r in rows])),contact_slip_rms_m_s=float(np.sqrt(np.mean([r['slip_ms']**2 for r in rows]))),min_com_margin_mm=min(r['com_margin_m'] for r in rows)*1000,both_feet_ratio=float(np.mean([r['both_feet'] for r in rows])),final_goal_distance_mm=float(np.mean([r['goal_distance_m'] for r in late])*1000) if late else rows[-1]['goal_distance_m']*1000,final_head_error_deg=float(np.mean([r['head_error_deg'] for r in late])) if late else rows[-1]['head_error_deg'],final_trunk_error_deg=float(np.mean([r['trunk_error_deg'] for r in late])) if late else rows[-1]['trunk_error_deg'],minimum_face_gap_mm=min(r['face_gap_m'] for r in rows)*1000,fallen=any(r['fallen'] for r in rows),warnings=[int(w.number) for w in env.d.warning],ball_state_writes_after_reset=0,neq=env.m.neq,reset_method=reset_info['reset_method'])
    for side in ('R','L'):
        for metric in ('ref_target_mm','target_actual_mm','ref_actual_mm','orientation_error_deg'):summary[side+'_'+metric+'_rms']=float(np.sqrt(np.mean([r[side+'_'+metric]**2 for r in rows])))
    return summary,reward_rows


def save(env,folder,summary,reward_rows):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    write_csv(folder/'physics.csv',env.rows);write_csv(folder/'joint_targets_torques.csv',env.goals);write_csv(folder/'reward_terms.csv',reward_rows)
    if env.contact_rows:write_csv(folder/'contacts.csv',env.contact_rows)
    np.savez_compressed(folder/'states.npz',qpos=env.states,time_s=env.state_times)
    (folder/'summary.json').write_text(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',choices=list('ABCD'),default='B');p.add_argument('--stage',type=int,default=1);p.add_argument('--reference',type=Path);p.add_argument('--model',type=Path);p.add_argument('--seed',type=int,default=7);p.add_argument('--gui',action='store_true');p.add_argument('--open-loop',action='store_true');p.add_argument('--guide-start',type=float,default=1.);p.add_argument('--output',type=Path);p.add_argument('--speed',type=float,default=1.,help='GUI playback multiplier; 0 disables pacing (physics timestep unchanged)');a=p.parse_args()
    if not np.isfinite(a.speed) or a.speed<0:p.error('--speed must be finite and >=0')
    env=UpperBodyResidualEnv(condition=a.condition,stage=a.stage,reference=a.reference,render_mode='human' if a.gui else None,record=True,guide_feedback=not a.open_loop,guide_start=a.guide_start)
    policy=None
    if a.model:
        from stable_baselines3 import PPO
        policy=PPO.load(a.model,device='cpu')
    summary,terms=evaluate(env,policy,a.seed,speed=a.speed);save(env,a.output or OUT/(f'{a.condition}_stage{a.stage}_seed{a.seed}'+('_gui' if a.gui else '')),summary,terms);env.close();print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
