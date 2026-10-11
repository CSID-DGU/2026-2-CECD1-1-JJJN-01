"""M5 smoke curves, physical comparison, reward shares and preservation audit."""
import csv,hashlib,importlib.metadata,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from examples.m5_residual_ppo_env import OUT,BASE,BasketballResidualEnv
from examples.m3_control import write_csv
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    smoke=json.loads((OUT/'smoke/summary.json').read_text());gate=json.loads((OUT/'baseline_gate.json').read_text());old=json.loads((BASE/'C/summary.json').read_text());names=['M4 Lifting C','Zero residual','PPO before','PPO after'];values=[old,gate['baseline'],smoke['before'],smoke['after']];keys=['wrist_rise_m','ball_rise_m','max_relative_drift_m','contact_ratio','max_contact_gap_s','both_feet_ratio','min_com_margin_m','mean_vertical_force_N','duration_s'];write_csv(OUT/'evaluation_comparison.csv',[dict(condition=n,**{k:v[k] for k in keys}) for n,v in zip(names,values)])
    rows=list(csv.DictReader((OUT/'smoke/step_rewards.csv').open()));steps=np.array([int(r['step']) for r in rows]);rew=np.array([float(r['reward']) for r in rows]);fig,axes=plt.subplots(2,1,figsize=(10,7),constrained_layout=True);axes[0].plot(steps,rew,alpha=.4,label='Per control-step reward');axes[0].plot(steps[15:],np.convolve(rew,np.ones(16)/16,'valid'),label='16-step mean');axes[0].set_ylabel('Reward');axes[0].legend();progress=list(csv.DictReader((OUT/'smoke/training/progress.csv').open()));x=np.array([int(r['time/total_timesteps']) for r in progress]);y=np.array([float(r['train/value_loss']) if r.get('train/value_loss') else np.nan for r in progress]);axes[1].plot(x,y,'o-');axes[1].set_ylabel('Value loss');axes[1].set_xlabel('PPO control steps');fig.suptitle('256-step PPO pipeline smoke test; not shooting learning');fig.savefig(OUT/'training_curve.png',dpi=150);plt.close(fig)
    weights={'ball_target':4.,'grasp':1.,'motion':.5,'balance':.5,'action':.1,'slip':.2};means=[]
    for label,path in [('zero',OUT/'baseline_warmup/reward_terms.csv'),('ppo_after',OUT/'smoke/after/reward_terms.csv')]:
        r=list(csv.DictReader(path.open()));contributions={k:float(np.mean([float(x[k]) for x in r]))*w*(-1 if k in ('action','slip') else 1) for k,w in weights.items()};means.append(dict(condition=label,**contributions))
    write_csv(OUT/'reward_contributions.csv',means);fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
    for i,r in enumerate(means):ax.bar(np.arange(6)+i*.35,[r[k] for k in weights],width=.35,label=r['condition'])
    ax.set_xticks(np.arange(6)+.175,list(weights));ax.set_ylabel('Mean weighted contribution');ax.legend();ax.grid(axis='y',alpha=.2);fig.savefig(OUT/'reward_components.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True,constrained_layout=True)
    for label,folder in [('zero',OUT/'baseline_warmup'),('ppo_before',OUT/'smoke/before'),('ppo_after',OUT/'smoke/after')]:
        z=np.genfromtxt(folder/'physics.csv',delimiter=',',names=True)
        for ax,k in zip(axs,['ball_z_m','drift_m','force_z_N']):ax.plot(z['time_s'],z[k],label=label)
    for ax,label in zip(axs,['Ball height (m)','Wrist-relative drift (m)','Vertical contact force (N)']):ax.set_ylabel(label);ax.legend();ax.grid(alpha=.2)
    axs[-1].set_xlabel('Physics time (s)');fig.savefig(OUT/'contact_and_lift_comparison.png',dpi=150);plt.close(fig)
    before=json.loads((OUT/'preservation_before.json').read_text());changed=[p for p,h in before.items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];dependency=json.loads((OUT/'dependency_check.json').read_text());runtime={k:v['version'] for k,v in dependency['requirements'].items()};runtime_after={k:importlib.metadata.version(k) for k in runtime};audit=dict(checked_files=len(before),changed_files=changed,checked_dependency_versions_unchanged=runtime_after==runtime,runtime_after=runtime_after,mujoco_version=importlib.metadata.version('mujoco'));(OUT/'preservation_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit));print(json.dumps(means));
    if changed or runtime_after!=runtime:raise RuntimeError('Protected files or runtime changed')
if __name__=='__main__':main()
