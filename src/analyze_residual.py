"""Compare actual trajectories and keep readiness separate from scoring."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from humanoid_model import ROOT
from shoot_spl import HOOP,RADIUS


def analyze(output=ROOT/'runs/spl_residual_ready'):
    output=Path(output);summary=json.loads((output/'readiness.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for entry in summary['runs']:
        folder=output if entry['name']=='nominal' else output/'probes'/entry['name']
        data=np.load(folder/'shooting.npz');t=data['ball_times'];p=data['ball_positions']
        after=t>=entry['release']['time_s'];ground=np.flatnonzero(after&(p[:,2]<=RADIUS+.002))
        last=int(ground[0]) if len(ground) else len(t)-1
        flight=after&(np.arange(len(t))<=last);pts=p[flight]
        entry['closest_airborne_hoop_distance_m']=float(np.min(np.linalg.norm(pts-HOOP,axis=1)))
        axes[0].plot(pts[:,0],pts[:,2],label=entry['name'])
        axes[1].plot(pts[:,0],pts[:,1],label=entry['name'])
    for ax,k,label in [(axes[0],2,'Height (m)'),(axes[1],1,'Lateral position (m)')]:
        ax.scatter(HOOP[0],HOOP[k],s=80,c='black',marker='x',label='fixed hoop')
        ax.set_xlabel('Forward position (m)');ax.set_ylabel(label);ax.grid(alpha=.25)
    axes[0].axhline(HOOP[2],color='gray',linestyle='--');axes[0].legend(fontsize=8)
    fig.suptitle('Bounded residual probes: actual contact launch, no RL training')
    fig.tight_layout();fig.savefig(output/'residual_comparison.png',dpi=150);plt.close(fig)
    summary['dense_trajectory_curriculum_ready']=summary['physics_ready'] and summary['trajectory_sensitivity_observed'] and all(all(r['gates'].values()) for r in summary['runs'])
    summary['scoring_curriculum_ready']=summary['dense_trajectory_curriculum_ready'] and any(r['apex_position_m'][2]>=HOOP[2] and r['closest_airborne_hoop_distance_m']<.5 for r in summary['runs'])
    (output/'readiness.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'runs/spl_residual_ready');analyze(p.parse_args().output)
