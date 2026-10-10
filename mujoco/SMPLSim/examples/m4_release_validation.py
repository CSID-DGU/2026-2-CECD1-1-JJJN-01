"""Reproducible M4-2 one-body parameter experiments, never edits source motion."""
import sys
import json
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from examples.m4_ball_release import simulate
from examples.m3_control import write_csv


def main():
    output=ROOT/'outputs/m4_ball_release'
    experiments={
        'early':{'release_time':2.1},
        'late':{'release_time':2.9},
        'wrist_test':{'test_wrist_amplitude':.35},
        'left':{'hand':'left'},
        # Intentional partial overlap: exercise physical hand recontact on release.
        # Not a recommended grip or fitted observation; ball velocity remains hand-derived.
        'contact_stress':{'offset':[-.07,-.12,0.], 'check':False},
    }
    summaries=[]
    for name,changes in experiments.items():
        args=SimpleNamespace(output=output/name,scene=ROOT/'outputs/m4_basketball/body_01_basketball.xml',
            humanoid=ROOT.parents[2]/'basketball-r1/SMPLSim/outputs/smplh_physics/body_01.xml',
            source=ROOT/'outputs/m3_spl_fitting/spl_P0002_T0049.npz',hand='right',offset=None,release_time=2.5,
            hand_collision='shooting',test_wrist_amplitude=0.,duration=6.,gui=False,video=False,check=True)
        for k,v in changes.items():setattr(args,k,v)
        s=simulate(args)
        if s['warnings'] or s['fall_frames'] or not s['completed']:raise AssertionError(name+' physical failure')
        if s['ball_state_writes_after_release'] or s['max_held_error_m']>1e-10 or s['release']['position_jump_m']>1e-10 or s['release']['velocity_jump_m_s']>1e-10:
            raise AssertionError(name+' discontinuity/state overwrite')
        if name=='contact_stress' and s['hand_contact_frames']==0:raise AssertionError('No stress hand contact')
        summaries.append((name,s))
    default=json.loads((output/'default/summary.json').read_text());summaries.insert(0,('default',default))
    import numpy as np
    by_name=dict(summaries)
    for a,b in (('default','wrist_test'),('early','late')):
        if np.linalg.norm(np.asarray(by_name[a]['release']['velocity_world_m_s'])-by_name[b]['release']['velocity_world_m_s'])<.01:
            raise AssertionError('Release/hand change did not affect velocity')
    rows=[]
    for name,s in summaries:
        rows.append({'case':name,'release_s':s['release']['actual_time_s'],
            **dict(zip(('release_vx_m_s','release_vy_m_s','release_vz_m_s'),s['release']['velocity_world_m_s'])),
            'body_rmse_rad':s['body_tracking_rmse_rad'],'root_rmse_m':s['root_aligned_rmse_m'],
            'hand_contact_frames':s['hand_contact_frames'],'ballistic_position_error_m':s['max_precontact_ballistic_position_error_m'],
            'fall_frames':s['fall_frames'],'warnings':len(s['warnings'])})
    write_csv(output/'comparison.csv',rows)
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,s in summaries:
        with (output/name/'trajectory.csv').open() as f:data=list(csv.DictReader(f))
        released=[r for r in data if int(r['released'])]
        # Compare first 0.6 seconds of post-release flight, before floor bounce.
        flight=[r for r in released if float(r['time_s']) <= s['release']['actual_time_s']+.6]
        axes[0].plot([float(r['ball_x_m']) for r in flight],[float(r['ball_z_m']) for r in flight],label=name)
        axes[1].plot([float(r['time_s']) for r in data],[float(r['vz_m_s']) for r in data],label=name)
        axes[0].scatter(s['release']['position_m'][0],s['release']['position_m'][2],s=20)
    axes[0].set(xlabel='World x (m)',ylabel='Ball centre z (m)',title='First 0.6 s after hand-derived release')
    axes[1].set(xlabel='Time (s)',ylabel='Ball vz (m/s)',title='Hold, release and floor rebound')
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'release_comparison.png',dpi=150);plt.close(fig)
    print('M4-2 PARAMETER VALIDATION PASSED')

if __name__=='__main__':main()
