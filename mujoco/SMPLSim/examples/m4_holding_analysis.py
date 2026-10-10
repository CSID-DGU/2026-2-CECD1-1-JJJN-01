"""Compare saved A/B/C states and torques without rerunning physics."""
from pathlib import Path
import sys
import json
import csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from examples.m3_control import write_csv


def main():
    output=ROOT/'outputs/m4_physical_holding';pairs=[];summaries=[]
    for body in ('body_01','body_02','body_03','body_04'):
        paths={m:output/body/m for m in ('A','B','C')}
        summary={m:json.loads((p/'summary.json').read_text()) for m,p in paths.items()}
        states={m:dict(np.load(p/'states.npz')) for m,p in paths.items()}
        torques={m:dict(np.load(p/'torques.npz')) for m,p in paths.items()}
        held=torques['A']['time_s']<summary['C']['release']['actual_time_s']
        delta=torques['C']['torque_Nm']-torques['A']['torque_Nm']
        qdiff=np.max(np.abs(states['A']['qpos']-states['B']['qpos'][:,:160]))
        vdiff=np.max(np.abs(states['A']['qvel']-states['B']['qvel'][:,:159]))
        tdiff=np.max(np.abs(torques['A']['torque_Nm']-torques['B']['torque_Nm']))
        if max(qdiff,vdiff,tdiff)>1e-12:raise AssertionError(body+' kinematic parity failed')
        if np.sqrt(np.mean(delta[held]**2))<.01:raise AssertionError(body+' no measurable physical load effect')
        pairs.append({'body':body,'max_A_B_human_qpos_difference':float(qdiff),'max_A_B_human_qvel_difference':float(vdiff),
            'max_A_B_torque_difference_Nm':float(tdiff),'C_minus_A_hold_torque_delta_rms_Nm':float(np.sqrt(np.mean(delta[held]**2))),
            'C_hold_torque_rms_increase_percent':100*(summary['C']['hold_joint_torque_rms_Nm']/summary['A']['hold_joint_torque_rms_Nm']-1),
            'C_minus_A_body_rmse_rad':summary['C']['body_joint_rmse_rad']-summary['A']['body_joint_rmse_rad'],
            'C_minus_A_root_rmse_m':summary['C']['root_position_rmse_m']-summary['A']['root_position_rmse_m'],
            'mean_C_weld_ball_z_force_N':summary['C']['mean_weld_ball_force_world_N'][2]})
        jointrows=[]
        for i,name in enumerate(torques['A']['joint_names']):
            jointrows.append({'joint':str(name),'A_hold_torque_rms_Nm':float(np.sqrt(np.mean(torques['A']['torque_Nm'][held,i]**2))),
                'C_hold_torque_rms_Nm':float(np.sqrt(np.mean(torques['C']['torque_Nm'][held,i]**2))),
                'hold_torque_delta_rms_Nm':float(np.sqrt(np.mean(delta[held,i]**2)))})
        write_csv(output/body/'joint_torque_comparison.csv',jointrows)
        summaries.extend(summary.values())
    # Separate bounded-overlap contact diagnostic, if it has been run.
    contact=output/'contact_stress/body_01/C/summary.json'
    if contact.exists():
        s=json.loads(contact.read_text())
        if not s['completed'] or s['warnings'] or s['fall_count'] or s['state_writes_during_weld_or_after_release'] or s['hand_contact_frames']==0 or s['max_hand_penetration_m']>.01 or s['max_hand_contact_force_N']>200:
            raise AssertionError('Weld hand-contact diagnostic failed')
    write_csv(output/'paired_comparison.csv',pairs)
    fields=('body','mode','ball_mass_kg','body_joint_rmse_rad','root_position_rmse_m','both_feet_contact_rate','balance_fraction','min_support_margin_m','min_coupled_support_margin_m','foot_slip_rms_m_s','joint_torque_rms_Nm','hold_joint_torque_rms_Nm','fall_count','hand_contact_frames')
    write_csv(output/'comparison.csv',[{k:s[k] for k in fields} for s in summaries])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8));labels=[f'{s["body"][-2:]} {s["mode"]}' for s in summaries]
    for ax,key,title,scale in [(axes[0,0],'body_joint_rmse_rad','Body tracking RMSE (rad)',1),
            (axes[0,1],'hold_joint_torque_rms_Nm','Holding joint torque RMS (Nm)',1),
            (axes[1,0],'root_position_rmse_m','Root tracking RMSE (mm)',1000),
            (axes[1,1],'both_feet_contact_rate','Both-feet contact (%)',100)]:
        ax.bar(labels,[s[key]*scale for s in summaries],color=[{'A':'#6699cc','B':'#55aa88','C':'#dd9955'}[s['mode']] for s in summaries])
        ax.set_title(title);ax.tick_params(axis='x',rotation=60);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(output/'holding_comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for mode in ('B','C'):
        with (output/'body_01'/mode/'trajectory.csv').open() as f:rows=list(csv.DictReader(f))
        free=[r for r in rows if float(r['time_s'])>=2.5 and float(r['time_s'])<3.1]
        axes[0].plot([float(r['ball_x_m']) for r in free],[float(r['ball_z_m']) for r in free],label=mode)
    for mode in ('A','B','C'):
        with np.load(output/'body_01'/mode/'torques.npz') as t:
            index=list(t['joint_names']).index('R_Shoulder_x')
            axes[1].plot(t['time_s'],t['torque_Nm'][:,index],label=mode)
    axes[0].set(xlabel='Ball x (m)',ylabel='Ball z (m)',title='Body 01 handoff: first 0.6 s flight')
    axes[1].set(xlabel='Time (s)',ylabel='R_Shoulder_x torque (Nm)',title='Body 01 actual actuator torque');axes[1].axvline(2.5,color='k',linestyle=':',label='release')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'flight_and_torque.png',dpi=150);plt.close(fig)
    print(json.dumps(pairs,indent=2));print('A/B PARITY AND C LOAD EFFECT VERIFIED')

if __name__=='__main__':main()
