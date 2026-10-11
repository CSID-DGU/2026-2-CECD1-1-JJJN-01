"""Measure source-speed reconstruction separately from tracking and shot success."""
from pathlib import Path
import json
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shoot_spl import ROOT,grasp_offset


def assess(report,run):
    """Engineering acceptance gates for this fixed avatar, not anatomical validation."""
    release=report['release']['time_s']
    end=report['settle_s']+report['motion_duration_s']
    hold=run['times']>=end+.5
    follow={str(name):float(np.sqrt(np.mean(run['errors'][hold,i]**2)))
            for i,name in enumerate(run['error_names'])} if np.any(hold) else {}
    checks={
        'no_fall':not report['fell'],
        'no_numerical_warning':not any(report['mujoco_warning_counts']),
        'no_qp_fallback':report['qp_fallback_count']==0,
        'bounded_motor_torques':report['maximum_torque_limit_fraction']<=1+1e-8,
        'no_external_force':report['max_external_applied_force']==0,
        'continuous_release':report['release']['qpos_change_at_release']==0 and report['release']['qvel_change_at_release']==0,
        'foot_displacement_under_15mm':report['maximum_ankle_horizontal_displacement_m']<.015,
        'both_wrist_rms_under_8cm':max(report['rms_error_m'][s+'_wrist'] for s in ['left','right'])<.08,
        'hand_ball_overlap_under_6mm':report['maximum_excluded_hand_ball_overlap_m']<.006,
        'held_ball_penetration_under_3mm':report['maximum_held_ball_contact_penetration_m']<.003,
        'grasp_reaction_under_150N':report['peak_grasp_force_n']<150,
        'followthrough_wrist_rms_under_6cm':bool(follow) and max(follow[s+'_wrist'] for s in ['left','right'])<.06,
    }
    source=np.array(report['adaptation']['source_release_analysis']['aligned_target_velocity_m_s'])
    actual=np.array(report['release']['linear_velocity_m_s'])
    requested=np.array(report['release_target_velocity_m_s'])
    return {'tracking_accepted':all(checks.values()),'checks':checks,
            'followthrough_rms_error_m':follow,
            'source_velocity_error_m_s':float(np.linalg.norm(actual-source)),
            'requested_velocity_error_m_s':float(np.linalg.norm(actual-requested)),
            'actual_to_source_speed_ratio':float(np.linalg.norm(actual)/np.linalg.norm(source)),
            'source_speed_reproduced_within_10pct':bool(np.linalg.norm(actual-source)<.1*np.linalg.norm(source)),
            'acceptance_note':'Project engineering gates for one fixed body/reference; passing does not validate human biomechanics or scoring.'}


def diagnose(output,baseline=ROOT/'runs/spl_shooting'):
    output=Path(output);baseline=Path(baseline)
    report=json.loads((output/'shooting_report.json').read_text(encoding='utf-8'))
    with np.load(output/'shooting.npz') as z:run={k:z[k] for k in z.files}
    diagnostic=assess(report,run)
    m=mujoco.MjModel.from_xml_string((output/'scene.xml').read_text(encoding='utf-8'))
    d=mujoco.MjData(m);offset=grasp_offset(m);jac=np.zeros((3,m.nv))
    grip_v=[];wrist_v=[]
    for q,v in zip(run['qpos'],run['qvel']):
        d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_forward(m,d)
        body=d.body('right_wrist');point=body.xpos+body.xmat.reshape(3,3)@offset
        mujoco.mj_jac(m,d,jac,None,point,m.body('right_wrist').id);grip_v.append(jac@v)
        mujoco.mj_jac(m,d,jac,None,body.xpos,m.body('right_wrist').id);wrist_v.append(jac@v)
    grip_v=np.array(grip_v);wrist_v=np.array(wrist_v)
    release=report['release']['time_s'];source=report['adaptation']['source_release_analysis']
    k=int(np.argmin(abs(run['times']-release)))
    wrist_target=np.array(report['adaptation']['release_correction']['reference_wrist_velocity_m_s'])
    diagnostic['physical_wrist_release_velocity_m_s']=wrist_v[k].tolist()
    diagnostic['physical_grip_release_velocity_m_s']=grip_v[k].tolist()
    diagnostic['reference_wrist_velocity_error_m_s']=float(np.linalg.norm(wrist_v[k]-wrist_target))
    source_wrist=np.array(report['adaptation']['release_correction']['source_adapted_wrist_velocity_m_s'])
    diagnostic['adapted_source_wrist_velocity_error_m_s']=float(np.linalg.norm(wrist_v[k]-source_wrist))
    source_v=np.array(source['aligned_target_velocity_m_s']);actual_v=np.array(report['release']['linear_velocity_m_s'])
    before=None
    if (baseline/'shooting_report.json').exists():
        before=json.loads((baseline/'shooting_report.json').read_text(encoding='utf-8'))
        diagnostic['baseline']={'release_speed_m_s':float(np.linalg.norm(before['release']['linear_velocity_m_s'])),
              'apex_height_m':before['apex_position_m'][2],'wrist_rms_error_m':before['rms_error_m'],
              'foot_displacement_m':before['maximum_ankle_horizontal_displacement_m']}
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    ax=axes[0,0];rel=run['times']-release
    ax.plot(rel,np.linalg.norm(grip_v,axis=1),label='Physical attached grip point')
    ax.plot(rel,np.linalg.norm(wrist_v,axis=1),label='Physical wrist origin')
    ax.plot(run['ball_times']-release,np.linalg.norm(run['ball_velocities'],axis=1),label='Physical ball',alpha=.7)
    ax.axhline(np.linalg.norm(source_v),color='black',ls='--',label='Source ball estimate (diagnostic)')
    ax.axhline(np.linalg.norm(source_wrist),color='green',ls=':',label='Adapted source wrist')
    ax.axvline(0,color='gray');ax.set(xlim=(-.45,.15),xlabel='Time from release (s)',ylabel='Speed (m/s)');ax.legend(fontsize=8)
    ax=axes[0,1];names=['vx','vy','vz'];x=np.arange(3)
    ax.bar(x-.2,source_v,.2,label='Source ball estimate');ax.bar(x,np.array(report['release_target_velocity_m_s']),.2,label='Reference attached grip')
    ax.bar(x+.2,actual_v,.2,label='Physical release');ax.set(xticks=x,xticklabels=names,ylabel='Velocity (m/s)');ax.legend(fontsize=8)
    ax=axes[1,0]
    for name,folder,r in [('Before',baseline,before),('Corrected',output,report)]:
        if r is None:continue
        with np.load(folder/'shooting.npz') as z:
            mask=z['ball_times']>=r['release']['time_s'];p=z['ball_positions'][mask]
            ax.plot(p[:,0],p[:,2],label=name)
    ax.plot([4.,4.45],[3.05,3.05],color='red',lw=3,label='Rim');ax.set(xlabel='Forward X (m)',ylabel='Height Z (m)',ylim=(0,3.7));ax.legend(fontsize=8)
    ax=axes[1,1]
    for i in [0,1]:ax.plot(rel,run['errors'][:,i]*100,label=str(run['error_names'][i]))
    ax.axvline(0,color='gray');ax.axhline(8,color='gray',ls=':');ax.set(xlabel='Time from release (s)',ylabel='Wrist tracking error (cm)');ax.legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('Release correction: source ball speed, physical grip motion, and form tracking')
    fig.tight_layout();fig.savefig(output/'release_comparison.png',dpi=150);plt.close(fig)
    np.savez_compressed(output/'release_diagnostics.npz',times=run['times'],grip_velocity=grip_v,wrist_velocity=wrist_v)
    report['release_diagnostics']=diagnostic
    (output/'shooting_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (output/'release_diagnostics.json').write_text(json.dumps(diagnostic,indent=2),encoding='utf-8')
    return diagnostic
