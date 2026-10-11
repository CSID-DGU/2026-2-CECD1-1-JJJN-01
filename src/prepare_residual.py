"""Bounded residual contract and deterministic physics probes; no RL training."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import mujoco
from scipy.interpolate import CubicSpline
from humanoid_model import ROOT
from shooting_model import contact_motor_model
from contact_reference import build_reference,smoothstep,GRIP_OFFSET
from contact_shot import shoot_contact

ACTION_NAMES=['elbow_flex','wrist_flex','forearm_aim','finger_flex','detach_time']
ACTION_LIMITS=np.array([.10,.16,.10,.10,.012])
OPTIONS={'arm_gain':300.,'arm_weight':2.,'foot_weight':180.,'acceleration_limit':1000.}


def residual_reference(ref,meta,action,detach=.99):
    action=np.asarray(action,dtype=float)
    if action.shape!=(5,) or not np.isfinite(action).all() or np.any(abs(action)>1):
        raise ValueError('Residual action must contain five finite values in [-1,1]')
    result={k:np.array(v,copy=True) for k,v in ref.items()}
    m=contact_motor_model();start=meta['push_start_s'];end=start+meta['push_duration_s']
    t=result['times'];window=np.array([smoothstep((x-(start-.22))/.22)*(1-smoothstep((x-end)/.30)) for x in t])
    offsets=action*ACTION_LIMITS;changes={}
    for name,delta in [('right_elbow_z',offsets[0]),('right_wrist_x',offsets[1]),('right_forearm_roll',offsets[2])]:
        j=m.joint(name).id;changes[j]=delta
    for j in range(m.njnt):
        name=m.joint(j).name
        if name.startswith('right_') and name.endswith('_x') and any(f in name for f in ['index','middle','ring','pinky']):changes[j]=offsets[3]
    for j,delta in changes.items():
        if delta==0:continue
        qa,va=m.jnt_qposadr[j],m.jnt_dofadr[j]
        candidate=np.clip(result['qpos'][:,qa]+delta*window,*m.jnt_range[j])
        correction=CubicSpline(t,candidate-result['qpos'][:,qa])
        result['qpos'][:,qa]=candidate
        result['qvel'][:,va]+=correction(t,1);result['qacc'][:,va]+=correction(t,2)
    if np.any(offsets[:4]):
        d=mujoco.MjData(m);grip=[]
        for q in result['qpos']:
            d.qpos[:]=q;mujoco.mj_forward(m,d)
            grip.append(d.body('right_wrist').xpos+d.body('right_wrist').xmat.reshape(3,3)@GRIP_OFFSET)
        result['grip_target']=np.array(grip)
    detail=dict(meta);detail['residual_action']=action.tolist();detail['residual_limits']=dict(zip(ACTION_NAMES,ACTION_LIMITS.tolist()))
    return result,detail,detach+float(offsets[4])


def gates(report):
    return {'balance':not report['fell'] and report['maximum_ankle_horizontal_displacement_m']<.015,
            'hand_tracking':max(report['rms_error_m'][k] for k in ['left_wrist','right_wrist'])<.08,
            'contact_transfer':report['grasp_transfer_valid'] and report['hand_contact_duration_after_detach_s']>0,
            'contact_penetration':report['maximum_hand_ball_contact_penetration_m']<.003,
            'preparation_force':report['peak_grasp_force_n']<150,
            'bounded_actuation':report['maximum_torque_limit_fraction']<=1.000001 and report['max_external_applied_force']==0,
            'stable_solver':report['qp_fallback_count']==0 and not any(report['mujoco_warning_counts']),
            'followthrough':max(report.get('followthrough_final_wrist_error_m',{'unknown':float('inf')}).values())<.06,
            'continuous_detach':report['grasp_detach']['qpos_change']==0 and report['grasp_detach']['qvel_change']==0}


def run(output=ROOT/'runs/spl_residual_ready',probes=True,render=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    ref,info,meta=build_reference(push_duration=.07,push_angle=1.1)
    np.savez_compressed(output/'nominal_reference.npz',**ref)
    (output/'nominal_reference.json').write_text(json.dumps({'source':info,'adaptation':meta,'controller':OPTIONS,'detach_motion_time_s':.99},indent=2),encoding='utf-8')
    samples=[('nominal',np.zeros(5))]
    if probes:
        for axis in [1,4]:
            for sign in [-1,1]:
                action=np.zeros(5);action[axis]=sign*.5;samples.append((ACTION_NAMES[axis]+('_minus' if sign<0 else '_plus'),action))
    reports=[]
    for name,action in samples:
        corrected,detail,detach=residual_reference(ref,meta,action)
        destination=output if name=='nominal' else output/'probes'/name
        report=shoot_contact(destination,prepared=(corrected,info,detail),detach=detach,controller_options=OPTIONS,quiet=True)
        checks=gates(report)
        reports.append({'name':name,'action':action.tolist(),'gates':checks,'release':report['release'],
                        'apex_position_m':report['apex_position_m'],'scored':report['scored']})
        print(json.dumps({'name':name,'gates':checks,'release_velocity':report['release']['linear_velocity_m_s'],'apex':report['apex_position_m']},ensure_ascii=False),flush=True)
    velocities=np.array([r['release']['linear_velocity_m_s'] for r in reports]);sensitivity=np.linalg.norm(velocities[1:]-velocities[0],axis=1)
    nominal=reports[0];physics_ready=all(nominal['gates'].values())
    summary={'training_started':False,'physics_ready':physics_ready,'bounded_action_contract_ready':True,
             'trajectory_sensitivity_observed':bool(len(sensitivity) and np.any(sensitivity>.05)),
             'scoring_curriculum_ready':False,'action_names':ACTION_NAMES,'action_limits':ACTION_LIMITS.tolist(),
             'action_units':['rad','rad','rad','rad','s'],'probe_velocity_changes_m_s':sensitivity.tolist(),'runs':reports,
             'failed_nominal_gates':[k for k,v in nominal['gates'].items() if not v],
             'next_required':'Improve upward launch speed and aim before sparse scoring reward; keep reference and balance gates.',
             'reference_sha256':hashlib.sha256((output/'nominal_reference.npz').read_bytes()).hexdigest()}
    (output/'readiness.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    from analyze_residual import analyze
    summary=analyze(output)
    if render:
        from render_shooting import render as render_result
        render_result(output)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'runs/spl_residual_ready');p.add_argument('--no-probes',action='store_true');p.add_argument('--render',action='store_true')
    a=p.parse_args();run(a.output,not a.no_probes,a.render)
