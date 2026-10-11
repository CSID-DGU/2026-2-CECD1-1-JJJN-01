"""Correct source-driven wrist motion and measure the remaining ball-speed gap."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from spl_motion import read_trial
from shoot_spl import ROOT,REFERENCE,prepare_reference,shoot
from release_reference import corrected_reference,hand_rotations
from track_spl import motor_model

OUTPUT=ROOT/'runs/spl_release'


def source_release(info):
    path=ROOT/info['selection']['path']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=info['selection']['source_sha256']:
        raise ValueError('Source changed since selection')
    _,fps,ids,_,points,_=read_trial(path)
    rotation=np.array(info['normalization']['rotation_rows'])
    ball=points['BALL']@rotation.T;wrist=points['RIGHT_WRIST']@rotation.T
    initial=info['selection']['release_frame'];t=(ids-initial)/fps
    # Fit several post-separation windows to quantify marker noise sensitivity.
    estimates=[]
    for lo,hi in [(initial+6,initial+30),(initial+10,initial+36),(initial+16,initial+41)]:
        mask=(ids>=lo)&(ids<=hi)
        if mask.sum()<10:raise ValueError('Insufficient post-release ball observations')
        X=np.c_[np.ones(mask.sum()),t[mask]]
        coef=np.linalg.lstsq(X,ball[mask]-t[mask,None]**2*[0,0,-4.905],rcond=None)[0]
        residual=ball[mask]-(X@coef+t[mask,None]**2*[0,0,-4.905])
        estimates.append({'frames':[lo,hi],'velocity_at_initial':coef[1].tolist(),
                          'coordinate_rms_m':float(np.sqrt(np.mean(residual**2))), 'coef':coef})
    coef=estimates[0]['coef'];prediction=coef[0]+t[:,None]*coef[1]+t[:,None]**2*[0,0,-4.905]
    errors=np.linalg.norm(ball-prediction,axis=1)
    candidates=[k for k in range(len(ids)-2) if initial-3<=ids[k]<=initial+12 and np.all(errors[k:k+3]<.04)
                and np.linalg.norm(ball[k]-wrist[k])>.22]
    if not candidates:raise ValueError('No consistent hand separation / free-flight transition')
    index=candidates[0];frame=int(ids[index]);release_time=(frame-info['selection']['start_frame'])/fps/info['playback_speed']
    v=coef[1]+[0,0,-9.81*t[index]]
    # Align observed flight heading to the fixed +X court direction. Do not
    # scale the ball's metres or gravity by avatar limb proportions.
    target=np.array([np.linalg.norm(v[:2]),0.,v[2]])
    sensitivity=[]
    for fit in estimates:
        velocity=np.array(fit['velocity_at_initial'])+[0,0,-9.81*t[index]]
        sensitivity.append(float(np.linalg.norm(velocity)));fit.pop('coef')
    report={'initial_release_frame':initial,'estimated_release_frame':frame,
            'release_motion_time_s':release_time,'frame_uncertainty_note':'Heuristic, approximately several frames; not ground-truth contact annotation.',
            'source_release_velocity_m_s':v.tolist(),'aligned_target_velocity_m_s':target.tolist(),
            'aligned_target_speed_m_s':float(np.linalg.norm(target)),
            'source_wrist_speed_m_s':float(np.linalg.norm(np.gradient(wrist,1/fps,axis=0)[index])),
            'source_wrist_velocity_m_s':np.gradient(wrist,1/fps,axis=0)[index].tolist(),
            'flight_fit_windows':estimates,'speed_sensitivity_m_s':sensitivity,
            'heading_alignment_rad':float(-np.arctan2(v[1],v[0])),
            'velocity_scale':'Observed metres/second, without body-length scaling; horizontal heading aligned to court +X.'}
    palm=hand_rotations(points,rotation,motor_model())
    return report,{'ids':ids,'time':t,'ball':ball,'wrist':wrist,'fit':prediction,'fit_error':errors,
                   'motion_times':(ids-info['selection']['start_frame'])/fps/info['playback_speed'],
                   'palm_rotation':palm}


def run(output=OUTPUT,weight=40.,render=False,release_delay=0.):
    if not all(np.isfinite(x) for x in [weight,release_delay]):
        raise ValueError('Calibration parameters must be finite')
    if weight<0:raise ValueError('Require nonnegative task weight')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    info=json.loads(REFERENCE.with_suffix('.json').read_text(encoding='utf-8'))
    diagnostic,source=source_release(info)
    release=diagnostic['release_motion_time_s']
    base,info,_,adapt=prepare_reference(REFERENCE,release_motion=release)
    ref,task,parameters=corrected_reference(base,release,source['motion_times'],source['palm_rotation'],weight)
    velocity=np.array(parameters['reference_grip_velocity_m_s'])
    adapt={**adapt,'release_correction':parameters,'source_release_analysis':diagnostic}
    report=shoot(REFERENCE,output,release_offset=release_delay,prepared=(ref,info,release,adapt),cartesian_task=task,quiet=True,
                 controller_options={'arm_gain':180.,'arm_weight':2.,'foot_weight':60.,'acceleration_limit':250.})
    actual=np.array(report['release']['linear_velocity_m_s']);report['release_velocity_error_m_s']=float(np.linalg.norm(actual-velocity))
    report['release_speed_m_s']=float(np.linalg.norm(actual));report['release_target_velocity_m_s']=velocity.tolist()
    (output/'shooting_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (output/'source_release_analysis.json').write_text(json.dumps(diagnostic,indent=2),encoding='utf-8')
    np.savez_compressed(output/'source_release_analysis.npz',**source)
    from release_diagnostics import diagnose
    report['release_diagnostics']=diagnose(output)
    print(json.dumps({k:report[k] for k in ['fell','release','release_speed_m_s','release_velocity_error_m_s','apex_position_m','rms_error_m','qp_fallback_count','peak_grasp_force_n','maximum_ankle_horizontal_displacement_m','maximum_held_ball_contact_penetration_m']},indent=2),flush=True)
    print(json.dumps(parameters,indent=2),flush=True)
    if render:
        from render_shooting import render as render_shot
        render_shot(output)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--weight',type=float,default=40.);p.add_argument('--render',action='store_true')
    p.add_argument('--release-delay',type=float,default=0.)
    args=p.parse_args();run(**vars(args))
