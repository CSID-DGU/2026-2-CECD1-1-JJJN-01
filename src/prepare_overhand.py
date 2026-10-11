"""Separate overhand preparation candidate; never overwrite a live PPO bundle."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json
from pathlib import Path
import numpy as np
import mujoco
from humanoid_model import ROOT
from contact_reference import build_reference,GRIP_OFFSET
from contact_shot import shoot_contact
from shooting_model import contact_motor_model
from prepare_residual import OPTIONS,gates


def run(output=ROOT/'runs/spl_overhand',render=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    ref,info,meta=build_reference(push_duration=.07,push_angle=1.1,preparation='overhand')
    m=contact_motor_model();d=mujoco.MjData(m);d.qpos[:]=ref['qpos'][0];mujoco.mj_forward(m,d)
    hand=d.site('right_palm').xpos.copy();ball=d.body('right_wrist').xpos+d.body('right_wrist').xmat.reshape(3,3)@GRIP_OFFSET
    meta['initial_palm_minus_ball_height_m']=float(hand[2]-ball[2])
    meta['gather_orientation_note']='Downward-facing palm above ball; lift turn .10 to .86 s; no late heading spin.'
    np.savez_compressed(output/'nominal_reference.npz',**ref)
    (output/'nominal_reference.json').write_text(json.dumps({'source':info,'adaptation':meta,'controller':OPTIONS,'detach_motion_time_s':.99},indent=2),encoding='utf-8')
    report=shoot_contact(output,prepared=(ref,info,meta),detach=.99,controller_options=OPTIONS,quiet=True)
    checks=gates(report);summary={'physics_ready':all(checks.values()),'gates':checks,'initial_palm_minus_ball_height_m':meta['initial_palm_minus_ball_height_m'],
            'fell':report['fell'],'release':report['release'],'apex_m':report['apex_position_m'][2],
            'wrist_rms_m':report['rms_error_m'],'grasp_force_n':report['peak_grasp_force_n'],'foot_slip_m':report['maximum_ankle_horizontal_displacement_m'],
            'adaptation':meta,'training_bundle_changed':False}
    (output/'preparation_review.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)
    if render:
        from render_shooting import render as render_result
        render_result(output)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'runs/spl_overhand');p.add_argument('--render',action='store_true');a=p.parse_args();run(a.output,a.render)
