"""Held preparation followed by real hand-ball contact and natural takeoff."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,hashlib,xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.interpolate import CubicSpline
from shoot_spl import ROOT,REFERENCE,MASS,RADIUS,HOOP,RIM_INNER,scene_xml,numbers,ScoreDetector
from track_spl import motor_model,Controller,reference_at
from contact_reference import build_reference,retime_arm_lift,GRIP_OFFSET,PUSH_START
from shooting_model import contact_motor_model,model_xml,save_model

OUTPUT=ROOT/'runs/spl_contact'


def contact_scene(timestep=.001):
    tree=ET.fromstring(scene_xml(timestep))
    derived=ET.fromstring(model_xml());world=tree.find('worldbody');old=world.find("body[@name='pelvis']");index=list(world).index(old)
    world.remove(old);world.insert(index,derived.find("worldbody/body[@name='pelvis']"))
    tree.remove(tree.find('actuator'));tree.append(derived.find('actuator'))
    for g in tree.iter('geom'):
        if g.get('contype')=='0':continue
        name=g.get('name','')
        if any(part in name for part in ['wrist','index','middle','ring','pinky','thumb']):
            g.set('contype','4');g.set('conaffinity','7')
        elif name=='ball_geom':g.set('contype','2');g.set('conaffinity','7')
        else:g.set('contype','1');g.set('conaffinity','7')
    contact=tree.find('contact')
    if contact is None:contact=ET.SubElement(tree,'contact')
    for g in tree.iter('geom'):
        if g.get('contype')=='4':
            ET.SubElement(contact,'pair',geom1=g.get('name'),geom2='ball_geom',condim='4',
                          friction='.8 .8 .005 .0001 .0001',solref='.002 1',solimp='.98 .999 .0001')
    tree.find("equality/weld").set('relpose',numbers(np.r_[GRIP_OFFSET,1,0,0,0]))
    return ET.tostring(tree,encoding='unicode')


def attached_model(xml):
    tree=ET.fromstring(xml);weld=tree.find('equality/weld');pose=np.fromstring(weld.get('relpose'),sep=' ')
    tree.remove(tree.find('equality'));world=tree.find('worldbody');ball=world.find("body[@name='basketball']")
    world.remove(ball);ball.remove(ball.find('freejoint'));ball.set('pos',numbers(pose[:3]));ball.set('quat',numbers(pose[3:]))
    tree.find(".//body[@name='right_wrist']").append(ball)
    return mujoco.MjModel.from_xml_string(ET.tostring(tree,encoding='unicode'))


def ball_reaction(model,data,ballid):
    """Read contact reaction into an independent buffer, never qfrc_applied."""
    reaction=np.zeros(model.nv)
    for ci,c in enumerate(data.contact):
        if ballid not in (c.geom1,c.geom2):continue
        other=c.geom2 if c.geom1==ballid else c.geom1
        body=int(model.geom_bodyid[other])
        if body==0:continue
        force=np.zeros(6);mujoco.mj_contactForce(model,data,ci,force)
        sign=1. if c.geom1==ballid else -1.
        frame=c.frame.reshape(3,3)
        mujoco.mj_applyFT(model,data,sign*(frame.T@force[:3]),sign*(frame.T@force[3:]),c.pos,body,reaction)
    return reaction


def shoot_contact(output=OUTPUT,prepared=None,detach=.95,controller_options=None,quiet=False):
    ref,info,adapt=build_reference() if prepared is None else prepared
    output=Path(output);output.mkdir(parents=True,exist_ok=True);xml=contact_scene();(output/'scene.xml').write_text(xml,encoding='utf-8')
    save_model();m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);unloaded=contact_motor_model();cm=attached_model(xml);cd=mujoco.MjData(cm)
    nq,nv=cm.nq,cm.nv;d.qpos[:nq]=ref['qpos'][0];mujoco.mj_forward(m,d)
    d.qpos[nq:nq+3]=d.body('right_wrist').xpos+d.body('right_wrist').xmat.reshape(3,3)@GRIP_OFFSET
    d.qpos[nq+3:nq+7]=d.body('right_wrist').xquat;mujoco.mj_forward(m,d)
    cd.qpos[:]=d.qpos[:nq];mujoco.mj_forward(cm,cd)
    options=controller_options or {'arm_gain':180.,'arm_weight':2.,'foot_weight':60.,'acceleration_limit':1000.}
    ctrl=Controller(cm,cd,**options);settle=.7;duration=float(ref['times'][-1]);total=settle+duration+2.
    target=mujoco.MjData(unloaded);wrist_points=[];left_points=[]
    for q in ref['qpos']:
        target.qpos[:]=q;mujoco.mj_forward(unloaded,target);wrist_points.append(target.body('right_wrist').xpos.copy());left_points.append(target.body('left_wrist').xpos.copy())
    curve=CubicSpline(ref['times'],wrist_points,axis=0)
    left_curve=CubicSpline(ref['times'],left_points,axis=0)
    ballid=m.geom('ball_geom').id;floor=m.geom('floor').id;hand={g for g in range(m.ngeom) if m.geom(g).contype[0]==4}
    feet={m.geom(s+'_'+p+'_geom').id for s in ['left','right'] for p in ['ankle','foot']}
    ankle0=[d.body(s+'_ankle').xpos.copy() for s in ['left','right']];tracked=['left_wrist','right_wrist','left_elbow','right_elbow','pelvis']
    records=[];balltimes=[];positions=[];velocities=[];contacts=[];powers=[];hand_flags=[];last_contact={};score=ScoreDetector()
    released=False;candidate=None;release=None;off=False;detach_record=None;fall=None;nonfoot=set()
    slip=0.;pen=0.;humanpen=0.;heldpen=0.;handpen=0.;peak_torque=0.;fraction=0.;external=0.;grasp_error=0.;grasp_force=0.;work=0.;hand_contact_steps=0
    generalized=np.zeros(m.nv);seen_post_contact=False
    for step in range(round(total/m.opt.timestep)):
        if not off and d.time>=settle+detach-1e-9:
            beforeq=d.qpos.copy();beforev=d.qvel.copy();d.eq_active[:]=False;mujoco.mj_forward(m,d)
            detach_record={'time_s':float(d.time),'qpos_change':float(np.max(abs(d.qpos-beforeq))),'qvel_change':float(np.max(abs(d.qvel-beforev)))}
            off=True;cm=unloaded;cd=mujoco.MjData(cm);ctrl.model=cm;ctrl.data=cd;ctrl.lastJ=None;ctrl.lasttime=None;ctrl.task_history.clear()
        if step%5==0:
            cd.qpos[:]=d.qpos[:nq];cd.qvel[:]=d.qvel[:nv];cd.time=d.time;mujoco.mj_forward(cm,cd)
            q,v,a=reference_at(ref,d.time-settle)
            if d.time<settle or d.time>settle+duration:v=np.zeros(nv);a=np.zeros(nv)
            interaction=ball_reaction(m,d,ballid) if off else np.zeros(m.nv)
            t=float(np.clip(d.time-settle,0,duration));moving=settle<=d.time<=settle+duration
            task={'body':'right_wrist','local':np.zeros(3),'position':curve(t),
                  'velocity':curve(t,1) if moving else np.zeros(3),
                  'acceleration':curve(t,2) if moving else np.zeros(3),'weight':80.}
            left_task={'body':'left_wrist','local':np.zeros(3),'position':left_curve(t),
                       'velocity':left_curve(t,1) if moving else np.zeros(3),
                       'acceleration':left_curve(t,2) if moving else np.zeros(3),'weight':80.}
            d.ctrl[:]=ctrl.command(q,v,a,cartesian=[task,left_task],interaction=interaction[:nv] if off else None)
        before=d.qpos[nq:nq+3].copy();mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
        if not np.isfinite(d.qpos).all() or np.any(d.warning.number):raise RuntimeError('Unstable contact shot')
        ball=d.qpos[nq:nq+3].copy();velocity=d.qvel[nv:nv+3].copy()
        score.update(before,ball,float(d.time));hand_contact=False
        for c in d.contact:
            pen=max(pen,-float(c.dist))
            if ballid in (c.geom1,c.geom2):
                other=c.geom2 if c.geom1==ballid else c.geom1
                if other in hand:
                    hand_contact=True;handpen=max(handpen,-float(c.dist))
                if not off:heldpen=max(heldpen,-float(c.dist))
                if d.time-last_contact.get(other,-1)>.025:
                    contacts.append({'time_s':float(d.time),'geom':m.geom(other).name,'penetration_m':max(0.,-float(c.dist))});last_contact[other]=d.time
            else:humanpen=max(humanpen,-float(c.dist))
            if floor in (c.geom1,c.geom2):
                other=c.geom2 if c.geom1==floor else c.geom1
                if other not in feet and other!=ballid:nonfoot.add(m.geom(other).name)
        if off and hand_contact:
            hand_contact_steps+=1;candidate=None;seen_post_contact=True
            # A later recontact invalidates an earlier apparent takeoff.
            released=False;release=None
        if off and not hand_contact and not released and seen_post_contact:
            if candidate is None:
                candidate={'time_s':float(d.time),'position_m':ball.tolist(),'linear_velocity_m_s':velocity.tolist(),
                           'angular_velocity_rad_s':d.qvel[nv+3:nv+6].tolist(),'qpos_change_at_release':0.,'qvel_change_at_release':0.}
            if d.time-candidate['time_s']>=.02:release=candidate;released=True
        rootup=d.body('pelvis').xmat.reshape(3,3)[2,2]
        if fall is None and (d.qpos[2]<.55 or rootup<.5 or nonfoot):fall=float(d.time)
        for k,s in enumerate(['left','right']):slip=max(slip,float(np.linalg.norm(d.body(s+'_ankle').xpos[:2]-ankle0[k][:2])))
        if not off:
            grip=d.body('right_wrist').xpos+d.body('right_wrist').xmat.reshape(3,3)@GRIP_OFFSET
            grasp_error=max(grasp_error,float(np.linalg.norm(ball-grip)))
            ef=np.where(d.efc_type==mujoco.mjtConstraint.mjCNSTR_EQUALITY,d.efc_force,0.)
            mujoco.mj_mulJacTVec(m,d,generalized,ef);grasp_force=max(grasp_force,float(np.linalg.norm(generalized[nv:nv+3])))
        external=max(external,float(np.max(abs(d.qfrc_applied))),float(np.max(abs(d.xfrc_applied))))
        peak_torque=max(peak_torque,float(np.max(abs(d.actuator_force))));fraction=max(fraction,float(np.max(abs(d.actuator_force)/m.actuator_forcerange[:,1])))
        power=float(np.dot(d.actuator_force,d.actuator_velocity));work+=power*m.opt.timestep
        balltimes.append(float(d.time));positions.append(ball);velocities.append(velocity);powers.append(power);hand_flags.append(hand_contact)
        if step%8==0:
            q,_,_=reference_at(ref,d.time-settle);target.qpos[:]=q;mujoco.mj_forward(unloaded,target)
            errors=[float(np.linalg.norm(d.body(n).xpos-target.body(n).xpos)) for n in tracked]
            records.append((float(d.time),d.qpos.copy(),d.qvel.copy(),errors,not off))
    valid_transfer=release is not None and seen_post_contact
    if release is None:
        # Retain an invalid rollout for diagnosis; do not mistake a dropped ball
        # or persistent contact for a successful preparation-to-contact transfer.
        k=int(np.argmin(abs(np.array(balltimes)-detach_record['time_s'])))
        release={'time_s':balltimes[k],'position_m':positions[k].tolist(),'linear_velocity_m_s':velocities[k].tolist(),
                 'angular_velocity_rad_s':[0.,0.,0.],'qpos_change_at_release':0.,'qvel_change_at_release':0.,
                 'invalid_takeoff':True}
    times=np.array([r[0] for r in records]);errors=np.array([r[3] for r in records]);p=np.array(positions);bt=np.array(balltimes)
    flight=bt>=release['time_s'];apex=np.flatnonzero(flight)[np.argmax(p[flight,2])];motion=(times>=settle)&(times<=settle+duration)
    np.savez_compressed(output/'shooting.npz',times=times,qpos=np.array([r[1] for r in records]),qvel=np.array([r[2] for r in records]),
         errors=errors,error_names=tracked,grasp_active=[r[4] for r in records],ball_times=bt,ball_positions=p,ball_velocities=velocities,motor_power_w=powers,hand_contact=hand_flags)
    np.savez_compressed(output/'holding_reference.npz',**ref)
    np.savetxt(output/'ball_trajectory.csv',np.c_[bt,p,velocities],delimiter=',',header='time_s,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s',comments='')
    report={'status':'completed' if valid_transfer else 'invalid_grasp_transfer','grasp_transfer_valid':valid_transfer,
        'fell':fall is not None,'fall_time_s':fall,'scored':score.scored,'release':release,
        'grasp_detach':detach_record,'release_offset_s':0.,'duration_s':float(d.time),'settle_s':settle,'motion_duration_s':duration,'followthrough_hold_s':2.,
        'source_selection':info['selection'],'apex_position_m':p[apex].tolist(),'apex_time_s':float(bt[apex]),
        'hoop_center_m':HOOP.tolist(),'rim_inner_radius_m':RIM_INNER,'ball_radius_m':RADIUS,'ball_mass_kg':MASS,
        'downward_rim_crossings':score.crossings,'ball_contacts':contacts,'max_grasp_position_error_m':grasp_error,'peak_grasp_force_n':grasp_force,
        'maximum_held_ball_contact_penetration_m':heldpen,'maximum_hand_ball_contact_penetration_m':handpen,
        'maximum_human_contact_penetration_m':humanpen,'maximum_contact_penetration_m':pen,'maximum_excluded_hand_ball_overlap_m':0.,
        'hand_ball_collision_excluded':False,'release_mode':'natural_contact_separation',
        'hand_contact_duration_after_detach_s':hand_contact_steps*m.opt.timestep,
        'maximum_ankle_horizontal_displacement_m':slip,'nonfoot_floor_contacts':sorted(nonfoot),
        'peak_joint_torque_nm':peak_torque,'maximum_torque_limit_fraction':fraction,'max_external_applied_force':external,
        'minimum_pelvis_height_m':float(min(r[1][2] for r in records)),
        'rms_error_m':{n:float(np.sqrt(np.mean(errors[motion,i]**2))) for i,n in enumerate(tracked)},
        'followthrough_final_wrist_error_m':{n:float(np.mean(errors[times>total-.3,i])) for i,n in enumerate(tracked[:2])},
        'qp_fallback_count':ctrl.failures,'qp_status_counts':ctrl.statuses,'mujoco_warning_counts':d.warning.number.tolist(),
        'adaptation':adapt,'controller_options':options,'motor_work_j':work,'physics_timestep_s':float(m.opt.timestep),
        'reference_sha256':hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),'scene_sha256':hashlib.sha256(xml.encode()).hexdigest(),
        'limitations':['Temporary weld during preparation only; real hand-ball contacts after detach.',
          'Finger group targets are synthetic, not measured force closure.','Fixed feet and engineering masses/torques/friction.',
          'No reinforcement learning, imposed ball velocity, launch impulse, or external support force.']}
    (output/'shooting_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not quiet:print(json.dumps({k:report[k] for k in ['fell','scored','release','apex_position_m','hand_contact_duration_after_detach_s','maximum_hand_ball_contact_penetration_m','peak_grasp_force_n','maximum_ankle_horizontal_displacement_m','rms_error_m','adaptation']},indent=2),flush=True)
    return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--detach',type=float,default=.95)
    p.add_argument('--duration',type=float,default=.10);p.add_argument('--angle',type=float,default=.85)
    p.add_argument('--fast-lift',action='store_true')
    a=p.parse_args();ref,info,meta=build_reference(push_duration=a.duration,push_angle=a.angle)
    if a.fast_lift:ref,meta=retime_arm_lift(ref,meta)
    shoot_contact(a.output,prepared=(ref,info,meta),detach=a.detach)
