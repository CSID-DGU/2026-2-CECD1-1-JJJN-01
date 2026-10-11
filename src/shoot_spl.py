"""Loaded reference shot: temporary hand-ball weld, torque tracking, free flight.

No learned policy, prescribed ball trajectory, launch impulse, or root support.
The weld is a simplified grasp and does not demonstrate finger-force closure.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from pathlib import Path
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from humanoid_model import ROOT, MODEL_PATH
from track_spl import Controller, motor_model, reference_at

REFERENCE = ROOT/'data/motions/processed/free_throw_reference.npz'
OUTPUT = ROOT/'runs/spl_shooting'
RADIUS = .12
MASS = .62
HOOP = np.array([4.225, 0., 3.05])
RIM_INNER = .225
RIM_TUBE = .009


def numbers(values):
    return ' '.join(f'{x:.10g}' for x in values)


def grasp_offset(model):
    # Distal offset avoids the rounded end of the forearm capsule at the wrist.
    return model.site('right_palm').pos + [0, -.05, -(RADIUS+model.geom('right_wrist_geom').size[2]+.004)]


def prepare_reference(path,release_motion=None):
    """Preserve the torso/legs and adapt the two arms for ball clearance.

    The grip is fixed in the shooting palm frame. Guide-arm IK is an explicit
    extra adaptation, saved separately from the original no-ball reference.
    """
    info=json.loads(Path(path).with_suffix('.json').read_text(encoding='utf-8'))
    if info['model_sha256'] != hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest():
        raise ValueError('Model changed: retarget the source before shooting')
    with np.load(path) as source:
        ref={key:source[key].copy() for key in source.files}
    m=motor_model(); d=mujoco.MjData(m); offset=grasp_offset(m)
    joints=[j for j in range(m.njnt) if m.joint(j).name.startswith(('left_shoulder_', 'left_elbow_', 'left_wrist_'))]
    adr=m.jnt_qposadr[joints]; bounds=m.jnt_range[joints]
    release=(info['selection']['release_frame']-info['selection']['start_frame'])/info['selection']['fps']/info['playback_speed']
    if release_motion is not None:
        if not np.isfinite(release_motion) or not 0<release_motion<float(ref['times'][-1]):
            raise ValueError('Release must be within the selected motion')
        release=float(release_motion)
    right_joints=[j for j in range(m.njnt) if m.joint(j).name.startswith(('right_shoulder_', 'right_elbow_', 'right_wrist_'))]
    right_adr=m.jnt_qposadr[right_joints];right_bounds=m.jnt_range[right_joints]
    previous_right=ref['qpos'][0,right_adr].copy()
    previous=ref['qpos'][0,adr].copy(); corrections=[];right_corrections=[]
    torso_pairs={side:[(m.geom(side+'_'+part+'_geom').id,m.geom(torso+'_geom').id)
                      for part in ['shoulder','elbow','wrist'] for torso in ['pelvis','spine1','head']]
                 for side in ['left','right']}
    cross_pairs=[(m.geom('left_'+part+'_geom').id,m.geom('right_'+other+'_geom').id)
                 for part in ['wrist','index3','middle3','ring3','pinky3','thumb3']
                 for other in ['wrist','elbow','index3','middle3','pinky3']]
    def clearance(pairs):
        return [25*min(mujoco.mj_geomDistance(m,d,g1,g2,.05,None)-.002,0.) for g1,g2 in pairs]
    def ball_forearm_clearance(ball,side):
        g=m.geom(side+'_elbow_geom');center=d.geom_xpos[g.id];axis=d.geom_xmat[g.id].reshape(3,3)[:,2]
        projected=np.clip((ball-center)@axis,-g.size[1],g.size[1]);distance=np.linalg.norm(ball-center-axis*projected)
        return 25*min(distance-RADIUS-g.size[0]-.004,0.)
    for i,t in enumerate(ref['times']):
        original=ref['qpos'][i].copy();d.qpos[:]=original;mujoco.mj_forward(m,d)
        right_palm=d.site('right_palm').xpos.copy();right_elbow=d.body('right_elbow').xpos.copy()
        phase=np.clip(t/release,0,1)
        normal=np.array([.6*(1-phase),.3*(1-phase),.7+.3*phase]);normal/=np.linalg.norm(normal)
        def right_residual(x):
            d.qpos[:]=original;d.qpos[right_adr]=x;mujoco.mj_forward(m,d)
            return np.r_[12*(d.site('right_palm').xpos-right_palm),
                         .7*(d.body('right_elbow').xpos-right_elbow),
                         .5*(-d.body('right_wrist').xmat.reshape(3,3)[:,2]-normal),
                         .025*(x-original[right_adr]),.02*(x-previous_right),clearance(torso_pairs['right'])]
        reach=np.array([.35,.35,.35,.3,2.4,2.,1.4])
        lo=np.maximum(right_bounds[:,0],original[right_adr]-reach)
        hi=np.minimum(right_bounds[:,1],original[right_adr]+reach)
        fit=least_squares(right_residual,np.clip(previous_right,lo+1e-6,hi-1e-6),
                          bounds=(lo,hi),max_nfev=45)
        blend=np.clip((t-release)/.3,0,1);blend=blend*blend*(3-2*blend)
        right_corrections.append(float(np.linalg.norm(fit.x-original[right_adr])))
        original[right_adr]=(1-blend)*fit.x+blend*original[right_adr]
        previous_right=fit.x;ref['qpos'][i,right_adr]=original[right_adr]
        d.qpos[:]=original;mujoco.mj_forward(m,d)
        ball=d.body('right_wrist').xpos+d.body('right_wrist').xmat.reshape(3,3)@offset
        elbow=d.body('left_elbow').xpos.copy()
        direction=np.array([0.,1.,0.])
        target=ball+direction*(RADIUS+.018)
        def residual(x):
            d.qpos[:]=original;d.qpos[adr]=x;mujoco.mj_forward(m,d)
            normal=d.body('left_wrist').xmat.reshape(3,3)[:,2]
            # Parallel or antiparallel to the surface normal are equivalent for a box.
            return np.r_[12*(d.site('left_palm').xpos-target),
                         1.5*(d.body('left_elbow').xpos-elbow),
                         .15*np.cross(normal,direction), .025*(x-original[adr]), .02*(x-previous),
                         clearance(torso_pairs['left']+cross_pairs),ball_forearm_clearance(ball,'left')]
        reach=np.array([.55,.55,.55,.45,2.4,2.,1.4])
        lo=np.maximum(bounds[:,0],original[adr]-reach);hi=np.minimum(bounds[:,1],original[adr]+reach)
        result=least_squares(residual,np.clip(previous,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=45)
        # Fade to the original follow-through after release, with a smooth blend.
        blend=np.clip((t-release)/.3,0,1);blend=blend*blend*(3-2*blend)
        ref['qpos'][i,adr]=(1-blend)*result.x+blend*original[adr]
        previous=result.x;corrections.append(float(np.linalg.norm(ref['qpos'][i,adr]-original[adr])))
    ref['qpos'][:,adr]=savgol_filter(ref['qpos'][:,adr],7,2,axis=0)
    ref['qpos'][:,adr]=np.clip(ref['qpos'][:,adr],bounds[:,0],bounds[:,1])
    ref['qpos'][:,right_adr]=np.clip(savgol_filter(ref['qpos'][:,right_adr],7,2,axis=0),right_bounds[:,0],right_bounds[:,1])
    for i in range(len(ref['times'])-1):
        mujoco.mj_differentiatePos(m,ref['qvel'][i],ref['times'][i+1]-ref['times'][i],ref['qpos'][i],ref['qpos'][i+1])
    ref['qvel'][-1]=ref['qvel'][-2]
    ref['qvel']=savgol_filter(ref['qvel'],7,2,axis=0)
    ref['qacc']=np.gradient(ref['qvel'],ref['times'],axis=0)
    return ref, info, release, {'guide_arm_max_joint_vector_change_rad':max(corrections),
                              'shooting_arm_max_joint_vector_change_rad':max(right_corrections),
                              'body_and_leg_reference_unchanged':True,
                              'hand_adaptation':'Both arms adapted for ball clearance/palm orientation; blend back after release.'}


def scene_xml(timestep=.001):
    tree=ET.fromstring(MODEL_PATH.read_text(encoding='utf-8'))
    tree.find('option').set('timestep',str(timestep))
    tree.remove(tree.find('keyframe'))
    for a in tree.find('actuator'):
        a.tag='motor';a.attrib.pop('kp',None);a.attrib.pop('kv',None)
        a.set('ctrlrange',a.get('forcerange'))
    world=tree.find('worldbody')
    # Collision bit 2 identifies the ball, bit 4 the hands. The simplified
    # grasp excludes hand-ball collision throughout; other collisions remain.
    for g in tree.iter('geom'):
        name=g.get('name','')
        hand=any(part in name for part in ('wrist','index','middle','pinky','ring','thumb'))
        g.set('contype','4' if hand else '1');g.set('conaffinity','5' if hand else '7')
    ball=ET.SubElement(world,'body',name='basketball',pos='0 0 1')
    ET.SubElement(ball,'freejoint',name='ball_free')
    ET.SubElement(ball,'geom',name='ball_geom',type='sphere',size=str(RADIUS),mass=str(MASS),
                  rgba='.95 .36 .045 1',contype='2',conaffinity='3',friction='.65 .015 .001',
                  condim='3',solref='.008 .65')
    # Visual seams are non-colliding and massless.
    for axis in range(3):
        for i in range(48):
            pts=[]
            for angle in [2*np.pi*i/48,2*np.pi*(i+1)/48]:
                p=np.zeros(3);p[(axis+1)%3]=RADIUS*np.cos(angle);p[(axis+2)%3]=RADIUS*np.sin(angle);pts.extend(p)
            ET.SubElement(ball,'geom',type='capsule',fromto=numbers(pts),size='.0012',mass='0',
                          contype='0',conaffinity='0',rgba='.12 .06 .025 1',group='2')
    for i in range(48):
        pts=[]
        for angle in [2*np.pi*i/48,2*np.pi*(i+1)/48]:
            pts.extend(HOOP+[(RIM_INNER+RIM_TUBE)*np.cos(angle),(RIM_INNER+RIM_TUBE)*np.sin(angle),0])
        ET.SubElement(world,'geom',name=f'rim_{i}',type='capsule',fromto=numbers(pts),size=str(RIM_TUBE),
                      rgba='1 .25 .03 1',contype='1',conaffinity='7',friction='.5 .01 .001',solref='.008 .65')
    ET.SubElement(world,'geom',name='backboard',type='box',pos=numbers(HOOP+[.375,0,.45]),
                  size='.025 .9 .525',rgba='.75 .86 .93 .45',contype='1',conaffinity='7')
    ET.SubElement(world,'geom',name='hoop_post',type='capsule',fromto=numbers([5.1,0,0,5.1,0,3.6]),
                  size='.065',rgba='.18 .22 .3 1',contype='1',conaffinity='7')
    ET.SubElement(world,'geom',name='free_throw_marker',type='box',pos='0 0 .001',size='.025 .9 .001',
                  rgba='1 1 1 1',contype='0',conaffinity='0')
    equality=ET.SubElement(tree,'equality')
    offset=grasp_offset(motor_model())
    ET.SubElement(equality,'weld',name='grasp',body1='right_wrist',body2='basketball',
                  relpose=numbers(np.r_[offset,1,0,0,0]),solref='.006 1',solimp='.95 .99 .001')
    return ET.tostring(tree,encoding='unicode')


def loaded_control_model(xml):
    """Inverse dynamics includes the mass/inertia of the temporarily held ball."""
    tree=ET.fromstring(xml);tree.remove(tree.find('equality'))
    world=tree.find('worldbody');ball=world.find("body[@name='basketball']")
    world.remove(ball);ball.remove(ball.find('freejoint'))
    ball.set('pos',numbers(grasp_offset(motor_model())))
    tree.find(".//body[@name='right_wrist']").append(ball)
    return mujoco.MjModel.from_xml_string(ET.tostring(tree,encoding='unicode'))


class ScoreDetector:
    """Require downward entry and then the complete ball below the rim."""
    def __init__(self):
        self.entered=False;self.scored=False;self.crossings=[]

    def update(self,before,after,time):
        for height,kind in [(HOOP[2],'center'),(HOOP[2]-RADIUS,'bottom_clear')]:
            if before[2]>height>=after[2]:
                alpha=(before[2]-height)/(before[2]-after[2]);point=before+alpha*(after-before)
                inside=np.linalg.norm(point[:2]-HOOP[:2])<=RIM_INNER-RADIUS
                if kind=='center':
                    self.entered=bool(inside);self.crossings.append({'time_s':time,'position_m':point.tolist(),'inside':bool(inside)})
                elif self.entered:
                    self.scored=bool(inside) or self.scored;self.entered=False
        if after[2]>HOOP[2]+RADIUS:self.entered=False


def shoot(reference=REFERENCE, output=OUTPUT, release_offset=0., settle=.7, after_motion=2., timestep=.001,
          prepared=None,cartesian_task=None,quiet=False,controller_options=None):
    if not all(np.isfinite(x) for x in [release_offset,settle,after_motion,timestep]) or settle<0 or after_motion<0 or timestep<=0:
        raise ValueError('Require finite timings, nonnegative settle/hold, and a positive timestep')
    ref,info,release_motion,adaptation=prepare_reference(reference) if prepared is None else prepared
    duration=float(ref['times'][-1]);release_motion+=release_offset
    if not 0<release_motion<duration:raise ValueError('Release must be within the selected motion')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    xml=scene_xml(timestep);(output/'scene.xml').write_text(xml,encoding='utf-8')
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
    unloaded=motor_model();cm=loaded_control_model(xml);cd=mujoco.MjData(cm);target=mujoco.MjData(unloaded)
    nq,nv=cm.nq,cm.nv
    d.qpos[:nq]=ref['qpos'][0];mujoco.mj_forward(m,d)
    d.qpos[nq:nq+3]=d.body('right_wrist').xpos+d.body('right_wrist').xmat.reshape(3,3)@grasp_offset(cm)
    d.qpos[nq+3:nq+7]=d.body('right_wrist').xquat;d.qvel[:]=0;mujoco.mj_forward(m,d)
    cd.qpos[:]=d.qpos[:nq];mujoco.mj_forward(cm,cd);controller=Controller(cm,cd,**(controller_options or {}))
    release_time=settle+release_motion;total=settle+duration+after_motion
    detector=ScoreDetector();records=[];all_ball=[];all_times=[];all_vel=[];contacts=[]
    released=False;release_record=None;fall_time=None;max_grasp=0.;peak_torque=0.;slip=0.;max_pen=0.
    max_fraction=0.;max_external=0.;max_grip_load=0.;max_hand_overlap=0.;max_ball_pen=0.
    max_held_ball_pen=0.;max_human_pen=0.;ball_geom=m.geom('ball_geom').id;floor=m.geom('floor').id
    hand_geoms=[g for g in range(m.ngeom) if any(part in m.geom(g).name for part in ['wrist','index','middle','ring','pinky','thumb'])]
    last_contact_time={}
    ankle0=[d.body(s+'_ankle').xpos.copy() for s in ['left','right']]
    feet={m.geom(s+'_'+p+'_geom').id for s in ['left','right'] for p in ['foot','ankle']}
    nonfoot=set();tracked=['left_wrist','right_wrist','left_elbow','right_elbow','pelvis']
    generalized=np.zeros(m.nv);control_every=max(1,round(.005/timestep));record_every=max(1,round(1/120/timestep))
    for step in range(int(np.ceil(total/timestep))):
        if not released and d.time>=release_time-1e-9:
            before_q=d.qpos.copy();before_v=d.qvel.copy()
            d.eq_active[m.equality('grasp').id]=False
            mujoco.mj_forward(m,d)
            release_record={'time_s':float(d.time),'position_m':d.qpos[nq:nq+3].tolist(),
                            'linear_velocity_m_s':d.qvel[nv:nv+3].tolist(),
                            'angular_velocity_rad_s':d.qvel[nv+3:nv+6].tolist(),
                            'qpos_change_at_release':float(np.max(abs(d.qpos-before_q))),
                            'qvel_change_at_release':float(np.max(abs(d.qvel-before_v)))}
            released=True
            cm=unloaded;cd=mujoco.MjData(cm);cd.qpos[:]=d.qpos[:nq];cd.qvel[:]=d.qvel[:nv]
            mujoco.mj_forward(cm,cd)
            old_contacts=controller.contacts
            controller.model=cm;controller.data=cd
            controller.contacts=old_contacts;controller.lastJ=None;controller.lasttime=None
        if step%control_every==0:
            cd.qpos[:]=d.qpos[:nq];cd.qvel[:]=d.qvel[:nv];cd.time=d.time;mujoco.mj_forward(cm,cd)
            # Measure the internal grasp reaction. Load compensation comes from
            # the attached-ball inertia model, avoiding delayed-force feedback.
            ef=np.where(d.efc_type==mujoco.mjtConstraint.mjCNSTR_EQUALITY,d.efc_force,0.)
            mujoco.mj_mulJacTVec(m,d,generalized,ef)
            max_grip_load=max(max_grip_load,float(np.linalg.norm(generalized[nv:nv+3])))
            q,v,a=reference_at(ref,d.time-settle)
            if d.time<settle or d.time>settle+duration:v=np.zeros(nv);a=np.zeros(nv)
            task=cartesian_task(d.time-settle) if cartesian_task is not None else None
            d.ctrl[:]=controller.command(q,v,a,cartesian=task)
        previous_ball=d.qpos[nq:nq+3].copy()
        mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
        if not np.isfinite(d.qpos).all() or np.any(d.warning.number):raise RuntimeError('Unstable shooting simulation')
        ball=d.qpos[nq:nq+3].copy()
        if released:detector.update(previous_ball,ball,float(d.time))
        all_ball.append(ball);all_times.append(float(d.time));all_vel.append(d.qvel[nv:nv+3].copy())
        root_up=d.body('pelvis').xmat.reshape(3,3)[2,2]
        for c in d.contact:
            max_pen=max(max_pen,-float(c.dist))
            if ball_geom in (c.geom1,c.geom2):
                other=c.geom2 if c.geom1==ball_geom else c.geom1
                max_ball_pen=max(max_ball_pen,-float(c.dist))
                if not released:max_held_ball_pen=max(max_held_ball_pen,-float(c.dist))
                if d.time-last_contact_time.get(other,-1.)>.025:
                    contacts.append({'time_s':float(d.time),'geom':m.geom(other).name,'penetration_m':max(0.,-float(c.dist))})
                    last_contact_time[other]=d.time
            else:max_human_pen=max(max_human_pen,-float(c.dist))
            if floor in (c.geom1,c.geom2):
                other=c.geom2 if c.geom1==floor else c.geom1
                if other not in feet and other!=ball_geom:nonfoot.add(m.geom(other).name)
        if fall_time is None and (d.qpos[2]<.55 or root_up<.5 or nonfoot):fall_time=float(d.time)
        for k,s in enumerate(['left','right']):slip=max(slip,float(np.linalg.norm(d.body(s+'_ankle').xpos[:2]-ankle0[k][:2])))
        if not released:
            grip=d.body('right_wrist').xpos+d.body('right_wrist').xmat.reshape(3,3)@grasp_offset(cm)
            max_grasp=max(max_grasp,float(np.linalg.norm(ball-grip)))
        peak_torque=max(peak_torque,float(np.max(abs(d.actuator_force))))
        max_fraction=max(max_fraction,float(np.max(abs(d.actuator_force)/m.actuator_forcerange[:,1])))
        max_external=max(max_external,float(np.max(abs(d.qfrc_applied))),float(np.max(abs(d.xfrc_applied))))
        if step%record_every==0:
            if not released:
                max_hand_overlap=max(max_hand_overlap,max(-mujoco.mj_geomDistance(m,d,ball_geom,g,.1,None) for g in hand_geoms))
            q,_,_=reference_at(ref,d.time-settle);target.qpos[:]=q;mujoco.mj_forward(unloaded,target)
            errors=[np.linalg.norm(d.body(name).xpos-target.body(name).xpos) for name in tracked]
            records.append((float(d.time),d.qpos.copy(),d.qvel.copy(),errors,not released))
    times=np.array([r[0] for r in records]);errors=np.array([r[3] for r in records])
    motion=(times>=settle)&(times<=settle+duration);trajectory=np.array(all_ball);flight=np.array(all_times)>=release_record['time_s']
    np.savez_compressed(output/'shooting.npz',times=times,qpos=np.array([r[1] for r in records]),
                        qvel=np.array([r[2] for r in records]),errors=errors,error_names=tracked,
                        grasp_active=[r[4] for r in records],ball_times=all_times,ball_positions=trajectory,ball_velocities=all_vel)
    np.savez_compressed(output/'holding_reference.npz',**ref)
    np.savetxt(output/'ball_trajectory.csv',np.c_[all_times,trajectory,all_vel],delimiter=',',
               header='time_s,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s',comments='')
    apex_index=np.flatnonzero(flight)[np.argmax(trajectory[flight,2])]
    report={'status':'completed','fell':fall_time is not None,'fall_time_s':fall_time,'scored':detector.scored,
            'duration_s':float(d.time),'settle_s':settle,'motion_duration_s':duration,'followthrough_hold_s':after_motion,
            'release':release_record,'release_offset_s':release_offset,'source_selection':info['selection'],
            'apex_position_m':trajectory[apex_index].tolist(),'apex_time_s':all_times[apex_index],
            'hoop_center_m':HOOP.tolist(),'rim_inner_radius_m':RIM_INNER,'ball_radius_m':RADIUS,'ball_mass_kg':MASS,
            'downward_rim_crossings':detector.crossings,'ball_contacts':contacts,
            'max_grasp_position_error_m':max_grasp,'peak_grasp_force_n':max_grip_load,
            'maximum_excluded_hand_ball_overlap_m':max_hand_overlap,
            'hand_overlap_sample_interval_s':record_every*timestep,
            'maximum_ball_contact_penetration_m':max_ball_pen,
            'maximum_held_ball_contact_penetration_m':max_held_ball_pen,
            'maximum_human_contact_penetration_m':max_human_pen,
            'maximum_ankle_horizontal_displacement_m':slip,'maximum_contact_penetration_m':max_pen,
            'peak_joint_torque_nm':peak_torque,'maximum_torque_limit_fraction':max_fraction,
            'minimum_pelvis_height_m':float(min(r[1][2] for r in records)),
            'nonfoot_floor_contacts':sorted(nonfoot),'max_external_applied_force':max_external,
            'rms_error_m':{name:float(np.sqrt(np.mean(errors[motion,i]**2))) for i,name in enumerate(tracked)},
            'qp_status_counts':controller.statuses,'qp_fallback_count':controller.failures,
            'mujoco_warning_counts':d.warning.number.tolist(),'adaptation':adaptation,
            'reference_sha256':hashlib.sha256(Path(reference).read_bytes()).hexdigest(),
            'scene_sha256':hashlib.sha256(xml.encode('utf-8')).hexdigest(),'physics_timestep_s':timestep,
            'controller':'Floating-base torque QP with attached-ball inertia before release, unloaded inertia after release.',
            'controller_options':controller_options or {},
            'limitations':['Temporary right-hand weld approximates grasp, not finger contact control.',
                          'Hand-ball collisions excluded throughout; guide hand is a visual pose target, not a second grip.',
                          'Feet remain planted; original heel lift is not reconstructed.',
                          'Release is a source-frame heuristic; no learned release or launch impulse.',
                          'Ball/rim contact parameters are engineering assumptions; no calibrated bounce, drag or Magnus force.',
                          'No reinforcement learning or scoring optimization performed.']}
    (output/'shooting_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not quiet:print(json.dumps({k:v for k,v in report.items() if k!='ball_contacts'},indent=2),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=REFERENCE);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--release-offset',type=float,default=0.);p.add_argument('--render',action='store_true')
    a=p.parse_args();shoot(a.reference,a.output,a.release_offset)
    if a.render:
        from render_shooting import render
        render(a.output)


if __name__=='__main__':main()
