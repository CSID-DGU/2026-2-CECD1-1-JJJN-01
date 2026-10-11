"""Functional shooting palm frame, consistent with the SMPL+H finger flexion axes."""
from pathlib import Path
import hashlib,json
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline,BPoly
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter
from shoot_spl import REFERENCE,MODEL_PATH,motor_model,scene_xml
from track_spl import reference_at
from shooting_model import contact_motor_model,expand_reference

GRIP_OFFSET=np.array([-.005,-.13,.128])
PUSH_START=.98
PUSH_DURATION=.10


def smoothstep(x):
    x=np.clip(x,0,1)
    return x**3*(10+x*(-15+6*x))


def retime_arm_lift(ref,meta):
    """Preserve the reference arm path, with a slower set and faster final lift."""
    m=contact_motor_model();times=ref['times'];knots=np.array([0.,.70,.87,1.07,1.35,float(times[-1])])
    phases=np.array([0.,.55,.60,1.10,1.35,float(times[-1])]);rates=np.array([1.,.2,.8,1.,1.,1.])
    clock=BPoly.from_derivatives(knots,[[p,v,0.] for p,v in zip(phases,rates)])
    if np.min(clock(times,1))<0:raise ValueError('Arm clock must be monotone')
    joints=[j for j in range(1,m.njnt) if m.joint(j).name.startswith(('right_shoulder','left_shoulder','right_elbow','left_elbow',
             'right_forearm','left_forearm','right_wrist','left_wrist','right_index','right_middle','right_ring','right_pinky','right_thumb',
             'left_index','left_middle','left_ring','left_pinky','left_thumb'))]
    qa=m.jnt_qposadr[joints];va=m.jnt_dofadr[joints]
    curve=CubicSpline(times,ref['qpos'][:,qa],axis=0);phase=clock(times);rate=clock(times,1);acc=clock(times,2)
    result={k:v.copy() for k,v in ref.items()};result['qpos'][:,qa]=curve(phase)
    result['qvel'][:,va]=curve(phase,1)*rate[:,None]
    result['qacc'][:,va]=curve(phase,2)*rate[:,None]**2+curve(phase,1)*acc[:,None]
    for j in joints:result['qpos'][:,m.jnt_qposadr[j]]=np.clip(result['qpos'][:,m.jnt_qposadr[j]],*m.jnt_range[j])
    data=mujoco.MjData(m);grip=[]
    for q in result['qpos']:
        data.qpos[:]=q;mujoco.mj_forward(m,data);grip.append(data.body('right_wrist').xpos+data.body('right_wrist').xmat.reshape(3,3)@GRIP_OFFSET)
    result['grip_target']=np.array(grip)
    start=brentq(lambda t:float(clock(t))-meta['push_start_s'],0,float(times[-1]))
    stop=brentq(lambda t:float(clock(t))-(meta['push_start_s']+meta['push_duration_s']),0,float(times[-1]))
    meta={**meta,'arm_phase_clock':{'time_knots_s':knots.tolist(),'source_phase_knots_s':phases.tolist(),
          'phase_rates':rates.tolist(),'maximum_phase_rate':float(np.max(rate))},
          'original_push_start_s':meta['push_start_s'],'push_start_s':start,'push_duration_s':stop-start,
          'timing_note':'Arm path retained with changed timing; torso and legs keep original timing.'}
    return result,meta


def build_reference(path=REFERENCE,push_duration=PUSH_DURATION,push_angle=.85,preparation='underhand',guide_withdraw_start=.74):
    if preparation not in ['underhand','overhand']:raise ValueError('Unknown preparation hand orientation')
    path=Path(path);info=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if info['model_sha256']!=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest():raise ValueError('Model changed; retarget first')
    with np.load(path) as z:original={k:z[k].copy() for k in z.files}
    m=contact_motor_model();original=expand_reference(original,m);d=mujoco.MjData(m)
    right=[j for j in range(m.njnt) if m.joint(j).name.startswith(('right_shoulder_','right_elbow_','right_wrist_','right_forearm_'))]
    left=[j for j in range(m.njnt) if m.joint(j).name.startswith(('left_shoulder_','left_elbow_','left_wrist_','left_forearm_'))]
    ra=m.jnt_qposadr[right];la=m.jnt_qposadr[left];rb=m.jnt_range[right];lb=m.jnt_range[left]
    cocked=np.array([[0,1,0],[-1,0,0],[0,0,1.]])
    front=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    guide=np.array([[1,0,0],[0,0,-1],[0,1,0.]])
    # Use the same expanded joint ordering for geometric clearance evaluation.
    import xml.etree.ElementTree as ET
    from shooting_model import model_xml
    collision_tree=ET.fromstring(scene_xml());new_tree=ET.fromstring(model_xml())
    world=collision_tree.find('worldbody');old_pelvis=world.find("body[@name='pelvis']");index=list(world).index(old_pelvis)
    world.remove(old_pelvis);world.insert(index,new_tree.find("worldbody/body[@name='pelvis']"))
    collision_tree.remove(collision_tree.find('actuator'));collision_tree.append(new_tree.find('actuator'))
    collision_model=mujoco.MjModel.from_xml_string(ET.tostring(collision_tree,encoding='unicode'));collision_data=mujoco.MjData(collision_model)
    ball_id=collision_model.geom('ball_geom').id
    avoid=[collision_model.geom(n+'_geom').id for n in ['pelvis','spine1','spine2','spine3','neck','head','right_shoulder','right_elbow','left_elbow']]
    previous=None;previous_left=None;qposes=[];wrist_error=[];elbow_error=[];orientation_error=[]
    for i,t in enumerate(original['times']):
        q=original['qpos'][i].copy();d.qpos[:]=q;mujoco.mj_forward(m,d)
        wrist=d.body('right_wrist').xpos.copy();elbow=d.body('right_elbow').xpos.copy()
        theta=push_angle*smoothstep((t-PUSH_START)/push_duration)
        if preparation=='overhand':
            # Palm faces down above the ball at preparation, then turns along
            # the lift over .76s. Avoid the old late 180-degree heading spin.
            gather=np.pi*(1-smoothstep((t-.10)/.76))
            target_rotation=Rotation.from_rotvec([0,theta+gather,0]).as_matrix()@cocked
        else:
            heading=np.pi*smoothstep((t-.60)/.28)
            target_rotation=Rotation.from_rotvec([0,theta,0]).as_matrix()@Rotation.from_rotvec([0,0,heading]).as_matrix()@front
        def residual(x):
            d.qpos[:]=q;d.qpos[ra]=x;mujoco.mj_forward(m,d)
            rot=d.body('right_wrist').xmat.reshape(3,3)
            collision_data.qpos[:m.nq]=d.qpos;collision_data.qpos[m.nq:m.nq+3]=d.body('right_wrist').xpos+rot@GRIP_OFFSET
            mujoco.mj_kinematics(collision_model,collision_data)
            clear=[60*min(mujoco.mj_geomDistance(collision_model,collision_data,ball_id,g,.08,None)-.015,0) for g in avoid]
            return np.r_[50*(d.body('right_wrist').xpos-wrist),15*(d.body('right_elbow').xpos-elbow),
                         1.2*(rot[:,2]-target_rotation[:,2]),1.0*(rot[:,1]-target_rotation[:,1]),
                         .025*(x-q[ra]),.10*(x-(previous if previous is not None else q[ra])),clear]
        initial=q[ra] if previous is None else previous
        fit=least_squares(residual,np.clip(initial,rb[:,0]+1e-6,rb[:,1]-1e-6),bounds=(rb[:,0],rb[:,1]),max_nfev=100)
        if i==0:
            # Forearm roll has multiple IK branches. Compare a second initial
            # wrist pose rather than accepting a back-of-hand support solution.
            alternative=initial.copy();alternative[-3:]=[-.8 if preparation=='overhand' else .8,0,0]
            second=least_squares(residual,np.clip(alternative,rb[:,0]+1e-6,rb[:,1]-1e-6),bounds=(rb[:,0],rb[:,1]),max_nfev=140)
            if np.linalg.norm(second.fun)<np.linalg.norm(fit.fun):fit=second
        q[ra]=fit.x;previous=fit.x.copy();d.qpos[:]=q;mujoco.mj_forward(m,d)
        wrist_error.append(np.linalg.norm(d.body('right_wrist').xpos-wrist));elbow_error.append(np.linalg.norm(d.body('right_elbow').xpos-elbow))
        orientation_error.append(np.linalg.norm(Rotation.from_matrix(target_rotation@d.body('right_wrist').xmat.reshape(3,3).T).as_rotvec()))
        ball=d.body('right_wrist').xpos+d.body('right_wrist').xmat.reshape(3,3)@GRIP_OFFSET
        guide_target=ball+[0,.21 if preparation=='overhand' else .17,0]
        withdraw=smoothstep((t-guide_withdraw_start)/.24) if preparation=='overhand' else smoothstep((t-(PUSH_START-.12))/.14)
        guide_target+=withdraw*np.array([0,.12,-.03])
        # Use the original guide elbow as a form prior and withdraw before push.
        d.qpos[:]=original['qpos'][i];mujoco.mj_forward(m,d);guide_elbow=d.body('left_elbow').xpos.copy()
        back=smoothstep((t-(PUSH_START+push_duration+.08))/.35)
        guide_target=(1-back)*guide_target+back*d.site('left_palm').xpos
        def guide_residual(x):
            d.qpos[:]=q;d.qpos[la]=x;mujoco.mj_forward(m,d)
            collision_data.qpos[:m.nq]=d.qpos;collision_data.qpos[m.nq:m.nq+3]=ball
            mujoco.mj_kinematics(collision_model,collision_data)
            left_geoms=[g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('left_') and
                        any(part in m.geom(g).name for part in ['wrist','elbow','index','middle','ring','pinky','thumb'])]
            clear=[100*min(mujoco.mj_geomDistance(collision_model,collision_data,ball_id,
                    collision_model.geom(m.geom(g).name).id,.06,None)-.004,0) for g in left_geoms]
            return np.r_[12*(d.site('left_palm').xpos-guide_target),1.5*(d.body('left_elbow').xpos-guide_elbow),
                         .4*(d.body('left_wrist').xmat.reshape(3,3)-guide).ravel(),
                         .04*(x-original['qpos'][i,la]),.08*(x-(previous_left if previous_left is not None else original['qpos'][i,la])),clear]
        fit=least_squares(guide_residual,np.clip(previous_left if previous_left is not None else q[la],lb[:,0]+1e-6,lb[:,1]-1e-6),bounds=(lb[:,0],lb[:,1]),max_nfev=65)
        q[la]=fit.x;previous_left=fit.x.copy()
        for j in range(m.njnt):
            name=m.joint(j).name
            if name.startswith('right_') and any(f in name for f in ['index','middle','ring','pinky']) and name.endswith('_x'):
                q[m.jnt_qposadr[j]]=-.15-.35*smoothstep((t-PUSH_START)/push_duration)+.42*smoothstep((t-(PUSH_START+push_duration))/.20)
        qposes.append(q)
    qposes=np.array(qposes)
    # Only the arm/finger corrections are smoothed. Body and leg samples remain
    # exactly those of the source retargeting.
    changed=np.r_[ra,la,[m.jnt_qposadr[j] for j in range(m.njnt) if m.joint(j).name.startswith('right_') and any(f in m.joint(j).name for f in ['index','middle','ring','pinky'])]]
    changed=np.unique(changed)
    qposes[:,changed]=savgol_filter(qposes[:,changed],5,2,axis=0)
    for j in range(1,m.njnt):qposes[:,m.jnt_qposadr[j]]=np.clip(qposes[:,m.jnt_qposadr[j]],*m.jnt_range[j])
    times=np.linspace(0,original['times'][-1],round(original['times'][-1]*240)+1)
    delta_curve=CubicSpline(original['times'],qposes[:,changed]-original['qpos'][:,changed],axis=0)
    dofs=np.array([m.jnt_dofadr[np.flatnonzero(m.jnt_qposadr==a)[0]] for a in changed])
    pos=[];vel=[];acc=[];grip=[]
    for t in times:
        q,v,a=reference_at(original,t);q[changed]+=delta_curve(t);v[dofs]+=delta_curve(t,1);a[dofs]+=delta_curve(t,2)
        for j in range(1,m.njnt):q[m.jnt_qposadr[j]]=np.clip(q[m.jnt_qposadr[j]],*m.jnt_range[j])
        pos.append(q);vel.append(v);acc.append(a);d.qpos[:]=q;mujoco.mj_forward(m,d)
        grip.append(d.body('right_wrist').xpos+d.body('right_wrist').xmat.reshape(3,3)@GRIP_OFFSET)
    ref={'times':times,'qpos':np.array(pos),'qvel':np.array(vel),'qacc':np.array(acc),'grip_target':np.array(grip)}
    meta={'palm_support_side':'+local Z, consistent with right finger negative-X flexion',
          'preparation':preparation,
          'guide_hand_clearance_offset_m':.21 if preparation=='overhand' else .17,
          'guide_withdraw_interval_s':[guide_withdraw_start,guide_withdraw_start+.24] if preparation=='overhand' else [PUSH_START-.12,PUSH_START+.02],
          'grip_offset_m':GRIP_OFFSET.tolist(),'push_start_s':PUSH_START,'push_duration_s':push_duration,'push_angle_rad':push_angle,
          'body_and_leg_reference_unchanged':True,'maximum_kinematic_wrist_position_change_m':float(max(wrist_error)),
          'maximum_kinematic_elbow_position_change_m':float(max(elbow_error)),
          'median_palm_orientation_error_rad':float(np.median(orientation_error)),
          'finger_motion_note':'Declared synthetic group flexion; not measured force closure.'}
    return ref,info,meta
