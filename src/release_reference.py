"""Small, source-driven wrist pose corrections around release, with form bounds."""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.optimize import least_squares
from shoot_spl import motor_model,grasp_offset,scene_xml
from track_spl import reference_at


def hand_rotations(points,rotation,model):
    """Palm rigid-frame proxy from wrist/index/middle/pinky MCP landmarks.

    This estimates a palm frame, not measured finger articulation or contact.
    Direction normalization avoids fitting athlete/avatar finger lengths.
    """
    d=mujoco.MjData(model);d.qpos[:]=model.key('standing').qpos;mujoco.mj_forward(model,d)
    wrist=d.body('right_wrist');local=[];observed=[]
    for body,key in [('right_index1','RIGHT_SECOND_FINGER_MCP'),('right_middle1','RIGHT_THIRD_FINGER_MCP'),
                     ('right_pinky1','RIGHT_FIFTH_FINGER_MCP')]:
        v=wrist.xmat.reshape(3,3).T@(d.body(body).xpos-wrist.xpos)
        local.append(v/np.linalg.norm(v))
        v=(points[key]-points['RIGHT_WRIST'])@rotation.T
        if np.any(np.linalg.norm(v,axis=1)<1e-5):raise ValueError('Degenerate palm landmarks')
        observed.append(v/np.linalg.norm(v,axis=1,keepdims=True))
    local=np.array(local);observed=np.stack(observed,axis=1);result=[]
    for target in observed:
        u,_,vt=np.linalg.svd(local.T@target)
        fix=np.diag([1.,1.,np.linalg.det(vt.T@u.T)])
        result.append(vt.T@fix@u.T)
    return np.array(result)


def envelope(t,start,release,end):
    """C2 zero at window edges, full correction near release."""
    u=np.clip((t-start)/.18,0,1);v=np.clip((end-t)/.18,0,1)
    smooth=lambda x:x**3*(10+x*(-15+6*x))
    return smooth(u)*smooth(v)


def corrected_reference(base,release,hand_times,hand_rotation,weight=40.):
    m=motor_model();d=mujoco.MjData(m);off=grasp_offset(m)
    start=max(0.,release-.45);end=min(float(base['times'][-1]),release+.40)
    raw_times=base['times'];idx=list(base['point_names']).index('right_wrist')
    source_curve=CubicSpline(raw_times,base['source_points'][:,idx],axis=0)
    nearest=lambda t:int(np.argmin(abs(hand_times-t)))
    q0,_,_=reference_at(base,start);d.qpos[:]=q0;mujoco.mj_forward(m,d)
    registration=hand_rotation[nearest(start)].T@d.body('right_wrist').xmat.reshape(3,3)
    qr,_,_=reference_at(base,release);d.qpos[:]=qr;mujoco.mj_forward(m,d)
    translation=d.body('right_wrist').xpos-source_curve(release)
    right=[j for j in range(m.njnt) if m.joint(j).name.startswith(('right_shoulder_','right_elbow_','right_wrist_'))]
    adr=m.jnt_qposadr[right];dv=m.jnt_dofadr[right];limits=m.jnt_range[right]
    reach=np.array([.15,.15,.15,.20,.65,.65,.45])
    cm=mujoco.MjModel.from_xml_string(scene_xml());cd=mujoco.MjData(cm)
    ballid=cm.geom('ball_geom').id
    other=[cm.geom(n+'_geom').id for n in ['head','neck','spine1','spine2','spine3','right_shoulder','right_elbow','left_elbow']]
    corrections=[];previous=base['qpos'][0,adr].copy()
    for i,t in enumerate(raw_times):
        q=base['qpos'][i];d.qpos[:]=q;mujoco.mj_forward(m,d)
        if not start<=t<=end:corrections.append(np.zeros(len(adr)));previous=q[adr];continue
        target_position=source_curve(t)+translation
        target_rotation=hand_rotation[nearest(t)]@registration
        def residual(x):
            d.qpos[:]=q;d.qpos[adr]=x;mujoco.mj_forward(m,d)
            rot=d.body('right_wrist').xmat.reshape(3,3)
            error=Rotation.from_matrix(target_rotation@rot.T).as_rotvec()
            cd.qpos[:m.nq]=d.qpos;cd.qpos[m.nq:m.nq+3]=d.body('right_wrist').xpos+rot@off
            mujoco.mj_kinematics(cm,cd)
            clear=[60*min(mujoco.mj_geomDistance(cm,cd,ballid,g,.06,None)-.005,0) for g in other]
            return np.r_[15*(d.body('right_wrist').xpos-target_position),.30*error,
                         .25*(x-q[adr]),.10*(x-previous),clear]
        lo=np.maximum(limits[:,0],q[adr]-reach);hi=np.minimum(limits[:,1],q[adr]+reach)
        fit=least_squares(residual,np.clip(previous,lo+1e-7,hi-1e-7),bounds=(lo,hi),max_nfev=45)
        corrections.append(fit.x-q[adr]);previous=fit.x
    delta=savgol_filter(np.array(corrections),9,3,axis=0)
    delta=np.clip(delta,-reach,reach);curve=CubicSpline(raw_times,delta,axis=0)
    # Dense sampling permits derivatives of the correction without differentiating
    # the original quaternion, body/leg trajectory, or noisy source markers again.
    times=np.linspace(0,raw_times[-1],round(raw_times[-1]*240)+1)
    values=np.clip(curve(times),-.995*reach,.995*reach)*envelope(times,start,release,end)[:,None]
    baseline_angles=np.array([reference_at(base,t)[0][adr] for t in times])
    # Project the sampled correction first, then differentiate its C2 spline.
    # Clipping qpos after taking derivatives would make qvel/qacc inconsistent.
    values=np.clip(values,limits[:,0]-baseline_angles,limits[:,1]-baseline_angles)
    correction=CubicSpline(times,values,axis=0)
    qpos=[];qvel=[];qacc=[];wrist=[];grip=[];max_clip=0.
    for t in times:
        q,v,a=reference_at(base,t)
        q[adr]+=correction(t);v[dv]+=correction(t,1);a[dv]+=correction(t,2)
        # SG/interpolation can overshoot near a joint limit. Clip and report it.
        clipped=np.clip(q[adr],limits[:,0],limits[:,1]);max_clip=max(max_clip,float(np.max(abs(q[adr]-clipped))))
        q[adr]=clipped
        qpos.append(q);qvel.append(v);qacc.append(a)
        d.qpos[:]=q;mujoco.mj_forward(m,d)
        wrist.append(d.body('right_wrist').xpos.copy());grip.append(d.body('right_wrist').xpos+d.body('right_wrist').xmat.reshape(3,3)@off)
    wrist=np.array(wrist);grip=np.array(grip)
    wrist_curve=CubicSpline(times,wrist,axis=0)
    def task(t):
        w=float(envelope(t,start,release,end))
        if t>end:w=float(np.clip((t-end)/.15,0,1))
        if w<=0:return None
        clipped=float(np.clip(t,times[0],times[-1]));moving=times[0]<=t<=times[-1]
        return {'body':'right_wrist','local':np.zeros(3),'position':wrist_curve(clipped),
                'velocity':wrist_curve(clipped,1) if moving else np.zeros(3),
                'acceleration':wrist_curve(clipped,2) if moving else np.zeros(3),'weight':weight*w}
    ref={'times':times,'qpos':np.array(qpos),'qvel':np.array(qvel),'qacc':np.array(qacc),
         'wrist_target':wrist,'grip_target':grip}
    meta={'method':'Source wrist path and MCP palm-frame proxy; bounded pose correction with C2 window.',
          'correction_window_s':[start,end],'task_weight':weight,'right_arm_delta_bounds_rad':reach.tolist(),
          'maximum_right_arm_delta_rad':np.max(abs(values),axis=0).tolist(),
          'maximum_reference_joint_clip_rad':max_clip,
          'source_adapted_wrist_velocity_m_s':source_curve(release,1).tolist(),
          'reference_wrist_velocity_m_s':wrist_curve(release,1).tolist(),
          'reference_grip_velocity_m_s':CubicSpline(times,grip,axis=0)(release,1).tolist(),
          'body_and_leg_reference_unchanged':True,'left_arm_reference_unchanged':True,
          'palm_frame_note':'Rigid frame estimated from three MCP landmarks; registered at window start. Finger force/contact is not inferred.',
          'ball_speed_is_not_an_arm_target':True}
    return ref,task,meta
