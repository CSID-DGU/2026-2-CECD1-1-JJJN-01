"""Geometric extension / trunk / head IK, using actual MuJoCo FK/Jacobians."""
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from examples.m5_upper_body_ik import joint_ids


def unit(v):return v/max(np.linalg.norm(v),1e-12)
def trunk_axis(m,d):
    hips=np.mean([d.xpos[m.body(n).id] for n in ('L_Hip','R_Hip')],axis=0);shoulders=np.mean([d.xpos[m.body(n).id] for n in ('L_Shoulder','R_Shoulder')],axis=0);return unit(shoulders-hips)
def calibration(m,d):
    E=d.xmat[m.body('R_Elbow').id].reshape(3,3);W=d.xmat[m.body('R_Wrist').id].reshape(3,3);f=unit(d.xpos[m.body('R_Middle1').id]-d.xpos[m.body('R_Wrist').id]);normal=-W[:,1];return dict(chest_rotation=d.xmat[m.body('Chest').id].reshape(3,3).copy(),trunk_q=np.array([d.qpos[int(m.joint(b+'_'+a).qposadr[0])] for b in ('Torso','Spine','Chest') for a in 'xyz']),finger_local=W.T@f,local_rotation=E.T@W,flexion_axis=unit(E.T@np.cross(f,normal)),forearm_rotation=E.copy(),forearm_axis=unit(d.xpos[m.body('R_Wrist').id]-d.xpos[m.body('R_Elbow').id]),trunk_axis=trunk_axis(m,d))
def extension_angle(m,d,cal):
    E=d.xmat[m.body('R_Elbow').id].reshape(3,3);W=d.xmat[m.body('R_Wrist').id].reshape(3,3);q=Rotation.from_matrix((E.T@W)@cal['local_rotation'].T).as_quat();return float((-2*np.arctan2(np.dot(q[:3],cal['flexion_axis']),q[3])+np.pi)%(2*np.pi)-np.pi)
def forearm_swing_twist(m,d,cal):
    E=d.xmat[m.body('R_Elbow').id].reshape(3,3);axis=unit(d.xpos[m.body('R_Wrist').id]-d.xpos[m.body('R_Elbow').id]);v=np.cross(cal['forearm_axis'],axis);angle=np.arccos(np.clip(cal['forearm_axis']@axis,-1,1));swing=Rotation.from_rotvec(unit(v)*angle) if np.linalg.norm(v)>1e-8 else Rotation.identity();delta=Rotation.from_matrix(E@cal['forearm_rotation'].T);q=(swing.inv()*delta).as_quat();twist=2*np.arctan2(q[:3]@cal['forearm_axis'],q[3]);twist=(twist+np.pi)%(2*np.pi)-np.pi;return float(np.degrees(angle)),float(np.degrees(twist))


def whole_body_ik(m,q,position,orientation,trunk_target,cal,extension,seed=None,iterations=35):
    names=[f'{b}_{a}' for b in ('Torso','Spine','Chest') for a in 'xyz']+[f'R_{b}_{a}' for b in ('Shoulder','Elbow','Wrist') for a in 'xyz'];ids=joint_ids(m,names);dofs=[int(m.joint(n).dofadr[0]) for n in names];bounds=np.array([m.jnt_range[m.joint(n).id] for n in names]);lo,hi=bounds[:,0]+1e-5,bounds[:,1]-1e-5;lo[:9]=np.maximum(lo[:9],cal['trunk_q']-.4);hi[:9]=np.minimum(hi[:9],cal['trunk_q']+.4);p=mujoco.MjData(m);p.qpos[:]=q;p.qpos[ids]=np.clip(q[ids] if seed is None else seed,lo,hi);anchor=p.qpos[ids].copy();w=m.body('R_Wrist').id;penalty=np.diag([.06]*9+[.018]*6+[.025]*3)
    for _ in range(iterations):
        mujoco.mj_kinematics(m,p);mujoco.mj_comPos(m,p);axis=trunk_axis(m,p);dp=position-p.xpos[w];dr=Rotation.from_matrix(orientation@p.xmat[w].reshape(3,3).T).as_rotvec();de=float((extension-extension_angle(m,p,cal)+np.pi)%(2*np.pi)-np.pi);jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv));mujoco.mj_jacBody(m,p,jp,jr,w)
        js=np.zeros((3,m.nv));jh=np.zeros((3,m.nv))
        for b in ('L_Shoulder','R_Shoulder'):
            temp=np.zeros((3,m.nv));rot=np.zeros((3,m.nv));mujoco.mj_jacBody(m,p,temp,rot,m.body(b).id);js+=temp*.5
        for b in ('L_Hip','R_Hip'):
            temp=np.zeros((3,m.nv));rot=np.zeros((3,m.nv));mujoco.mj_jacBody(m,p,temp,rot,m.body(b).id);jh+=temp*.5
        hs=np.mean([p.xpos[m.body(n).id] for n in ('L_Hip','R_Hip')],axis=0);ss=np.mean([p.xpos[m.body(n).id] for n in ('L_Shoulder','R_Shoulder')],axis=0);ja=(np.eye(3)-np.outer(axis,axis))@(js-jh)/max(np.linalg.norm(ss-hs),1e-8)
        v=np.cross(cal['trunk_axis'],trunk_target);angle=np.arccos(np.clip(cal['trunk_axis']@trunk_target,-1,1));trunk_rotation=(Rotation.from_rotvec(unit(v)*angle).as_matrix() if np.linalg.norm(v)>1e-8 else np.eye(3))@cal['chest_rotation'];chest=m.body('Chest').id;jch=np.zeros((3,m.nv));jcp=np.zeros((3,m.nv));mujoco.mj_jacBody(m,p,jcp,jch,chest);dc=Rotation.from_matrix(trunk_rotation@p.xmat[chest].reshape(3,3).T).as_rotvec()
        # Finite FK derivative of the calibrated extension component, wrist axes only.
        je=np.zeros(len(ids));base_ext=extension_angle(m,p,cal)
        for k in range(15,18):
            old=p.qpos[ids[k]];p.qpos[ids[k]]=old+1e-5;mujoco.mj_kinematics(m,p);je[k]=((extension_angle(m,p,cal)-base_ext+np.pi)%(2*np.pi)-np.pi)/1e-5;p.qpos[ids[k]]=old
        mujoco.mj_kinematics(m,p)
        J=np.vstack((35*jp[:,dofs],2*jr[:,dofs],5*ja[:,dofs],1.8*je,2*jch[:,dofs]));err=np.r_[35*dp,2*dr,5*(trunk_target-axis),1.8*de,2*dc];step=np.linalg.solve(J.T@J+penalty,J.T@err-penalty@(p.qpos[ids]-anchor));step*=min(1,.18/max(np.linalg.norm(step),1e-12));p.qpos[ids]=np.clip(p.qpos[ids]+step,lo,hi)
        if np.linalg.norm(dp)<.001 and np.linalg.norm(dr)<.02 and abs(de)<.02 and np.linalg.norm(trunk_target-axis)<.02:break
    mujoco.mj_kinematics(m,p);z=dict(position_error_mm=float(np.linalg.norm(position-p.xpos[w])*1000),orientation_error_deg=float(np.degrees(Rotation.from_matrix(orientation@p.xmat[w].reshape(3,3).T).magnitude())),extension_error_deg=float(np.degrees((extension-extension_angle(m,p,cal)+np.pi)%(2*np.pi)-np.pi)),trunk_error_deg=float(np.degrees(np.arccos(np.clip(trunk_target@trunk_axis(m,p),-1,1)))))
    return p.qpos.copy(),z


def head_ik(m,q,rim,seed=None):
    names=[f'{b}_{a}' for b in ('Neck','Head') for a in 'xyz'];ids=joint_ids(m,names);dofs=[int(m.joint(n).dofadr[0]) for n in names];p=mujoco.MjData(m);p.qpos[:]=q
    if seed is not None:p.qpos[ids]=seed
    bounds=np.array([m.jnt_range[m.joint(n).id] for n in names]);anchor=p.qpos[ids].copy();bid=m.body('Head').id
    for _ in range(15):
        mujoco.mj_kinematics(m,p);mujoco.mj_comPos(m,p);forward=p.xmat[bid].reshape(3,3)[:,2];desired=unit(rim-p.xpos[bid]);jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv));mujoco.mj_jacBody(m,p,jp,jr,bid);cross=np.array([[0,-forward[2],forward[1]],[forward[2],0,-forward[0]],[-forward[1],forward[0],0]]);J=-cross@jr[:,dofs];step=np.linalg.solve(J.T@J+.005*np.eye(6),J.T@(desired-forward)-.005*(p.qpos[ids]-anchor));step*=min(1,.15/max(np.linalg.norm(step),1e-12));p.qpos[ids]=np.clip(p.qpos[ids]+step,bounds[:,0]+1e-5,bounds[:,1]-1e-5)
        if np.linalg.norm(desired-forward)<.02:break
    return p.qpos.copy()
