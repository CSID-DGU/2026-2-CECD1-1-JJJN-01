"""Constrained position IK for a planted-foot free-throw reference."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json
import hashlib
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter
import mujoco
from spl_motion import ROOT, read_trial, smooth
from humanoid_model import load_model, reset_pose, MODEL_PATH

BODY_MAP = {'left_hip':'LEFT_HIP','right_hip':'RIGHT_HIP','left_knee':'LEFT_KNEE',
            'right_knee':'RIGHT_KNEE','left_shoulder':'LEFT_SHOULDER','right_shoulder':'RIGHT_SHOULDER',
            'left_elbow':'LEFT_ELBOW','right_elbow':'RIGHT_ELBOW','left_wrist':'LEFT_WRIST','right_wrist':'RIGHT_WRIST'}


def normalize(tracks, fps, start, model, data):
    p = {key:smooth(value,fps) for key,value in tracks.items() if np.isfinite(value).all()}
    left = np.mean(p['LEFT_SHOULDER']-p['RIGHT_SHOULDER'],axis=0)
    left[2] = 0
    left /= np.linalg.norm(left)
    forward = np.cross(left,[0,0,1])
    transform = np.stack([forward,left,[0,0,1]])
    origin = (p['LEFT_ANKLE'][start]+p['RIGHT_ANKLE'][start])/2
    ground = np.median(np.min(np.stack([p[k][max(0,start-5):start+6,2] for k in ['LEFT_HEEL','RIGHT_HEEL','LEFT_BIG_TOE','RIGHT_BIG_TOE']]),axis=0))
    origin[2] = ground
    source_leg = np.mean([np.median(np.linalg.norm(p[s+'_HIP']-p[s+'_KNEE'],axis=1)+np.linalg.norm(p[s+'_KNEE']-p[s+'_ANKLE'],axis=1)) for s in ['LEFT','RIGHT']])
    target_leg = np.mean([np.linalg.norm(data.body(s+'_hip').xpos-data.body(s+'_knee').xpos)+np.linalg.norm(data.body(s+'_knee').xpos-data.body(s+'_ankle').xpos) for s in ['left','right']])
    scale = float(target_leg/source_leg)
    p = {key:((value-origin)@transform.T)*scale for key,value in p.items()}
    return p, {'rotation_rows':transform.tolist(),'origin_m':origin.tolist(),'scale':scale}


def retarget(selection_path, output, speed=1.):
    selection = json.loads(Path(selection_path).read_text(encoding='utf-8'))
    path = ROOT/selection['path']
    if selection.get('source_sha256') and hashlib.sha256(path.read_bytes()).hexdigest()!=selection['source_sha256']:
        raise ValueError('Source checksum does not match selection')
    raw,fps,ids,times,tracks,_ = read_trial(path)
    indices = []
    for name in ['start_frame','release_frame','end_frame']:
        match = np.flatnonzero(ids==selection[name])
        if len(match)!=1: raise ValueError(f'Unknown frame: {name}')
        indices.append(int(match[0]))
    start,release,end = indices
    if not 0 <= start < release < end < len(ids):
        raise ValueError('Require start < release < end')
    if not np.isfinite(speed) or speed<=0: raise ValueError('Speed must be positive')
    model = load_model()
    data = mujoco.MjData(model)
    reset_pose(model,data,'t_pose')
    p,normalization = normalize(tracks,fps,start,model,data)
    # Body IDs are in tree traversal order. Select by names, not SMPL numeric IDs.
    active = np.array(list(range(3))+[int(model.jnt_dofadr[j]) for j in range(1,model.njnt)
                     if not any(s in model.body(model.jnt_bodyid[j]).name for s in ['index','middle','ring','pinky','thumb'])])
    hinge_q = model.jnt_qposadr[1:]
    limits = model.jnt_range[1:]
    anchors = {}
    for side in ['left','right']:
        anchor = data.body(side+'_ankle').xpos.copy()
        anchor[:2] = p[side.upper()+'_ANKLE'][start,:2]
        anchor[2] -= .003
        anchors[side] = anchor
    support_center = np.mean(list(anchors.values()),axis=0)
    support_center[0] += .035
    pelvis_offset = data.body('pelvis').xpos-(data.body('left_hip').xpos+data.body('right_hip').xpos)/2
    # Fit directions to the target's segment lengths instead of demanding the
    # source athlete's longer arms from a shorter avatar.
    for side in ['left','right']:
        prefix=side.upper()
        shoulder=p[prefix+'_SHOULDER']
        elbow=p[prefix+'_ELBOW'].copy()
        wrist=p[prefix+'_WRIST'].copy()
        upper=np.linalg.norm(data.body(side+'_elbow').xpos-data.body(side+'_shoulder').xpos)
        fore=np.linalg.norm(data.body(side+'_wrist').xpos-data.body(side+'_elbow').xpos)
        v=elbow-shoulder
        new_elbow=shoulder+v/np.linalg.norm(v,axis=1,keepdims=True)*upper
        v=wrist-elbow
        new_wrist=new_elbow+v/np.linalg.norm(v,axis=1,keepdims=True)*fore
        for key in p:
            if key.startswith(prefix) and any(x in key for x in ['FINGER','THUMB','PINKY']):
                p[key] += new_wrist-wrist
        p[prefix+'_ELBOW']=new_elbow
        p[prefix+'_WRIST']=new_wrist
    q = model.key('standing').qpos.copy()
    pelvis = (p['LEFT_HIP']+p['RIGHT_HIP'])/2+pelvis_offset
    # Adjust vertical reference so ankle landmarks match the collision-foot height.
    z_offset = np.mean([anchors[s][2]-p[s.upper()+'_ANKLE'][start,2] for s in anchors])
    for key in p: p[key][:,2] += z_offset
    pelvis[:,2] += z_offset
    normalization['vertical_alignment_m']=float(z_offset)
    q[:3] = pelvis[start]
    jacp, jacr, jaccom = (np.zeros((3,model.nv)) for _ in range(3))
    qposes, residuals = [], []
    for frame in range(start,end+1):
        previous = q.copy()
        q_indices=np.r_[np.arange(3),active[3:]+1]
        lower=np.r_[[-.5,-.5,.5],model.jnt_range[[int(model.dof_jntid[v]) for v in active[3:]],0]+.002]
        upper=np.r_[[.5,.5,1.5],model.jnt_range[[int(model.dof_jntid[v]) for v in active[3:]],1]-.002]
        q[3:7]=[1,0,0,0]
        def evaluate(x, want_jac=False):
            data.qpos[:]=q
            data.qpos[q_indices]=x
            mujoco.mj_forward(model,data)
            matrices,errors=[],[]
            def add_body(name,target,weight):
                bid=model.body(name).id
                mujoco.mj_jacBody(model,data,jacp,jacr,bid)
                matrices.append(jacp[:,active].copy()*weight)
                errors.append((data.xpos[bid]-target)*weight)
            for body,key in BODY_MAP.items():
                weight=1.6 if 'wrist' in body else .8 if 'hip' in body or 'knee' in body else 1.
                add_body(body,p[key][frame],weight)
            add_body('pelvis',pelvis[frame],.6)
            for side,anchor in anchors.items():
                add_body(side+'_ankle',anchor,7.)
                rot=data.body(side+'_ankle').xmat.reshape(3,3)
                mujoco.mj_jacBody(model,data,jacp,jacr,model.body(side+'_ankle').id)
                matrices.append(jacr[:,active].copy()*.7)
                errors.append(Rotation.from_matrix(rot).as_rotvec()*.7)
                keys=[side.upper()+'_SECOND_FINGER_MCP',side.upper()+'_FIFTH_FINGER_MCP']
                if all(k in p for k in keys):
                    target=(p[keys[0]][frame]+p[keys[1]][frame]+2*p[side.upper()+'_WRIST'][frame])/4
                    site=model.site(side+'_palm').id
                    mujoco.mj_jacSite(model,data,jacp,jacr,site)
                    matrices.append(jacp[:,active].copy()*.6)
                    errors.append((data.site_xpos[site]-target)*.6)
            mujoco.mj_jacSubtreeCom(model,data,jaccom,1)
            matrices.append(jaccom[:2,active].copy()*2.5)
            errors.append((data.subtree_com[1,:2]-support_center[:2])*2.5)
            matrices.append(np.eye(len(active))*.015)
            errors.append((x-previous[q_indices])*.015)
            return np.concatenate(matrices) if want_jac else np.concatenate(errors)
        result=least_squares(evaluate,np.clip(q[q_indices],lower+1e-5,upper-1e-5),
                             jac=lambda x:evaluate(x,True),bounds=(lower,upper),
                             max_nfev=65 if frame==start else 30,ftol=1e-5,xtol=1e-5,gtol=1e-5)
        q[q_indices]=result.x
        for side in ['left','right']:q[model.joint(side+'_foot_y').qposadr[0]]=0.
        # Missing/noisy individual finger rotations use a declared hand-shape prior.
        fraction=np.clip((frame-release+3)/12,0,1)
        for j in range(1,model.njnt):
            name=model.joint(j).name
            if any(f in name for f in ['index','middle','pinky','ring','thumb']):
                value=(.25*(1-fraction)+.08*fraction)*(1 if name.startswith('left') else -1) if name.endswith('_x') else 0.
                q[model.jnt_qposadr[j]]=value
        data.qpos[:]=q;mujoco.mj_forward(model,data)
        residuals.append({name:float(np.linalg.norm(data.body(name).xpos-p[key][frame])) for name,key in BODY_MAP.items()})
        qposes.append(q.copy())
    qposes=np.array(qposes)
    # Smooth hinge and translation trajectories; quaternions remain normalized.
    qposes[:,:3]=savgol_filter(qposes[:,:3],7,2,axis=0)
    qposes[:,7:]=savgol_filter(qposes[:,7:],7,2,axis=0)
    qposes[:,hinge_q]=np.clip(qposes[:,hinge_q],limits[:,0]+.002,limits[:,1]-.002)
    dt=1/fps/speed
    velocities=np.zeros((len(qposes),model.nv))
    for i in range(len(qposes)-1):mujoco.mj_differentiatePos(model,velocities[i],dt,qposes[i],qposes[i+1])
    velocities[-1]=velocities[-2]
    velocities=savgol_filter(velocities,7,2,axis=0)
    acceleration=np.gradient(velocities,dt,axis=0)
    residuals=[]
    for i,qframe in enumerate(qposes):
        data.qpos[:]=qframe;mujoco.mj_forward(model,data)
        residuals.append({name:float(np.linalg.norm(data.body(name).xpos-p[key][start+i])) for name,key in BODY_MAP.items()})
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,qpos=qposes,qvel=velocities,qacc=acceleration,
                        times=np.arange(len(qposes))*dt,source_frames=ids[start:end+1],
                        source_points=np.stack([p[k][start:end+1] for k in BODY_MAP.values()],axis=1),
                        point_names=np.array(list(BODY_MAP)),
                        source_ball=p['BALL'][start:end+1],anchors=np.stack(list(anchors.values())))
    info={'selection':selection,'normalization':normalization,'playback_speed':speed,
          'duration_s':(len(qposes)-1)*dt,'frames':len(qposes),
          'mean_landmark_error_m':float(np.mean([list(r.values()) for r in residuals])),
          'mean_error_by_body_m':{key:float(np.mean([r[key] for r in residuals])) for key in BODY_MAP},
          'notes':['Arm segment lengths are adapted to the avatar, preserving source bone directions.',
                   'Feet constrained to planted anchors; source heel rise is intentionally not reproduced.',
                   'COM projected toward support center to make a first tracking reference feasible.',
                   'Finger poses use a synthetic smooth prior, not the noisy source finger measurements.',
                   'Reference rendering is kinematic; physical tracking is evaluated separately.'],
          'model_sha256':hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()}
    out.with_suffix('.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps({k:info[k] for k in ['frames','duration_s','mean_landmark_error_m','mean_error_by_body_m']},indent=2),flush=True)
    return info


if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--selection',type=Path,default=ROOT/'runs/spl_tracking/selection.json')
    a.add_argument('--output',type=Path,default=ROOT/'data/motions/processed/free_throw_reference.npz')
    a.add_argument('--speed',type=float,default=1.)
    args=a.parse_args();retarget(args.selection,args.output,args.speed)
