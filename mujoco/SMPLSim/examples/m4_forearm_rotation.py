"""3C: forearm observability and source-target-only twist experiments.
Original motion, controller, contact models and physical ball states are untouched.
"""
import argparse, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco,numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from smpl_sim.smpllib.smpl_joint_names import SMPLH_BONE_ORDER_NAMES
from examples import m4_dynamic_grasp as dyn
from examples.m3_control import write_csv
OUT=ROOT/'outputs/m4_2d_3c_forearm';PREVIOUS=dyn.OUT
NAMES=['R_Shoulder','R_Elbow','R_Wrist']

def transported_roll(rotations,axes):
    """Material roll relative to a minimal-rotation parallel-transport frame.
    This convention separates bone swing from axial roll; roll is path dependent.
    """
    a=axes[0];basis=np.eye(3)[np.argmin(abs(a))];u=basis-a*np.dot(a,basis);u/=np.linalg.norm(u)
    local=rotations[0].T@u;roll=[];frames=[]
    for i,(R,a) in enumerate(zip(rotations,axes)):
        if i:
            old=axes[i-1];cross=np.cross(old,a);s=np.linalg.norm(cross);c=np.clip(np.dot(old,a),-1,1)
            if s>1e-10:u=Rotation.from_rotvec(cross/s*np.arctan2(s,c)).apply(u)
            elif c<0:raise ValueError('Antiparallel adjacent axes: transport undefined')
        u-=a*np.dot(a,u);u/=np.linalg.norm(u);v=np.cross(a,u);material=R@local
        roll.append(np.arctan2(np.dot(material,v),np.dot(material,u)));frames.append(np.column_stack((u,v,a)))
    return np.unwrap(roll),np.array(frames)

def euler_near(R,ref):
    principal=Rotation.from_matrix(R).as_euler('XYZ');other=np.array([principal[0]+np.pi,np.pi-principal[1],principal[2]+np.pi])
    options=[a+2*np.pi*np.round((ref-a)/(2*np.pi)) for a in (principal,other)]
    return min(options,key=lambda a:np.linalg.norm(a-ref))

def apply_twist(m,d,q,angle,reference=None):
    """Exact axial counter-rotation; elbow and wrist centers stay fixed in FK."""
    d.qpos[:]=q;mujoco.mj_forward(m,d);e=m.body('R_Elbow').id;w=m.body('R_Wrist').id
    R=d.xmat[e].reshape(3,3).copy();a=d.xpos[w]-d.xpos[e];a/=np.linalg.norm(a)
    parent=d.xmat[m.body_parentid[e]].reshape(3,3);ids=np.array([int(m.joint('R_Elbow_'+k).qposadr[0]) for k in 'xyz'])
    desired=parent.T@Rotation.from_rotvec(a*angle).as_matrix()@R
    # These three hinges compose intrinsic XYZ, as verified by matrix round-trip.
    result=q.copy();result[ids]=euler_near(desired,q[ids] if reference is None else reference)
    d.qpos[:]=result;mujoco.mj_forward(m,d)
    np.testing.assert_allclose(d.xmat[e].reshape(3,3),parent@desired,atol=1e-10)
    return result

def analyze(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    m=mujoco.MjModel.from_xml_path(str(args.models/'body_01.xml'));b=dyn.prior.HumanoidControllerBridge(args.models/'body_01.xml',m,args.source);d=mujoco.MjData(m)
    ts=np.arange(len(b.motion.poses))/b.motion.fps;qs=np.array([b.target(t) for t in ts]);pos=[];rot=[]
    for q in qs:
        d.qpos[:]=q;mujoco.mj_forward(m,d);pos.append([d.xpos[m.body(n).id].copy() for n in NAMES]);rot.append([d.xmat[m.body(n).id].reshape(3,3).copy() for n in NAMES])
    pos=np.array(pos);rot=np.array(rot);axes=pos[:,2]-pos[:,1];axes/=np.linalg.norm(axes,axis=1)[:,None]
    roll,transport=transported_roll(rot[:,1],axes);i0=90;mask=(ts>=1.5)&(ts<=2)
    full=np.degrees(Rotation.from_matrix(rot[i0,1].T@rot[:,1]).magnitude());swing=np.degrees(np.arccos(np.clip(axes@axes[i0],-1,1)))
    rows=[]
    for i,t in enumerate(ts):
        row=dict(source_time_s=t,forearm_full_from_1p5_deg=full[i],bone_direction_from_1p5_deg=swing[i],transport_roll_deg=np.degrees(roll[i]),roll_from_1p5_deg=np.degrees(roll[i]-roll[i0]))
        for k,n in enumerate(NAMES):
            quat=Rotation.from_matrix(rot[i,k]).as_quat();omega=0 if i==0 else np.degrees(Rotation.from_matrix(rot[i-1,k].T@rot[i,k]).magnitude())
            row[n+'_matrix_step_deg']=omega
            for a,x in enumerate('xyz'):
                row[n+'_'+x+'_m']=pos[i,k,a]
                row[n+'_local_hinge_'+x+'_rad']=qs[i,int(m.joint(n+'_'+x).qposadr[0])]
                row[n+'_smpl_axis_angle_'+x+'_rad']=b.motion.poses.reshape(-1,52,3)[i,SMPLH_BONE_ORDER_NAMES.index(n),a]
            for a,x in enumerate('xyzw'):row[n+'_quaternion_'+x]=quat[a]
        rows.append(row)
    write_csv(out/'source_rotations.csv',rows)
    armids=np.array([int(m.joint(n).qposadr[0]) for n in dyn.lift.ARM]);dofs=np.array([int(m.joint(n).dofadr[0]) for n in dyn.lift.ARM]);perturb=[];jac=[]
    for t in [1.,1.5,1.85,2.,2.5]:
        i=round(t*60);q=qs[i];d.qpos[:]=q;mujoco.mj_forward(m,d);p=pos[i].copy();R=rot[i,1];mat=[]
        for n in NAMES:
            jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv));mujoco.mj_jacBody(m,d,jp,jr,m.body(n).id);mat.append(jp[:,dofs])
        J=np.vstack(mat);U,s,V=np.linalg.svd(J,full_matrices=True);null=V[-1];rotationJ=jr[:,dofs] # last body wrist orientation
        Je=J[:,3:6];_,se,Ve=np.linalg.svd(Je);ne=Ve[-1];
        jac.append(dict(elbow_position_rank=int(np.linalg.matrix_rank(Je,tol=1e-8)),elbow_singular_values=se.tolist(),elbow_null_position_velocity_norm=float(np.linalg.norm(Je@ne)),elbow_null_wrist_angular_velocity_norm=float(np.linalg.norm(rotationJ[:,3:6]@ne)),source_time_s=t,rank=int(np.linalg.matrix_rank(J,tol=1e-8)),singular_values=s.tolist(),null_position_velocity_norm=float(np.linalg.norm(J@null)),null_wrist_angular_velocity_norm=float(np.linalg.norm(rotationJ@null))))
        for delta in [-.5,-.2,.2,.5]:
            corrected=apply_twist(m,d,q,delta);pnew=np.array([d.xpos[m.body(n).id].copy() for n in NAMES]);angle=Rotation.from_matrix(R.T@d.xmat[m.body('R_Elbow').id].reshape(3,3)).magnitude()
            perturb.append(dict(source_time_s=t,twist_delta_rad=delta,max_joint_position_change_mm=np.max(np.linalg.norm(pnew-p,axis=1))*1000,forearm_rotation_change_deg=np.degrees(angle),elbow_euler_delta_norm_rad=np.linalg.norm(corrected[armids[3:6]]-q[armids[3:6]])))
    write_csv(out/'position_preserving_twist.csv',perturb);(out/'position_jacobian.json').write_text(json.dumps(jac,indent=2))
    # B removes only short-timescale roll variation. C limits accumulated roll
    # relative to 1.0s to +/-20deg; both use exactly position-preserving FK.
    window=dyn.prior.smooth((ts-1)/.25)*(1-dyn.prior.smooth((ts-2.5)/.25))
    B=(gaussian_filter1d(roll,9,mode='nearest')-roll)*window
    C=np.clip(roll[60]+np.clip(roll-roll[60],-np.radians(20),np.radians(20))-roll,-np.radians(30),np.radians(30))*window
    corrections={'A':np.zeros(len(ts)),'B':B,'C':C};correction_rows=[]
    for condition,delta in corrections.items():
        result=[];reference=None;errors=[]
        for i,(q,a) in enumerate(zip(qs,delta)):
            x=apply_twist(m,d,q,a,reference);reference=x[armids[3:6]];result.append(x)
            pnew=np.array([d.xpos[m.body(n).id].copy() for n in NAMES]);errors.append(np.linalg.norm(pnew-pos[i],axis=1))
        result=np.array(result)
        if condition!='A':
            bounds=np.array([m.jnt_range[m.joint(n).id] for n in dyn.lift.ARM[3:6]])
            if np.any(result[:,armids[3:6]]<bounds[:,0]) or np.any(result[:,armids[3:6]]>bounds[:,1]):raise RuntimeError('Corrected elbow target exceeds XML ranges')
        np.savez_compressed(out/(condition+'_targets.npz'),times=ts,qpos=result,twist_correction_rad=delta)
        correction_rows.append(dict(condition=condition,max_correction_deg=float(np.max(abs(np.degrees(delta)))),max_FK_position_error_mm=float(np.max(errors)*1000),roll_range_1p5_2p0_deg=float(np.degrees(np.ptp((roll+delta)[mask]))),max_elbow_euler_frame_step_rad=float(np.max(abs(np.diff(result[:,armids[3:6]],axis=0))))))
    write_csv(out/'adapter_fk_comparison.csv',correction_rows)
    np.savez_compressed(out/'source_analysis.npz',times=ts,qpos=qs,positions=pos,rotations=rot,axes=axes,roll=roll,transport=transport)
    summary=dict(full_rotation_max_1p5_2p0_deg=float(max(full[mask])),bone_direction_max_1p5_2p0_deg=float(max(swing[mask])),transport_roll_range_1p5_2p0_deg=float(np.degrees(np.ptp(roll[mask]))),max_matrix_frame_step_1p0_2p5_deg=float(max(r['R_Elbow_matrix_step_deg'] for r in rows if 1<=r['source_time_s']<=2.5)),jacobian_position_ranks=[r['rank'] for r in jac],max_twist_perturbation_position_error_mm=max(r['max_joint_position_change_mm'] for r in perturb),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest())
    # Original observations constrain the limb vector and a hand landmark frame,
    # but do not directly provide a forearm material orientation / release label.
    from smpl_sim.motion.spl_data import load_trial
    trial=load_trial(args.dataset);raw=json.loads(args.dataset.read_text());available=trial.names
    handkeys=['RIGHT_WRIST','RIGHT_SECOND_FINGER_MCP','RIGHT_FIFTH_FINGER_MCP'];p=trial.normalized_world_m[:,[available.index(k) for k in handkeys]]
    v1=p[:,1]-p[:,0];v2=p[:,2]-p[:,0];normal=np.cross(v1,v2);length=np.linalg.norm(normal,axis=1);valid=np.isfinite(normal).all(axis=1)&(length>1e-5);normal[valid]/=length[valid,None]
    elbow=trial.normalized_world_m[:,available.index('RIGHT_ELBOW')];wrist=trial.normalized_world_m[:,available.index('RIGHT_WRIST')];bone=wrist-elbow;bone/=np.linalg.norm(bone,axis=1)[:,None]
    source_rows=[]
    for i,t in enumerate(ts):
        source_rows.append(dict(source_time_s=t,hand_plane_valid=int(valid[i]),raw_bone_direction_from_1p5_deg=float(np.degrees(np.arccos(np.clip(bone[i]@bone[i0],-1,1)))),raw_hand_plane_from_1p5_deg=float(np.degrees(np.arccos(np.clip(normal[i]@normal[i0],-1,1)))) if valid[i] and valid[i0] else np.nan))
    write_csv(out/'original_spl_observations.csv',source_rows)
    summary['raw_bone_direction_max_1p5_2p0_deg']=max(r['raw_bone_direction_from_1p5_deg'] for r in source_rows if 1.5<=r['source_time_s']<=2)
    summary['raw_hand_plane_max_1p5_2p0_deg']=max(r['raw_hand_plane_from_1p5_deg'] for r in source_rows if 1.5<=r['source_time_s']<=2)
    summary['raw_hand_plane_valid_frames']=int(sum(valid));summary['source_video_available']=False
    (out/'analysis_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

class CorrectedBridge:
    def __init__(self,base,path):
        self.base=base;data=np.load(path);self.times=data['times'];self.correction=data['twist_correction_rad'];self.targets=data['qpos'];self.elbow=np.array([int(base.model.joint('R_Elbow_'+k).qposadr[0]) for k in 'xyz']);self.probe=mujoco.MjData(base.model)
    def __getattr__(self,key):return getattr(self.base,key)
    def target(self,t):
        q=self.base.target(t);angle=float(np.interp(t,self.times,self.correction))
        if abs(angle)<1e-14:return q
        # Compute the exact correction at the requested interpolated source pose.
        ref=np.array([np.interp(t,self.times,self.targets[:,j]) for j in self.elbow]);return apply_twist(self.base.model,self.probe,q,angle,ref)

OriginalTargets=dyn.DynamicTargets
class CorrectedTargets(OriginalTargets):
    mode='A';output=OUT
    def __init__(self,b,c,condition,speed):
        proxy=b if self.mode=='A' else CorrectedBridge(b,self.output/(self.mode+'_targets.npz'))
        super().__init__(proxy,c,'C',speed)

def run(args):
    c=json.loads((PREVIOUS/'selected.json').read_text());gates=json.loads((PREVIOUS/'candidate_gates.json').read_text())
    assert any(g['candidate']==c['name'] and g['static_gate'] and g['lifting_gate'] for g in gates)
    for mode in ['A','B','C'] if args.condition=='all' else [args.condition]:
        folder=args.output/mode;folder.mkdir(exist_ok=True);link=folder/'candidates'
        if not link.exists():link.symlink_to(PREVIOUS/'candidates',target_is_directory=True)
        options=argparse.Namespace(**vars(args));options.output=folder
        CorrectedTargets.mode=mode;CorrectedTargets.output=args.output;dyn.DynamicTargets=CorrectedTargets
        try:s=dyn.run(options,'C',1.,c)
        finally:dyn.DynamicTargets=OriginalTargets
        s['rotation_condition']=mode;(folder/'C_1/summary.json').write_text(json.dumps(s,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT);p.add_argument('--models',type=Path,default=ROOT.parents[2]/'basketball-r1/SMPLSim/outputs/smplh_physics');p.add_argument('--source',type=Path,default=ROOT/'outputs/m3_spl_fitting/spl_P0002_T0049.npz');p.add_argument('--dataset',type=Path,default=ROOT.parents[2]/'SPL-Open-Data/basketball/freethrow/data/2025-12-18/P0002/BB_FT_P0002_T0049.json');p.add_argument('--analyze',action='store_true');p.add_argument('--condition',choices=['all','A','B','C'],default='all');p.add_argument('--gui',action='store_true');args=p.parse_args()
    protected=json.loads((args.output/'preservation_before.json').read_text()) if (args.output/'preservation_before.json').exists() else json.loads((OUT/'preservation_before.json').read_text())
    for path in [args.source,args.models/'body_01.xml']:
        resolved=str(path.resolve())
        if resolved not in protected or hashlib.sha256(path.read_bytes()).hexdigest()!=protected[resolved]:raise RuntimeError('Protected input changed: '+resolved)
    if args.analyze:analyze(args)
    else:run(args)
if __name__=='__main__':main()
