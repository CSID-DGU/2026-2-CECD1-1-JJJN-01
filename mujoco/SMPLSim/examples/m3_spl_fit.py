"""Fit one real SPL trial to source SMPL+H; no target-shape retargeting."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation
from smpl_sim.motion.spl_data import load_trial,mapping_table
from smpl_sim.motion.spl_fitting import fit,observation_positions
from smpl_sim.motion.smplh_motion import SMPLHMotion,SMPL_TO_WORLD
from smpl_sim.smpllib.smpl_parser import SMPLH_Parser

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def visualize(out,trial,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys=result['keys'];names=result['names'];idx=[trial.names.index(k) for k in keys]
    target=trial.normalized_world_m[:,idx];pred=SMPL_TO_WORLD.apply(result['joints'].reshape(-1,3)).reshape(-1,52,3)
    initial=SMPL_TO_WORLD.apply(result['initial_joints'].reshape(-1,3)).reshape(-1,52,3)
    for skeleton in (pred,initial):
        skeleton[:,0]=(skeleton[:,1]+skeleton[:,2])/2  # Compare the same MID_HIP observation.
    from smpl_sim.smpllib.smpl_joint_names import SMPLH_BONE_ORDER_NAMES as order
    edges=[('Pelvis','Neck')]
    for side in ('L','R'):
        edges += [('Pelvis',side+'_Hip'),(side+'_Hip',side+'_Knee'),(side+'_Knee',side+'_Ankle'),('Neck',side+'_Shoulder'),(side+'_Shoulder',side+'_Elbow'),(side+'_Elbow',side+'_Wrist')]
    frames=[0,len(target)//2,int(np.nanargmax(target[:,names.index('R_Wrist'),2]))]
    fig=plt.figure(figsize=(13,8))
    for row,estimate in enumerate([initial,pred]):
        for col,t in enumerate(frames):
            ax=fig.add_subplot(2,3,row*3+col+1,projection='3d')
            ax.scatter(*target[t].T,color='black',s=13,label='SPL')
            for a,b in edges:
                xy=target[t,[names.index(a),names.index(b)]];ax.plot(*xy.T,color='black',alpha=.45)
                xy=estimate[t,[order.index(a),order.index(b)]];ax.plot(*xy.T,color='tab:orange' if row==0 else 'tab:blue')
            ax.set(xlim=(-.8,.8),ylim=(-.8,.8),zlim=(0,2.3),xlabel='world X (m)',ylabel='world Y (m)',zlabel='world Z (m)',title=f"{'Before' if row==0 else 'After'}: {t/trial.fps:.2f}s")
            ax.view_init(elev=15,azim=-55);ax.set_box_aspect((1.6,1.6,2.3))
    fig.tight_layout();fig.savefig(out/'skeleton_before_after.png',dpi=150);plt.close(fig)
    observed=observation_positions(torch.tensor(result['joints']),names).numpy()
    errors=np.linalg.norm(observed-result['observations'],axis=-1)*1000
    fig,ax=plt.subplots(figsize=(10,4))
    for name in ['L_Shoulder','L_Elbow','L_Wrist','R_Shoulder','R_Elbow','R_Wrist']:
        ax.plot(np.arange(len(target))/trial.fps,errors[:,names.index(name)],label=name)
    ax.set(xlabel='time (s)',ylabel='position error (mm)',title='Arm fitting residuals (not physics tracking)');ax.legend(ncol=3);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'arm_position_errors.png',dpi=150);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--xml',type=Path,required=True,help='Existing XML for loader compatibility only')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/m3_spl_fitting')
    p.add_argument('--iterations',type=int,default=800);p.add_argument('--max-gap',type=int,default=3)
    p.add_argument('--include-fingers',action='store_true',help='Experimental joint-center hand fit; excluded in body MVP')
    p.add_argument('--fixed-betas',action='store_true')
    args=p.parse_args()
    if args.iterations<1 or args.max_gap<0:p.error('Invalid iteration/gap count')
    torch.set_num_threads(1);torch.manual_seed(0);args.output.mkdir(parents=True,exist_ok=True);out=args.output
    source_hash=hashlib.sha256(args.source.read_bytes()).hexdigest();trial=load_trial(args.source,args.max_gap)
    if (trial.metadata['trial_date'],trial.metadata['participant_id'],trial.metadata['trial_id'])!=('2025-12-18','P0002','T0049'):
        p.error('This MVP is limited to 2025-12-18 / P0002 / T0049')
    parser=SMPLH_Parser(str(args.model_dir),gender='male',num_betas=16,use_pca=False,flat_hand_mean=True,create_transl=False)
    result=fit(trial,parser,args.iterations,args.include_fingers,not args.fixed_betas)
    motion=result['motion'];path=out/'spl_P0002_T0049.npz';motion.save(path)
    # Extra descriptive fields are ignored by the existing schema-v1 loader.
    with np.load(path,allow_pickle=False) as f:fields={k:f[k] for k in f.files}
    np.savez_compressed(path,**fields,root_translation=result['translation'],root_orientation=motion.poses[:,:3],
                        body_pose=motion.poses[:,3:66],betas_source=result['betas'])
    loaded=SMPLHMotion.load(path);model=mujoco.MjModel.from_xml_path(str(args.xml));bound=loaded.bind(model)
    assert np.isfinite(bound.qpos).all();assert abs(np.linalg.det(trial.world_rotation)-1)<1e-12
    floor=model.geom('floor').id;floor_matrix=np.zeros(9)
    mujoco.mju_quat2Mat(floor_matrix,model.geom_quat[floor]);floor_normal=floor_matrix.reshape(3,3)[:,2]
    if not np.allclose(floor_normal,[0,0,1]) or not (model.opt.gravity[2]<0 and np.allclose(model.opt.gravity[:2],0)):
        raise ValueError('Current normalization requires the verified +Z floor / -Z gravity XML')
    selected=np.arange(0,len(motion.poses),max(1,len(motion.poses)//5))
    with torch.no_grad():
        beta=torch.tensor(result['betas']).expand(len(selected),-1)
        verts,full=parser.get_joints_verts(torch.tensor(motion.poses[selected]),beta,torch.tensor(result['translation'][selected]))
    parity=float(np.max(abs(full[:,:52].numpy()-result['joints'][selected])))
    if parity>2e-6:raise RuntimeError(f'SMPL forward/FK mismatch: {parity}')
    from smpl_sim.smpllib.smpl_joint_names import SMPLH_BONE_ORDER_NAMES as names
    indices=[names.index(n) for n in result['names']]
    pred=observation_positions(torch.tensor(result['joints']),result['names']).numpy();observations=result['observations']
    errors=np.linalg.norm(pred-observations,axis=-1)*1000
    initial_observed=observation_positions(torch.tensor(result['initial_joints']),result['names']).numpy()
    initial_errors=np.linalg.norm(initial_observed-observations,axis=-1)*1000
    valid=(result['confidence']>0)&np.isfinite(errors)
    joint_rows=[]
    for j,key in enumerate(result['keys']):
        ok=valid[:,j];raw=trial.original_m[:,trial.names.index(key)];raw_smpl=SMPL_TO_WORLD.inv().apply(((raw-trial.origin_m)@trial.world_rotation.T))
        raw_err=np.linalg.norm(pred[:,j]-raw_smpl,axis=-1)*1000
        joint_rows.append({'spl':key,'smplh':result['names'][j],'samples':int(ok.sum()),'mean_error_mm':float(errors[ok,j].mean()),
                           'p95_error_mm':float(np.percentile(errors[ok,j],95)),'max_error_mm':float(errors[ok,j].max()),
                           'initial_mean_error_mm':float(initial_errors[ok,j].mean()),'raw_finite_mean_error_mm':float(np.nanmean(raw_err))})
    rotations=Rotation.from_rotvec(motion.poses.reshape(-1,3)).as_matrix().reshape(-1,52,3,3)
    relative=np.einsum('tjki,tjkl->tjil',rotations[:-1],rotations[1:])
    angles=Rotation.from_matrix(relative.reshape(-1,3,3)).magnitude().reshape(-1,52)
    root_steps=np.linalg.norm(np.diff(motion.root_position,axis=0),axis=-1)
    frame_rows=[{'frame':i,'time_s':i/trial.fps,'mpjpe_mm':float(np.mean(errors[i,valid[i]])),
                 'root_step_m':0. if i==0 else float(root_steps[i-1]),'max_joint_rotation_step_rad':0. if i==0 else float(angles[i-1].max())} for i in range(len(errors))]
    ratios=[{'joint':key,'original_valid_fraction':float(trial.valid_original[:,j].mean()),'outlier_frames':int(trial.outliers[:,j].sum()),
             'interpolated_frames':int(trial.filled[:,j].sum()),'usable_fraction':float((trial.weights[:,j]>0).mean())} for j,key in enumerate(trial.names)]
    restored=trial.restore_world(trial.normalized_world_m)
    unchanged=trial.valid_original&~trial.outliers&~trial.filled
    inverse_error=float(np.max(abs(restored[unchanged]-trial.original_m[unchanged])))
    smpl_inverse=SMPL_TO_WORLD.apply(trial.smpl_m.reshape(-1,3)).reshape(trial.smpl_m.shape)
    smpl_error=float(np.nanmax(abs(smpl_inverse-trial.normalized_world_m)))
    left_arm=[result['names'].index(n) for n in ('L_Shoulder','L_Elbow','L_Wrist')];right_arm=[result['names'].index(n) for n in ('R_Shoulder','R_Elbow','R_Wrist')]
    summary={'frames':len(motion.poses),'fps':trial.fps,'duration_s':motion.duration,'mpjpe_mm':float(errors[valid].mean()),
             'initial_mpjpe_mm':float(initial_errors[valid].mean()),'left_arm_mpjpe_mm':float(np.mean(errors[:,left_arm][valid[:,left_arm]])),
             'right_arm_mpjpe_mm':float(np.mean(errors[:,right_arm][valid[:,right_arm]])),
             'max_root_step_mm':float(root_steps.max()*1000),'max_root_speed_m_s':float(root_steps.max()*trial.fps),
             'max_joint_step_rad':float(angles.max()),'max_joint_speed_rad_s':float(angles.max()*trial.fps),
             'root_acceleration_rms_m_s2':float(np.sqrt(np.mean((np.diff(motion.root_position,n=2,axis=0)*trial.fps**2)**2))),
             'rotation_acceleration_matrix_rms_s2':float(np.sqrt(np.mean((np.diff(rotations,n=2,axis=0)*trial.fps**2)**2))),
             'source_shape_fk_parity_m':parity,'coordinate_inverse_error_m':inverse_error,'smpl_world_roundtrip_error_m':smpl_error,
             'root_hip_midpoint_error_mm':joint_rows[0]['mean_error_mm'],
             'finite_motion':bool(np.isfinite(motion.poses).all() and np.isfinite(motion.root_position).all()),
             'npz_xml_loader_passed':True,'xml_range_violations':0,'include_fingers':args.include_fingers,
             'physical_rollout_performed':False}
    if not summary['finite_motion'] or not summary['mpjpe_mm']<summary['initial_mpjpe_mm']:raise RuntimeError('Fitting did not improve finite baseline')
    metadata={'source_path':str(args.source.resolve()),'source_sha256':source_hash,'trial':trial.metadata,
              'units':'meter','input_coordinate':'SPL Z-up','fitting_coordinate':'SMPL Y-up',
              'xml_world_coordinate':'Z-up verified from actual floor and gravity',
              'xml_gravity':model.opt.gravity.tolist(),'xml_floor_normal':floor_normal.tolist(),
              'origin_m':trial.origin_m.tolist(),'world_rotation':trial.world_rotation.tolist(),
              'betas_source':result['betas'].tolist(),'gender_model':'male; participant gender unknown',
              'hand_mean':'flat_hand_mean=True; no hand fit in default body run; XML hand rest not calibrated',
              'root_translation_semantics':'library transl; NPZ root_position is actual pelvis position',
              'optimizer':'CPU Adam with cosine LR decay .025 to .0005','iterations':args.iterations,
              'observation_adapter':{'MID_HIP':'mean(L_Hip,R_Hip); actual SMPL pelvis remains NPZ root'},
              'beta_constraint':[-3,3],'loss_weights':{'joint_huber_delta_m':.05,'rotation_velocity':1e-6,'rotation_acceleration':1e-9,'translation_acceleration':1e-6,'pose_prior':2e-5,'beta_prior':1e-5,'flexion_proxy_limits':1e-4},
              'fitted_observations':result['keys'],'experimental_fingers':args.include_fingers,
              'license':'CC BY-NC-SA 4.0 plus SPL repository usage terms',
              'attribution':'SPL Open Data, Maple Leaf Sports & Entertainment Sport Performance Lab',
              'license_url':'https://github.com/Sport-Performance-Lab/SPL-Open-Data/blob/main/LICENSE'}
    write_csv(out/'joint_errors.csv',joint_rows);write_csv(out/'frame_errors.csv',frame_rows);write_csv(out/'validity.csv',ratios)
    write_csv(out/'mapping.csv',mapping_table(trial.names));write_csv(out/'optimization.csv',result['history']);write_csv(out/'summary.csv',[summary])
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2));(out/'summary.json').write_text(json.dumps(summary,indent=2))
    np.savez_compressed(out/'fitting_diagnostics.npz',observations=observations,fitted=pred,initial=initial_observed,
                        confidence=result['confidence'],original_m=trial.original_m,normalized_m=trial.normalized_world_m,source_keys=np.array(result['keys']))
    visualize(out,trial,result)
    if hashlib.sha256(args.source.read_bytes()).hexdigest()!=source_hash:raise RuntimeError('Source changed')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
