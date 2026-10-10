"""M4-2D-3A: analyze fitted SPL motion and connect it to contact Holding.
No release command, ball-state writes after initialization or controller edits.
"""
import argparse,hashlib,json,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco,numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from examples import m4_contact_lifting as lift
from examples.m3_control import write_csv
from smpl_sim.physics.basketball import HumanoidControllerBridge


def smooth(u):
    u=np.clip(u,0,1);return u**3*(10-15*u+6*u*u)


def geodesic_goal(holding,source,blend,limit_deg):
    delta=Rotation.from_matrix(holding.T@source).as_rotvec();angle=np.linalg.norm(delta)
    alpha=float(blend)*min(1.,np.radians(limit_deg)/max(angle,1e-12))
    return holding@Rotation.from_rotvec(alpha*delta).as_matrix(),alpha


def analyze(args):
    args.output.mkdir(parents=True,exist_ok=True)
    m=mujoco.MjModel.from_xml_path(str(args.models/'body_01.xml'));b=HumanoidControllerBridge(args.models/'body_01.xml',m,args.source);times=np.arange(len(b.motion.poses))/b.motion.fps
    names=[f'{side}_{part}' for side in ['R','L'] for part in ['Shoulder','Elbow','Wrist']];pos={n:[] for n in names};rot={n:[] for n in names};qs=[]
    for t in times:
        q=b.target(t);qs.append(q);b.data.qpos[:]=q;mujoco.mj_forward(b.model,b.data)
        for n in names:pos[n].append(b.data.xpos[b.model.body(n).id].copy());rot[n].append(b.data.xmat[b.model.body(n).id].reshape(3,3).copy())
    dt=1/b.motion.fps;rows=[];positions={n:np.array(p) for n,p in pos.items()};rotations={n:np.array(r) for n,r in rot.items()}
    velocity={n:savgol_filter(p,11,3,deriv=1,delta=dt,axis=0) for n,p in positions.items()};acceleration={n:savgol_filter(p,11,3,deriv=2,delta=dt,axis=0) for n,p in positions.items()}
    angular={n:np.vstack((np.zeros(3),Rotation.from_matrix(r[1:]@r[:-1].transpose(0,2,1)).as_rotvec()/dt)) for n,r in rotations.items()}
    e1=positions['R_Shoulder']-positions['R_Elbow'];e2=positions['R_Wrist']-positions['R_Elbow'];extension=np.degrees(np.arccos(np.clip(np.sum(e1*e2,axis=1)/(np.linalg.norm(e1,axis=1)*np.linalg.norm(e2,axis=1)),-1,1)));rate=savgol_filter(extension,11,3,deriv=1,delta=dt)
    for i,t in enumerate(times):
        row=dict(frame=i,source_time_s=t,elbow_extension_deg=extension[i],extension_rate_deg_s=rate[i])
        for n in names:
            for a,k in enumerate('xyz'):row[n+'_'+k+'_m']=positions[n][i,a];row[n+'_v'+k+'_ms']=velocity[n][i,a];row[n+'_a'+k+'_ms2']=acceleration[n][i,a];row[n+'_omega_'+k+'_rad_s']=angular[n][i,a]
            row[n+'_speed_ms']=np.linalg.norm(velocity[n][i]);row[n+'_orientation_from_initial_deg']=np.degrees(Rotation.from_matrix(rotations[n][0].T@rotations[n][i]).magnitude());row[n+'_palm_up']=float(-rotations[n][i,2,1])
        rows.append(row)
    write_csv(args.output/'source_kinematics.csv',rows);np.savez_compressed(args.output/'source_fk.npz',times=times,qpos=qs,**{n+'_positions':p for n,p in positions.items()},**{n+'_rotations':r for n,r in rotations.items()})
    vz=velocity['R_Wrist'][:,2];az=acceleration['R_Wrist'][:,2];eligible=np.flatnonzero(vz>.2);peak_v=int(np.argmax(vz));peak_e=int(eligible[np.argmax(rate[eligible])]) if len(eligible) else int(np.argmax(rate));peak_a_raw=int(np.argmax(az));interior=np.arange(6,len(times)-6);peak_a=int(interior[np.argmax(az[interior])])
    def intervals(mask):
        segments=[];start=None
        for i,ok in enumerate(np.r_[mask,False]):
            if ok and start is None:start=i
            elif not ok and start is not None:
                if i-start>=3:segments.append([float(times[start]),float(times[i-1])])
                start=None
        return segments
    rising=intervals(vz>.25*max(vz));accelerating=intervals((az>.25*az[peak_a])&(times>=.1)&(times<=times[-1]-.1))
    pose=b.motion.poses.reshape(-1,52,3)
    summary=dict(frames=len(times),fps=b.motion.fps,duration_s=b.motion.duration,derivative_filter='Savitzky-Golay 11 frames, polynomial 3, native 60Hz; no motion modification',
                 zero_wrist_rotations=bool(np.all(pose[:,20:22]==0)),zero_finger_rotations=bool(np.all(pose[:,22:]==0)),
                 rising_intervals_s=rising,positive_acceleration_intervals_s=accelerating,edge_acceleration_peak_time_s=float(times[peak_a_raw]),edge_acceleration_warning='Derivative endpoints excluded from main peak; fitted FK is not raw measured acceleration',peak_upward_speed_ms=float(vz[peak_v]),peak_upward_speed_time_s=float(times[peak_v]),peak_upward_acceleration_ms2=float(az[peak_a]),peak_acceleration_time_s=float(times[peak_a]),
                 peak_extension_rate_deg_s=float(rate[peak_e]),peak_extension_time_s=float(times[peak_e]),
                 shooting_candidate_interval_s=[max(0,float(times[min(peak_v,peak_e)])-.25),min(b.motion.duration,float(times[max(peak_v,peak_e)])+.25)],
                 release_candidates=[dict(proxy='peak upward wrist velocity',time_s=float(times[peak_v]),ground_truth=False),dict(proxy='peak elbow extension rate while rising',time_s=float(times[peak_e]),ground_truth=False)],ground_truth_release=None)
    c=next(c for c in json.loads((lift.PREVIOUS/'candidates.json').read_text()) if c['name']=='fk_fit_2');holding=lift.hand.GraspTargets(b,c).target(0);source=b.target(0);wr=b.model.body('R_Wrist').id
    b.data.qpos[:]=holding;mujoco.mj_forward(b.model,b.data);RH=b.data.xmat[wr].reshape(3,3).copy();PH=b.data.xpos[wr].copy()
    b.data.qpos[:]=source;mujoco.mj_forward(b.model,b.data);RS=b.data.xmat[wr].reshape(3,3).copy();PS=b.data.xpos[wr].copy()
    addresses=[int(b.model.joint(n).qposadr[0]) for n in lift.ARM]
    summary['initial_pose_gap']=dict(wrist_position_mm=float(np.linalg.norm(PH-PS)*1000),wrist_orientation_deg=float(np.degrees(Rotation.from_matrix(RH.T@RS).magnitude())),arm_goal_differences_rad=(holding[addresses]-source[addresses]).tolist(),source_palm_normal_up=float(-RS[2,1]),holding_palm_normal_up=float(-RH[2,1]))
    (args.output/'source_analysis.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary));return b


class IntegrationTargets(lift.hand.GraspTargets):
    def __init__(self,b,c,start,blend,condition,limit_deg):
        super().__init__(b,c);self.start=start;self.blend=blend;self.condition=condition;self.limit=limit_deg
        q=super().target(0);self.probe.qpos[:]=q;mujoco.mj_forward(self.m,self.probe);self.R=self.probe.xmat[self.m.body('R_Wrist').id].reshape(3,3).copy()
        self.R_goal=self.R.copy();self.R_source=self.R.copy();self.alpha=0.;self.source_time=0.
    def target(self,t,actual=None):
        holding=super().target(0,actual)
        if t<self.start:
            self.R_goal=self.R.copy();self.R_source=self.R.copy();self.alpha=0.;self.source_time=0.;return holding
        self.source_time=float(np.clip(t-self.start-self.blend,0,self.bridge.motion.duration))
        source=self.bridge.target(self.source_time);weight=float(smooth((t-self.start)/self.blend))
        # Holding root and source frame 0 coincide; retain the loader's unit quaternion.
        q=(1-weight)*holding+weight*source;q[3:7]=source[3:7]
        for name,value in self.candidate['angles'].items():q[int(self.m.joint(name).qposadr[0])]=value
        self.probe.qpos[:]=source;mujoco.mj_forward(self.m,self.probe);w=self.m.body('R_Wrist').id;self.R_source=self.probe.xmat[w].reshape(3,3).copy()
        if self.condition=='C':
            self.R_goal,self.alpha=geodesic_goal(self.R,self.R_source,weight,self.limit)
            self.probe.qpos[:]=q;mujoco.mj_forward(self.m,self.probe);data=actual if actual is not None else self.probe
            parent=data.xmat[self.m.body_parentid[w]].reshape(3,3)
            q[self.wrist]=lift.rotation_angles(parent.T@self.R_goal,data.qpos[self.wrist])
        else:
            self.probe.qpos[:]=q;mujoco.mj_forward(self.m,self.probe);self.R_goal=self.probe.xmat[w].reshape(3,3).copy();self.alpha=weight
        return q


def prepare(args):
    paths=list((ROOT/'outputs/m4_2d_1_hand').rglob('*'))+list((ROOT/'outputs/m4_2d_2_lifting').rglob('*'))
    paths += [ROOT/'examples'/n for n in ['m4_hand_kinematics.py','m4_hand_kinematics_report.py','m4_contact_lifting.py','m4_contact_lifting_report.py']]
    paths += [ROOT/'tests'/n for n in ['test_hand_kinematics.py','test_contact_lifting.py']]+[ROOT/'M4_2D_1_HAND_KINEMATICS.md',ROOT/'M4_2D_2_CONTACT_LIFTING.md']
    protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()};(args.output/'previous_hashes.json').write_text(json.dumps(protected,indent=2))
    c=next(c for c in json.loads((lift.PREVIOUS/'candidates.json').read_text()) if c['name']=='fk_fit_2')
    m,d,b,a,q=lift.hand.setup(args);(args.output/'axis_mapping.json').write_text((lift.PREVIOUS/'axis_mapping.json').read_text())
    saved=args.gui;args.gui=False
    try:lift.hand.trial(args,c,3.5)
    finally:args.gui=saved
    (args.output/'static_inputs.json').write_text(json.dumps(lift.input_signature(args),indent=2));lift.gate(args)
    targets,R,adrs=lift.arm_path(b.model,lift.hand.GraspTargets(b,c).target(0),.25,args.output)
    base=args.output/'baseline';base.mkdir(exist_ok=True);(base/'trials/fk_fit_2').mkdir(parents=True,exist_ok=True)
    for name in ['contact.xml','ik_path.npz']:shutil.copyfile(args.output/name,base/name)
    shutil.copyfile(args.output/'trials/fk_fit_2/summary.json',base/'trials/fk_fit_2/summary.json')
    opts=argparse.Namespace(**vars(args));opts.output=base;opts.height=.25;opts.start=1.;opts.ramp=4.;opts.end_hold=1.5;opts.gui=False
    A=lift.simulate(opts,'C',c)
    if not A['success']:raise RuntimeError('Lifting C baseline gate failed; no integration')
    return c


def simulate(args,condition,c,folder_name=None):
    lift.gate(args);A=json.loads((args.output/'baseline/C/summary.json').read_text())
    if not A['success']:raise RuntimeError('Lifting baseline must pass before integration')
    out=args.output/(folder_name or condition);out.mkdir(exist_ok=True);m=mujoco.MjModel.from_xml_path(str(args.output/'contact.xml'));d=mujoco.MjData(m)
    b=HumanoidControllerBridge(args.models/'body_01.xml',m,args.source);adapter=IntegrationTargets(b,c,args.start,args.blend,condition,args.max_wrist_angle)
    q,v=lift.initialize(m,d,b,adapter,c);w=m.body('R_Wrist').id;W=d.xmat[w].reshape(3,3);rel0=W.T@(d.qpos[q:q+3]-d.xpos[w]);root0=d.qpos[2]
    if not np.allclose(d.qpos[q:q+3],json.loads((args.output/'trials/fk_fit_2/summary.json').read_text())['initial_position_m'],atol=1e-10,rtol=0):raise RuntimeError('Initial placement changed')
    arm=np.array([int(m.joint(n).qposadr[0]) for n in lift.ARM]);names=lift.ARM+list(c['angles']);rows=[];joint_rows=[];contact_rows=[];states=[];times=[];viewer=None;loss_start=None;loss=None;seen=False
    duration=args.start+args.blend+b.motion.duration+.5;entry_end=args.start+args.blend+.5
    if args.gui:
        from mujoco import viewer as mv
        viewer=mv.launch_passive(m,d);viewer.cam.lookat[:]=[.25,0,1.1];viewer.cam.distance=2.4;viewer.cam.azimuth=110;viewer.cam.elevation=-10
    begun=time.monotonic()
    for step in range(round(duration/m.opt.timestep)):
        target=adapter.target(d.time,d);source=b.target(adapter.source_time);bp=d.qpos[q:q+7].copy();bv=d.qvel[v:v+6].copy();adapter.control(d,target)
        assert np.array_equal(bp,d.qpos[q:q+7]) and np.array_equal(bv,d.qvel[v:v+6]);assert m.neq==0 and not d.qfrc_applied.any() and not d.xfrc_applied.any()
        mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():raise RuntimeError('Nonfinite state')
        t=float(d.time);n,nf,F,ff,depth,slip=lift.contact.hand_contacts(m,d);W=d.xmat[w].reshape(3,3);rel=W.T@(d.qpos[q:q+3]-d.xpos[w])
        seen=seen or n>0
        if n>0:loss_start=None
        elif seen and loss_start is None:loss_start=t
        if loss is None and loss_start is not None and t-loss_start>=.05:loss=loss_start
        feet,margin,_=lift.contact.contacts(m,d);tilt=np.degrees(np.arccos(np.clip(d.xmat[m.body('Pelvis').id].reshape(3,3)[2,1],-1,1)))
        fs,_=lift.foot_contact_slip(m,d)
        rows.append(dict(time_s=t,source_time_s=adapter.source_time,alpha=adapter.alpha,wrist_x_m=d.xpos[w,0],wrist_y_m=d.xpos[w,1],wrist_z_m=d.xpos[w,2],ball_x_m=d.qpos[q],ball_y_m=d.qpos[q+1],ball_z_m=d.qpos[q+2],
                         ball_vx_ms=d.qvel[v],ball_vy_ms=d.qvel[v+1],ball_vz_ms=d.qvel[v+2],ball_speed_ms=np.linalg.norm(d.qvel[v:v+3]),contacts=n,finger_contacts=nf,force_z_N=F[2],slip_ms=slip,drift_m=np.linalg.norm(rel-rel0),
                         relative_x_m=rel[0],relative_y_m=rel[1],relative_z_m=rel[2],palm_normal_up=-W[2,1],prescribed_orientation_error_deg=np.degrees(Rotation.from_matrix(adapter.R_goal.T@W).magnitude()),source_orientation_error_deg=np.degrees(Rotation.from_matrix(adapter.R_source.T@W).magnitude()),
                         goal_orientation_from_holding_deg=np.degrees(Rotation.from_matrix(adapter.R.T@adapter.R_goal).magnitude()),arm_tracking_rmse_rad=np.sqrt(np.mean((d.qpos[arm]-target[arm])**2)),source_arm_rmse_rad=np.sqrt(np.mean((d.qpos[arm]-source[arm])**2)),
                         source_shoulder_elbow_rmse_rad=np.sqrt(np.mean((d.qpos[arm[:6]]-source[arm[:6]])**2)),target_source_arm_rmse_rad=np.sqrt(np.mean((target[arm]-source[arm])**2)),
                         com_margin_m=margin,both_feet=int(min(feet.values())>.1),foot_slip_ms=np.sqrt((fs['L']['rms_speed']**2+fs['R']['rms_speed']**2)/2),fallen=int(d.qpos[2]<.65*root0 or tilt>35)))
        if step%15==0:
            states.append(d.qpos.copy());times.append(t);jr=dict(time_s=t,source_time_s=adapter.source_time)
            for name in names:
                j=m.joint(name);adr=int(j.qposadr[0]);dof=int(j.dofadr[0]);jr[name+'_error_rad']=d.qpos[adr]-target[adr];jr[name+'_source_error_rad']=d.qpos[adr]-source[adr];jr[name+'_torque_Nm']=d.qfrc_actuator[dof]
            joint_rows.append(jr)
            bg=m.geom('basketball_geom').id
            for ci in range(d.ncon):
                ct=d.contact[ci]
                if bg not in (ct.geom1,ct.geom2):continue
                other=int(ct.geom2 if ct.geom1==bg else ct.geom1);name=m.geom(other).name
                if name!='R_Wrist' and not lift.contact.is_finger(name):continue
                force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force);normal=ct.frame[:3]*(-1 if ct.geom1==bg else 1)
                contact_rows.append(dict(time_s=t,geom=name,normal_force_N=force[0],normal_x=normal[0],normal_y=normal[1],normal_z=normal[2]))
        if viewer:
            if not viewer.is_running():break
            viewer.sync();time.sleep(max(0,t-(time.monotonic()-begun)))
    if viewer:viewer.close()
    write_csv(out/'telemetry.csv',rows);write_csv(out/'joint_tracking_torques.csv',joint_rows)
    if contact_rows:write_csv(out/'contacts.csv',contact_rows)
    np.savez_compressed(out/'states.npz',qpos=states,time_s=times)
    def metrics(window):
        r=[r for r in rows if args.start<=r['time_s']<=window];gap=longest=0
        for x in r:gap=gap+1 if x['contacts']==0 else 0;longest=max(longest,gap)
        ok=bool(seen and (loss is None or loss>window) and max(x['drift_m'] for x in r)<.05 and not any(x['fallen'] for x in r) and min(x['com_margin_m'] for x in r)>=0 and np.mean([x['both_feet'] for x in r])>=.99 and not any(w.number for w in d.warning) and d.time>=window-1e-8)
        return dict(success=ok,end_time_s=window,max_relative_drift_m=max(x['drift_m'] for x in r),contact_ratio=float(np.mean([x['contacts']>0 for x in r])),max_contact_gap_s=longest*m.opt.timestep,
                    mean_vertical_force_N=float(np.mean([x['force_z_N'] for x in r])),arm_tracking_rmse_rad=float(np.sqrt(np.mean([x['arm_tracking_rmse_rad']**2 for x in r]))),source_arm_rmse_rad=float(np.sqrt(np.mean([x['source_arm_rmse_rad']**2 for x in r]))),
                    source_shoulder_elbow_rmse_rad=float(np.sqrt(np.mean([x['source_shoulder_elbow_rmse_rad']**2 for x in r]))),max_source_orientation_error_deg=max(x['source_orientation_error_deg'] for x in r),max_prescribed_orientation_error_deg=max(x['prescribed_orientation_error_deg'] for x in r),
                    min_com_margin_m=min(x['com_margin_m'] for x in r),both_feet_ratio=float(np.mean([x['both_feet'] for x in r])),fallen=bool(any(x['fallen'] for x in r)))
    summary=dict(goal_algorithm='bounded geodesic; angle bound multiplied by minimum-jerk blend',condition=condition,duration_s=float(d.time),blend_s=args.blend,first_contact_loss_s=loss,source_time_at_loss_s=max(0,loss-args.start-args.blend) if loss else None,max_wrist_angle_deg=args.max_wrist_angle if condition=='C' else None,
                 entry_definition='blend plus first 0.5s source; full playback reported separately',entry=metrics(entry_end),full=metrics(duration-m.opt.timestep),warnings=[int(w.number) for w in d.warning],ball_state_writes_after_initialization=0,neq=m.neq,release_implemented=False)
    perjoint=[dict(joint=n,tracking_rmse_rad=float(np.sqrt(np.mean([r[n+'_error_rad']**2 for r in joint_rows if r['time_s']>=args.start]))),source_rmse_rad=float(np.sqrt(np.mean([r[n+'_source_error_rad']**2 for r in joint_rows if r['time_s']>=args.start])))) for n in names]
    write_csv(out/'per_joint_rmse.csv',perjoint);(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary));return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',type=Path,default=ROOT.parents[2]/'basketball-r1/SMPLSim/outputs/smplh_physics');p.add_argument('--source',type=Path,default=ROOT/'outputs/m3_spl_fitting/spl_P0002_T0049.npz');p.add_argument('--output',type=Path,default=ROOT/'outputs/m4_2d_3a_integration')
    p.add_argument('--condition',choices=['all','A','B','C'],default='all');p.add_argument('--analyze-only',action='store_true');p.add_argument('--start',type=float,default=1.);p.add_argument('--blend',type=float,default=2.);p.add_argument('--max-wrist-angle',type=float,default=15.);p.add_argument('--gui',action='store_true')
    args=p.parse_args()
    if not np.isfinite([args.start,args.blend,args.max_wrist_angle]).all() or args.start<.3 or args.blend<1 or not 0<args.max_wrist_angle<=30:p.error('Invalid integration parameters')
    if args.analyze_only:analyze(args);return
    if args.condition in ['all','A']:
        analyze(args);c=prepare(args)
    else:c=next(c for c in json.loads((lift.PREVIOUS/'candidates.json').read_text()) if c['name']=='fk_fit_2')
    for condition in ['B','C'] if args.condition=='all' else ([args.condition] if args.condition in ['B','C'] else []):simulate(args,condition,c)

if __name__=='__main__':main()
