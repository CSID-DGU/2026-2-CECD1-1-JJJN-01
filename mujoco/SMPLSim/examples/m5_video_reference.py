"""Video-guided feature timing, independent adapter and contact-only stage gates.
Pixel observations constrain timing only. 3D posture amplitudes are hypotheses.
"""
import sys,json,argparse,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation
from examples.m5_shooting_trajectory import TrajectoryEnv,TrajectoryTargets,so3_interpolate
from examples.m5_bimanual_reorientation import pose_ik
from examples.m5_bimanual_body_ik import head_ik,calibration,trunk_axis
from examples.m5_coordinated_reference import lift,MODELS,SOURCE,unit
from examples.m5_upper_body_ik import joint_ids,minimum_jerk
from examples.m3_control import write_csv
OUT=ROOT/'outputs/m5_2_video_reference'

def knee_bend(m,d,side):
 h,k,a=[d.xpos[m.body(side+'_'+n).id] for n in ('Hip','Knee','Ankle')];return float(np.arccos(np.clip(unit(k-h)@unit(a-k),-1,1)))

def observed_progress():
 rows=np.genfromtxt(OUT/'motion_features_2d.csv',names=True,delimiter=',');rows=rows[rows['frame']<=235];t=rows['pts_s'];u=(t-t[0])/(t[-1]-t[0]);curves={}
 for key,values in [('ball',-rows['ball_y_px']),('legs',-rows['knee_bend_2d_deg']),('trunk',-rows['trunk_tilt_2d_deg'])]:
  if not np.isfinite(values).all():raise ValueError('Observed feature missing; correct annotations/detection before timing fitting')
  values=np.maximum.accumulate(values);span=values[-1]-values[0]
  if span<=0:raise ValueError('No observed progress')
  curves[key]=PchipInterpolator(u,(values-values[0])/span)
 return curves,float(t[0]),float(t[-1])

def lower_path(m,initial,extra_bend_deg):
 names=[f'{s}_{b}_{a}' for s in ('R','L') for b in ('Hip','Knee','Ankle') for a in 'xyz'];ids=joint_ids(m,names);bounds=np.array([m.joint(n).range for n in names]);probe=mujoco.MjData(m);probe.qpos[:]=initial;mujoco.mj_kinematics(m,probe);footp=[probe.xpos[m.body(s+'_Ankle').id].copy() for s in ('R','L')];footR=[probe.xmat[m.body(s+'_Ankle').id].reshape(3,3).copy() for s in ('R','L')];bend0=np.array([knee_bend(m,probe,s) for s in ('R','L')]);anchor=np.r_[initial[ids],initial[:3]];previous=anchor.copy();samples=[];rows=[]
 lo=np.r_[np.maximum(bounds[:,0]+1e-5,anchor[:18]-.7),initial[:3]-.18];hi=np.r_[np.minimum(bounds[:,1]-1e-5,anchor[:18]+.7),initial[:3]+.12]
 for u in np.linspace(0,1,21):
  desired=bend0+np.radians(extra_bend_deg)*u
  def f(x):
   probe.qpos[:]=initial;probe.qpos[ids]=x[:18];probe.qpos[:3]=x[18:];mujoco.mj_kinematics(m,probe)
   residual=[]
   for s,p,R in zip(('R','L'),footp,footR):
    b=m.body(s+'_Ankle').id;residual.extend(40*(probe.xpos[b]-p));residual.extend(2*Rotation.from_matrix(R@probe.xmat[b].reshape(3,3).T).as_rotvec())
   return np.r_[residual,3*(np.array([knee_bend(m,probe,s) for s in ('R','L')])-desired),.02*(x-anchor),.05*(x-previous),3*(x[18:20]-anchor[18:20])]
  fit=least_squares(f,np.clip(previous,lo,hi),bounds=(lo,hi),max_nfev=70);f(fit.x);q=probe.qpos.copy();samples.append(q);rows.append(dict(bend_progress=u,R_bend_deg=np.degrees(knee_bend(m,probe,'R')),L_bend_deg=np.degrees(knee_bend(m,probe,'L')),virtual_root_z=q[2],foot_residual_peak_mm=max(np.linalg.norm(probe.xpos[m.body(s+'_Ankle').id]-p)*1000 for s,p in zip(('R','L'),footp))));previous=fit.x
 return ids,np.array(samples),rows

def chest_ik(m,q,cal,extension_deg,previous):
 names=[f'{b}_{a}' for b in ('Torso','Spine','Chest') for a in 'xyz'];ids=joint_ids(m,names);dofs=[int(m.joint(n).dofadr[0]) for n in names];p=mujoco.MjData(m);p.qpos[:]=q;p.qpos[ids]=previous;axis=cal['trunk_axis'];tilt=np.arccos(np.clip(axis[2],-1,1));goal=unit(unit(np.r_[axis[:2],0])*np.sin(max(0,tilt-np.radians(extension_deg)))+[0,0,np.cos(max(0,tilt-np.radians(extension_deg)))]);v=np.cross(axis,goal);angle=np.arccos(np.clip(axis@goal,-1,1));R=(Rotation.from_rotvec(unit(v)*angle).as_matrix() if np.linalg.norm(v)>1e-8 else np.eye(3))@cal['chest_rotation'];bounds=np.array([m.joint(n).range for n in names]);lo=np.maximum(bounds[:,0]+1e-5,cal['trunk_q']-.4);hi=np.minimum(bounds[:,1]-1e-5,cal['trunk_q']+.4)
 for _ in range(20):
  mujoco.mj_kinematics(m,p);mujoco.mj_comPos(m,p);bid=m.body('Chest').id;jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv));mujoco.mj_jacBody(m,p,jp,jr,bid);J=jr[:,dofs];e=Rotation.from_matrix(R@p.xmat[bid].reshape(3,3).T).as_rotvec();step=np.linalg.solve(J.T@J+.01*np.eye(9),J.T@e-.01*(p.qpos[ids]-cal['trunk_q']));step*=min(1,.1/max(np.linalg.norm(step),1e-12));p.qpos[ids]=np.clip(p.qpos[ids]+step,lo,hi)
 return p.qpos[ids].copy()

class VideoTargets(TrajectoryTargets):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.curves,self.source_start,self.source_end=observed_progress();self.new_q=None;self.new_v=None;self.new_time=0.;self.fit_time=-1.;self.new_fit=None;self.initial_root=None
 def target(self,t,actual=None):
  original=super().target(t,actual)
  if t<=0 or actual is None:
   if self.exp['video_stage']!='A' and hasattr(self,'lower_ids'):original[self.lower_ids]=self.lower_samples[-1,self.lower_ids]
   return original
  c=self.exp;stage=c['video_stage']
  if stage=='A':return original
  if self.rotation_start is None:
   if c.get('initialized_bent',False):original[self.lower_ids]=self.lower_samples[-1,self.lower_ids]
   return original
  if self.new_q is None:
   self.new_q=original.copy();self.new_q[self.lower_ids]=actual.qpos[self.lower_ids];self.new_v=np.zeros_like(original);self.initial_root=actual.qpos[:3].copy();self.initial_pelvis=actual.xpos[actual.model.body('Pelvis').id].copy();self.initial_cal=calibration(self.m,self.probe_from(actual));self.initial_ball=self.reference['ball_initial'].copy();self.new_time=t
  elapsed=0 if self.rotation_start is None else max(0,t-self.rotation_start);clock=minimum_jerk(elapsed/c['motion_seconds']);progress={k:float(v(clock)) for k,v in self.curves.items()};self.rise_progress=progress['ball'];self.rotation_progress=progress['ball'];self.leg_progress=progress['legs'];self.trunk_progress=progress['trunk'];self.source_pts=self.source_start+clock*(self.source_end-self.source_start)
  prep=minimum_jerk(t/c['prep_seconds']);depth=(1-progress['legs']) if c.get('initialized_bent',False) else prep*(1-progress['legs']);at=depth*(len(self.lower_samples)-1);i=min(int(at),len(self.lower_samples)-2);qleg=(1-(at-i))*self.lower_samples[i]+(at-i)*self.lower_samples[i+1]
  root_delta=actual.xpos[actual.model.body('Pelvis').id]-self.initial_pelvis
  background=actual.qpos[:self.m.nq].copy();background[self.lower_ids]=qleg[self.lower_ids];background[self.trunk_ids]=chest_ik(self.m,background,self.initial_cal,c['video_trunk_deg']*progress['trunk'],self.new_q[self.trunk_ids]);background=head_ik(self.m,background,self.reference['rim_center'],self.new_q[self.head_ids]);probe=mujoco.MjData(self.m);probe.qpos[:]=background;mujoco.mj_kinematics(self.m,probe)
  goal=probe.xpos[self.m.body('Head').id]+.26*self.reference['forward']+[0,0,.13];start=self.initial_ball+(1-progress['ball'])*root_delta;center=(1-progress['ball'])*start+progress['ball']*goal
  self.ball_target=center;W=self.desired_right_R.copy();rel=self.reference['rel0'].copy()
  if stage in ('C','D'):
   W=Rotation.from_rotvec(self.reference['side']*np.radians(c['shooting_tilt_deg'])*progress['ball']).as_matrix()@W
   rel=rel+np.array([c['grip_shift_m'],0.,0.])*progress['ball']
  self.new_grip_relative=rel.copy();self.desired_right_R=W;self.cartesian_right=center-W@rel;self.cartesian_right+=np.clip(.25*(center-actual.qpos[int(actual.model.joint('basketball_free').qposadr[0]):][:3]),-.02,.02)
  if t-self.fit_time>=.033-1e-9:
   right,zR=pose_ik(self.m,background,'R',self.cartesian_right,W,self.new_q[self.adrs],iterations=45);left,zL=pose_ik(self.m,right,'L',self.cartesian_left,self.desired_left_R,self.new_q[self.left_ids],iterations=35);self.new_fit=left.copy();self.new_fit[self.adrs]=right[self.adrs];self.jac_reports={'R':zR,'L':zL};self.fit_time=t
  desired=original.copy();selected=np.r_[self.lower_ids,self.trunk_ids,self.head_ids,self.adrs,self.left_ids];desired[selected]=self.new_fit[selected];dt=max(t-self.new_time,self.m.opt.timestep);self.new_time=t;acc=np.clip(64*(desired-self.new_q)-16*self.new_v,-8,8);self.new_v=np.clip(self.new_v+acc*dt,-1.5,1.5);self.new_q+=self.new_v*dt;return self.new_q.copy()
 def probe_from(self,actual):
  p=mujoco.MjData(self.m);p.qpos[:]=actual.qpos[:self.m.nq];mujoco.mj_kinematics(self.m,p);return p

class VideoEnv(TrajectoryEnv):
 def __init__(self,stage='A',motion_seconds=4.,knee_extra=20.,trunk_degrees=10.,shooting_tilt=10.,grip_shift=.01,gui=False):
  self.video_config=dict(video_stage=stage,video_trunk_deg=trunk_degrees,shooting_tilt_deg=shooting_tilt,grip_shift_m=grip_shift)
  super().__init__(condition='D' if stage=='D' else 'B',motion_seconds=motion_seconds,face_path=True,stable_grip=True,yaw=-15.,trunk_extension=0,wrist_extension=0,gui=gui)
  self.lower_ids,self.lower_samples,self.lower_rows=lower_path(self.bridge.model,self.d.qpos[:self.bridge.model.nq].copy(),knee_extra)
  self.adapter.lower_ids=self.lower_ids;self.adapter.lower_samples=self.lower_samples
  if stage!='A':
   # Initial placement only, before the first simulation step: virtual FK stance
   # becomes a free-root starting state. No root/ball state writes in control.
   before=self.d.qpos.copy();new=before.copy();new[:self.bridge.model.nq]=self.lower_samples[-1];self.d.qpos[:]=new;self.d.qvel[:]=0.;mujoco.mj_forward(self.m,self.d)
   W=self.d.xmat[self.m.body('R_Wrist').id].reshape(3,3);self.d.qpos[self.ball_q:self.ball_q+3]=self.d.xpos[self.m.body('R_Wrist').id]+W@self.rel0;mujoco.mj_forward(self.m,self.d)
   self.ref=dict(self.ref);self.ref['ball_initial']=self.d.qpos[self.ball_q:self.ball_q+3].copy();self.ref['initial']=self.d.qpos.copy();self.ref['ball_goal_shoot']=self.d.xpos[self.m.body('Head').id]+.26*self.ref['forward']+[0,0,.13];self.adapter.reference=self.ref
   self.frames=[self.d.qpos.copy()];self.frame_times=[0.];self.initial_landmarks={n:self.d.xpos[self.m.body(n).id].copy() for n in ('R_Shoulder','R_Elbow','R_Wrist')};self.experiment['initialized_bent']=True

  extra=[f'{s}_{b}_{a}' for s in ('R','L') for b in ('Hip','Knee','Ankle') for a in 'xyz'];self.joint_names=list(self.joint_names)+extra;self.ids=np.r_[self.ids,joint_ids(self.m,extra)];self.max_seconds=15.
 def _adapter(self):
  if not self.coord_ready:return super()._adapter()
  self.experiment.update(self.trajectory_config);self.experiment.update(self.video_config);self.bridge=lift.HumanoidControllerBridge(MODELS/'body_01.xml',self.m,SOURCE);self._upper_cache_time=None;self.adapter=VideoTargets(self.bridge,self.candidate,self.arm_ids,self.ref,4,self.condition,config=self.config,experiment=self.experiment)
 def sample(self):
  r=super().sample()
  for s in ('R','L'):r[s+'_knee_bend_deg']=float(np.degrees(knee_bend(self.m,self.d,s)))
  r['pelvis_z_m']=float(self.d.xpos[self.m.body('Pelvis').id,2]);r['leg_progress']=getattr(self.adapter,'leg_progress',0.);r['trunk_progress']=getattr(self.adapter,'trunk_progress',0.);r['source_pts_s']=getattr(self.adapter,'source_pts',0.);return r
def run_stage(name,**cfg):
    folder=OUT/name
    if folder.exists():raise FileExistsError(folder)
    folder.mkdir();e=VideoEnv(**cfg);reason=None;start=time.perf_counter();initial=e.d.qpos.copy()
    write_csv(folder/'lower_fk.csv',e.lower_rows);np.savez_compressed(folder/'lower_reference.npz',qpos=e.lower_samples,leg_qpos_indices=e.lower_ids,virtual_root_only=True)
    try:
        while e.d.time<e.max_seconds:
            reason=e.physical_step()
            if reason:break
            a=e.adapter
            if a.rotation_start is not None and a.rise_progress>=1 and a.rotation_progress>=1 and e.d.time>a.rotation_start+e.experiment['motion_seconds']*(2 if False else 1)+max(0,e.experiment['lag_seconds'])+.6:
                reason='sequence_completed';break
        data=e.telemetry;p=np.array([r['time_s'] for r in data]);z=np.array([r['ball_z'] for r in data]);finger=np.array([r['finger_change_deg'] for r in data]);vz=np.gradient(z,p);vr=np.gradient(finger,p)
        mask=np.array([r['phase']!='A' for r in data]);overlap=mask&(vz>.01)&(abs(vr)>2.)
        q=np.array([[r[n+'_actual'] for n in e.joint_names] for r in e.control_rows]);target=np.array([[r[n+'_target'] for n in e.joint_names] for r in e.control_rows]);ts=np.array([r['time_s'] for r in e.control_rows]);stats={}
        for label,x in [('actual',q),('target',target)]:
            for order in ('velocity','acceleration','jerk'):
                x=np.gradient(x,ts,axis=0);stats[label+'_'+order+'_rms']=float(np.sqrt(np.mean(x*x)))
        requested_height=float(e.ref['ball_goal_shoot'][2]-e.ref['ball_initial'][2]) if e.experiment['face_path'] else e.experiment['height_m']
        late=data[-min(len(data),225):];rise=float((max(z)-initial[e.ball_q+2])*1000);valid=reason=='sequence_completed' and rise>requested_height*800 and max(r['ball_path_error_mm'] for r in late)<50 and all(r['com_margin_mm']>0 for r in late)
        s=dict(name=name,config=e.experiment,duration_s=float(e.d.time),reason=reason or 'time_limit',motion_start_s=e.adapter.rotation_start,ball_rise_mm=rise,wrist_rise_mm=float((max(r['wrist_z'] for r in data)-e.initial_landmarks['R_Wrist'][2])*1000),finger_change_peak_deg=max(finger),wrist_extension_peak_deg=max(r['wrist_extension_delta_deg'] for r in data),trunk_extension_peak_deg=max(r['trunk_extension_delta_deg'] for r in data),actual_overlap_seconds=float(np.sum(np.diff(p,prepend=p[0])*overlap)),actual_overlap_motion_fraction=float(np.mean(overlap[mask])) if mask.any() else 0.,left_contact_ratio=float(np.mean([r['left_contacts']>0 for r in data if r['phase']!='A'])) if mask.any() else 0.,right_contact_ratio=float(np.mean([r['right_contacts']>0 for r in data])),R_vertical_mean_N=float(np.mean([r['R_force_z'] for r in data])),L_vertical_mean_N=float(np.mean([r['L_force_z'] for r in data])),left_force_peak_N=max(r['left_normal_N'] for r in data),left_work_J=float(np.trapezoid([r['left_force_power_W'] for r in data],p)),path_peak_mm=max(r['ball_path_error_mm'] for r in data),local_movement_peak_mm=max(r['right_local_change_mm'] for r in data),slip_rms_mm_s=float(np.sqrt(np.mean([r['right_slip_m_s']**2 for r in data]))*1000),tracking_rmse_rad=float(np.sqrt(np.mean((q-target)**2))),com_min_mm=min(r['com_margin_mm'] for r in data),both_feet_ratio=float(np.mean([r['both_feet'] for r in data])),fallen=any(r['fallen'] for r in data),warnings=[int(w.number) for w in e.d.warning],ball_writes_after_initialization=0,neq=e.m.neq,raising_gate_pass=bool(valid),shooting_ready_success=False,physical_reference_valid=False,large_ppo_allowed=False,wall_seconds=time.perf_counter()-start,**stats)
        write_csv(folder/'physics.csv',data);write_csv(folder/'contacts.csv',e.points);write_csv(folder/'joints.csv',e.control_rows);np.savez_compressed(folder/'states.npz',time_s=e.frame_times,qpos=e.frames);np.savez_compressed(folder/'reference.npz',time_s=e.frame_times[1:],target_qpos=e.targets,fps=30.,physical_reference_valid=False,representation='MuJoCo qpos; not SMPL axis-angle');(folder/'summary.json').write_text(json.dumps(s,indent=2));print(json.dumps(s),flush=True);return s
    finally:e.close()


def audit():
 old=json.loads((OUT/'preservation_before.json').read_text());changes=[p for p,h in old.items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];z=dict(files_checked=len(old),changed=changes);(OUT/'preservation_audit.json').write_text(json.dumps(z,indent=2));assert not changes;return z
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=list('ABCD'),default='A');p.add_argument('--name',default='A_baseline');p.add_argument('--duration',type=float,default=4.);p.add_argument('--knee-extra',type=float,default=20.);p.add_argument('--trunk',type=float,default=10.);p.add_argument('--tilt',type=float,default=10.);p.add_argument('--grip-shift',type=float,default=.01);p.add_argument('--audit',action='store_true');a=p.parse_args()
 if a.audit:print(audit())
 else:run_stage(a.name,stage=a.stage,motion_seconds=a.duration,knee_extra=a.knee_extra,trunk_degrees=a.trunk,shooting_tilt=a.tilt,grip_shift=a.grip_shift)
