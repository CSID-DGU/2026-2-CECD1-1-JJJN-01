"""M6-2: sequential shooting IK, unchanged M6 physics and local velocity tracking.

Ball Cartesian paths are offline IK objectives only. Runtime never writes ball
state after initialization. All outputs and derived scenes live in a new folder.
"""
import argparse, json, hashlib, time
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import least_squares
from examples.m6 import core
from examples.m6.core import ROOT, MODELS, ids, smooth, lower_path, RIM_CENTER
from examples.m6.palm_to_hoop import basis, blend
from examples.m5_bimanual_body_ik import unit, trunk_axis, head_ik
from examples.m5_bimanual_reorientation import pose_ik
from examples.m5_coordinated_reference import elbow_bend
from examples.m3_control import write_csv

OUT=ROOT/'outputs/m6_2_natural_shooting'
OLD=core.OUT
core.OUT=OUT # runtime output redirection; original source is unchanged

def audit(start=False):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/'preservation_before.json'
 if start and not p.exists():
  p.write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.rglob('*') if f.is_file() and not any(n in f.parts for n in ('.git','__pycache__','m6_2_natural_shooting')) and f.name!='.DS_Store'},indent=2))
 if not start:
  z=json.loads(p.read_text());changed=[f for f,h in z.items() if not Path(f).exists() or hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h]
  (OUT/'preservation_audit.json').write_text(json.dumps(dict(protected=len(z),changed=changed),indent=2));assert not changed,changed

def model():return mujoco.MjModel.from_xml_path(str(MODELS/'body_01.xml'))
def initial():return np.load(ROOT/'outputs/m5_1_motion_redesign/B_face/states.npz')['qpos'][0,:160].copy()
def geometry(m,d,offset):
 w=m.body('R_Wrist').id;W=d.xmat[w].reshape(3,3);return d.xpos[w]+W@offset, W, unit(d.xpos[m.body('R_Middle1').id]-d.xpos[w])

def diagnose():
 m=model();d=mujoco.MjData(m);d.qpos[:]=initial();mujoco.mj_forward(m,d)
 offset=np.load(OLD/'reference_shoot.npz')['offset'];p0,W0,f0=geometry(m,d,offset);a0=unit(d.xpos[m.body('R_Wrist').id]-d.xpos[m.body('R_Elbow').id]);rows=[]
 names=[f'R_{b}_{a}' for b in ('Shoulder','Elbow','Wrist','Hip','Knee','Ankle') for a in 'xyz']+[f'{b}_{a}' for b in ('Torso','Spine','Chest') for a in 'xyz']
 for name in names:
  j=m.joint(name);q=d.qpos.copy();axis=d.xaxis[j.id].copy()
  for delta in [-.1,.1]:
   p=mujoco.MjData(m);p.qpos[:]=q;p.qpos[int(j.qposadr[0])]=np.clip(q[int(j.qposadr[0])]+delta,*j.range);mujoco.mj_forward(m,p);c,W,f=geometry(m,p,offset)
   rows.append(dict(joint=name,delta_rad=delta,actual_delta_rad=float(p.qpos[int(j.qposadr[0])]-q[int(j.qposadr[0])]),axis_x=axis[0],axis_y=axis[1],axis_z=axis[2],lower_rad=j.range[0],upper_rad=j.range[1],ball_dx_mm=(c-p0)[0]*1000,ball_dy_mm=(c-p0)[1]*1000,ball_dz_mm=(c-p0)[2]*1000,world_rotation_deg=np.degrees(Rot.from_matrix(W@W0.T).magnitude()),forearm_axis_alignment=axis@a0,finger_change_deg=np.degrees(np.arccos(np.clip(f@f0,-1,1))),palm_change_deg=np.degrees(np.arccos(np.clip((-W[:,1])@(-W0[:,1]),-1,1)))))
 write_csv(OUT/'axis_probes.csv',rows)
 (OUT/'anchor.json').write_text(json.dumps(dict(local_offset_m=offset.tolist(),distance_from_wrist_m=float(np.linalg.norm(offset)),palm_normal_proxy='negative wrist local Y; anatomical normal not certified',world_up='Z; gravity [0,0,-9.81]',rim_center_m=RIM_CENTER.tolist(),forearm_joint='R_Elbow three hinges; no separate forearm body hinge'),indent=2))

def make(name,raise_s=1.1,push_s=.4,snap=0.,release_fraction=.55,lower_s=None):
 path=OUT/'references'/f'{name}.npz';path.parent.mkdir(exist_ok=True)
 if path.exists():return path
 m=model();base=initial();leg,qs,_=lower_path(m,base,22);start=qs[-1].copy();d=mujoco.MjData(m);d.qpos[:]=start;mujoco.mj_forward(m,d)
 offset=np.load(OLD/'reference_shoot.npz')['offset'];p0,W0,f0=geometry(m,d,offset);forward=unit(np.r_[(RIM_CENTER-p0)[:2],0]);side=unit(np.cross([0,0,1],forward));local=basis(-W0[:,1],f0);local=W0.T@local
 # Set point relative to head, not a long forward reach. Elbow stays below ball.
 head=d.xpos[m.body('Head').id].copy();shoulder=d.xpos[m.body('R_Shoulder').id].copy();setpoint=head+.12*forward+.09*side+np.array([0,0,.04]);setpoint[2]=max(head[2]+.04,shoulder[2]+.25)
 # Palm supports below/behind; finger direction lies in the sagittal plane.
 normal=unit(.32*forward+np.array([0,0,.95]));cock=basis(normal,forward)@local.T
 push_end=setpoint+.30*forward+np.array([0,0,.32])
 la=ids(m,[f'L_{b}_{a}' for b in ('Shoulder','Elbow','Wrist') for a in 'xyz']);ra=ids(m,[f'R_{b}_{a}' for b in ('Shoulder','Elbow','Wrist') for a in 'xyz']);tr=ids(m,[f'{b}_{a}' for b in ('Torso','Spine','Chest') for a in 'xyz']);hd=ids(m,[f'{b}_{a}' for b in ('Neck','Head') for a in 'xyz']);bounds=np.array([m.jnt_range[m.jnt_qposadr==a][0] for a in ra]);tl=np.array([m.jnt_range[m.jnt_qposadr==a][0] for a in tr]);L0=d.xpos[m.body('L_Wrist').id].copy();WL=d.xmat[m.body('L_Wrist').id].reshape(3,3).copy();axis0=trunk_axis(m,d);previous=start.copy();samples=[];rows=[]
 t0=.18;push_start=t0+raise_s*.82;release=push_start+push_s*release_fraction;total=max(push_start+push_s+.45,release+2.8);times=np.arange(0,total+1/120,1/120);lower_s=lower_s or raise_s*.85
 for t in times:
  up=smooth((t-t0)/raise_s);ext=smooth((t-t0)/lower_s);push=smooth((t-push_start)/push_s);q=start.copy();q[leg]=(1-ext)*start[leg]+ext*base[leg];q[:3]=(1-ext)*start[:3]+ext*base[:3]
  tilt=(1-ext)*np.arccos(np.clip(axis0[2],-1,1))+ext*np.radians(5);axisgoal=unit(np.r_[axis0[:2],0])*np.sin(tilt)+np.array([0,0,np.cos(tilt)])
  def trunk(x):
   d.qpos[:]=q;d.qpos[tr]=x;mujoco.mj_kinematics(m,d);return np.r_[7*(trunk_axis(m,d)-axisgoal),.04*(x-previous[tr])]
  q[tr]=least_squares(trunk,np.clip(previous[tr],tl[:,0]+1e-6,tl[:,1]-1e-6),bounds=(tl[:,0]+1e-6,tl[:,1]-1e-6),max_nfev=25).x
  center=(1-up)*p0+up*setpoint+push*(push_end-setpoint);W=blend(W0,cock,up)
  # World sagittal snap defined by palm/finger geometry. Sign is probed, not inferred from a joint name.
  snap_progress=smooth((t-push_start-push_s*.15)/(push_s*.7));W=Rot.from_rotvec(side*np.radians(snap)*snap_progress).as_matrix()@W
  elbowgoal=np.radians((1-up)*58+up*95-65*push)
  def arm(x):
   d.qpos[:]=q;d.qpos[ra]=x;mujoco.mj_kinematics(m,d);Wr=d.xmat[m.body('R_Wrist').id].reshape(3,3);e=d.xpos[m.body('R_Elbow').id];c=d.xpos[m.body('R_Wrist').id]+Wr@offset
   return np.r_[35*(c-center),2.5*Rot.from_matrix(W@Wr.T).as_rotvec(),2*(elbow_bend(m,d)-elbowgoal),8*max(0,e[2]-center[2]+.06),.04*(x-previous[ra])]
  lo=np.maximum(bounds[:,0]+1e-5,previous[ra]-12/120);hi=np.minimum(bounds[:,1]-1e-5,previous[ra]+12/120)
  q[ra]=least_squares(arm,np.clip(previous[ra],lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=40).x
  left=(1-up)*L0+up*(center+.20*side-WL@offset)+.13*side*smooth((t-release+.12)/.35);q,_=pose_ik(m,q,'L',left,WL,previous[la],iterations=15);q=head_ik(m,q,RIM_CENTER,previous[hd]);d.qpos[:]=q;mujoco.mj_forward(m,d);c,actualW,f=geometry(m,d,offset)
  rows.append(dict(time_s=t,raising=up,lower_extension=ext,arm_push=push,snap=snap_progress,ball_goal_x=center[0],ball_goal_y=center[1],ball_goal_z=center[2],fk_ball_error_mm=1000*np.linalg.norm(c-center),elbow_deg=np.degrees(elbow_bend(m,d)),orientation_error_deg=np.degrees(Rot.from_matrix(W@actualW.T).magnitude())))
  samples.append(q.copy());previous=q.copy()
 np.savez_compressed(path,time_s=times,qpos=samples,offset=offset,release_time_s=release,source='M6-2 synthetic head-relative shooting trajectory; not ground truth',setpoint=setpoint,raise_s=raise_s,push_s=push_s,snap_deg=snap,push_start=push_start)
 write_csv(path.with_suffix('.csv'),rows);return path

class VelocityBridge:
 """Local additive desired-velocity torque; original PD/COM source untouched."""
 def __init__(self,base,ref,amount):self.base=base;self.ref=ref;self.amount=amount
 def __getattr__(self,n):return getattr(self.base,n)
 @property
 def goal(self):return self.base.goal
 @goal.setter
 def goal(self,q):self.base.goal=q
 def control(self,d):
  self.base.control(d)
  # All actuated hinges are in qpos[7:], qvel[6:]. Ball has no actuator.
  velocity=self.ref.spline(np.clip(d.time,0,self.ref.duration),1)
  kd=self.base.controller.pd.kd if hasattr(self.base.controller.pd,'kd') else np.full(len(d.ctrl),30*self.base.controller.gain_scale)
  # Broad upper-body mask avoids destabilizing ankle COM feedback.
  mask=np.array([any(b in self.base.model.joint(i+1).name for b in ('Shoulder','Elbow','Wrist','Torso','Spine','Chest')) for i in range(len(d.ctrl))])
  extra=self.amount*np.asarray(kd)*velocity*mask
  d.ctrl[:]=np.clip(d.ctrl+extra,-self.base.controller.limit,self.base.controller.limit)

class ShootingEnv(core.WeldShootingEnv):
 def __init__(self,*args,velocity_ff=0.,**kwargs):self.velocity_ff=velocity_ff;super().__init__(*args,**kwargs)
 def reset(self,**kwargs):
  result=super().reset(**kwargs)
  if self.velocity_ff:self.bridge=VelocityBridge(self.bridge,self.reference,self.velocity_ff)
  return result

def run_one(name,ref,velocity_ff=0.,release=None):
 e=ShootingEnv(reference=ref,record=True,velocity_ff=velocity_ff);e.reset(seed=7)
 if release is not None:e.grip.release_time=float(release)
 kin=[]
 while True:
  _,_,t,u,_=e.step(np.zeros(e.action_space.shape));d=e.d;row=dict(time_s=float(d.time))
  for n in ['R_Shoulder','R_Elbow','R_Wrist','Pelvis','Head']:
   b=e.m.body(n).id
   for k,v in zip('xyz',d.xpos[b]):row[n+'_'+k]=v
   for k,v in enumerate(d.xquat[b]):row[n+f'_quat{k}']=v
  W=d.xmat[e.wrist].reshape(3,3);finger=unit(d.xpos[e.m.body('R_Middle1').id]-d.xpos[e.wrist]);normal=-W[:,1]
  for k,v in zip('xyz',finger):row['finger_'+k]=v
  for k,v in zip('xyz',normal):row['palm_'+k]=v
  kin.append(row)
  if t or u:break
 p=OUT/'runs'/name;z=e.save(p);write_csv(p/'kinematics.csv',kin);r=z['release']
 if r:
  pos=np.array(r['position_m']);v=np.array(r['velocity_world_m_s']);f=unit(np.r_[(RIM_CENTER-pos)[:2],0]);side=unit(np.cross([0,0,1],f));horizontal=float(v@f);vertical=float(v[2]);dist=float(np.linalg.norm((RIM_CENTER-pos)[:2]));height=float(RIM_CENTER[2]-pos[2]);g=9.81;minspeed=np.sqrt(g*(height+np.sqrt(dist**2+height**2)))
  z.update(release_speed_m_s=float(np.linalg.norm(v)),forward_m_s=horizontal,lateral_m_s=float(v@side),vertical_m_s=vertical,release_angle_deg=float(np.degrees(np.arctan2(vertical,np.linalg.norm(v[:2])))),minimum_ballistic_speed_m_s=float(minspeed),release_horizontal_distance_m=dist,height_difference_m=height,above_forward=bool(horizontal>0 and vertical>0),max_ball_height_m=max(x['ball_z'] for x in e.rows),minimum_rim_distance_m=min(np.linalg.norm(np.array([x['ball_x'],x['ball_y'],x['ball_z']])-RIM_CENTER) for x in e.rows if not x['holding']))
  target=e.reference.spline(np.array(e.times),1);actual=np.array([[j['qvel_rad_s'] for j in e.joint_log[i:i+30]] for i in range(0,len(e.joint_log),30)]);z['selected_velocity_rmse_rad_s']=float(np.sqrt(np.mean((target[:,e.ids-7]-actual)**2)))
 z['max_torque_saturation']=max(x['torque_saturation'] for x in e.rows);z['velocity_ff']=velocity_ff;z['reference']=str(ref);z['kinematic_readiness_not_certified']=True
 (p/'summary.json').write_text(json.dumps(z,indent=2));e.close();print(name,{k:z.get(k) for k in ['release_speed_m_s','forward_m_s','vertical_m_s','body_tracking_rmse_rad','falls','warnings','score']},flush=True);return z

def experiments():
 audit(True);diagnose();results={}
 results['A_old']=run_one('A_old',OLD/'reference_shoot.npz')
 params=[('B_pose',1.6,.75,0,1.2),('C_lower',1.1,.75,0,.5),('D_arm',1.1,.26,0,.5),('E_snap',1.1,.26,35,.5)]
 for name,rs,ps,snap,ls in params:results[name]=run_one(name,make(name,rs,ps,snap,lower_s=ls))
 for snap in [-35,15,35,65]:
  name=f'wrist_{snap:+g}';results[name]=run_one(name,make(name,1.1,.26,snap,lower_s=.5))
 # Timing scan is a real rerun, not an injected ballistic velocity.
 for fraction in [.35,.5,.65,.8]:
  name=f'F_timing_{fraction:g}';results[name]=run_one(name,make(name,.9,.22,35,release_fraction=fraction,lower_s=.45))
 for ff in [.35,.7]:
  name=f'controller_ff_{ff:g}';results[name]=run_one(name,make(name,.9,.22,35,release_fraction=.65,lower_s=.45),ff)
 (OUT/'comparison.json').write_text(json.dumps(results,indent=2));audit();return results

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['experiment','audit']);a=p.parse_args();experiments() if a.command=='experiment' else audit()
