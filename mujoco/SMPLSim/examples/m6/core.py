"""Independent M6: solver-held ball, continuous reference, residual joint targets."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/m6_weld_shooting'
import numpy as np,mujoco,gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
from smpl_sim.physics.physical_holding import build_holding_xml,WeldBallRelease
from smpl_sim.physics.basketball import HumanoidControllerBridge,RIM_CENTER,RIM_INNER_RADIUS,RIM_CENTERLINE_RADIUS,RIM_TUBE_RADIUS,BALL_RADIUS
from examples.m5_residual_ppo_env import MODELS,SOURCE
from examples.m5_bimanual_reorientation import pose_ik
from examples.m5_bimanual_body_ik import calibration,head_ik,trunk_axis,extension_angle
from examples.m5_coordinated_reference import elbow_bend
from examples.m5_video_reference import lower_path,chest_ik,knee_bend
from examples.m4_physical_holding import support_margin_at
from examples.m3_control import contacts
from smpl_sim.motion.physics_metrics import foot_contact_slip


def smooth(u):
 u=np.clip(u,0,1);return u*u*u*(10+u*(-15+6*u))
def ids(m,names):return np.array([int(m.joint(n).qposadr[0]) for n in names])

class ScoringDetector:
 """Entire sphere crosses a conservative polygonal aperture downwards once."""
 def __init__(self,center=RIM_CENTER):
  self.center=np.asarray(center,float);self.clearance=RIM_CENTERLINE_RADIUS*np.cos(np.pi/64)-RIM_TUBE_RADIUS-BALL_RADIUS;self.reset()
 def reset(self):self.previous=None;self.armed=False;self.entered=False;self.scored=False
 def update(self,p,released):
  p=np.asarray(p,float);old=self.previous;self.previous=p.copy()
  if not released:self.armed=False;self.entered=False;return False
  if self.scored:return False
  z=self.center[2];top=z+BALL_RADIUS;bottom=z-BALL_RADIUS
  if p[2]>=top:self.armed=True
  if old is None:return False
  aperture=self.clearance+BALL_RADIUS
  if old[2]>=top:self.armed=True
  descending=p[2]<old[2]
  if self.armed and descending and old[2]>=top>p[2]:
   cross=old+(p-old)*((old[2]-top)/(old[2]-p[2]));self.entered=np.linalg.norm(cross[:2]-self.center[:2])<=aperture
  if self.entered:
   # Sphere section at rim plane; unlike a full-radius cylinder this permits
   # oblique exits. Physics is sampled at 450Hz plus interpolated sections.
   points=list(np.linspace(old,p,9))
   for plane in (top,z,bottom):
    if descending and old[2]>=plane>=p[2]:points.append(old+(p-old)*((old[2]-plane)/(old[2]-p[2])))
   invalid=False
   for point in points:
    dz=point[2]-z
    if abs(dz)<=BALL_RADIUS:
     section=np.sqrt(max(0,BALL_RADIUS**2-dz**2))
     invalid|=np.linalg.norm(point[:2]-self.center[:2])+section>aperture+1e-10
   if not descending or invalid:self.entered=False;self.armed=False
   elif old[2]>=bottom>p[2]:self.scored=True;return True
  return False

class Reference:
 def __init__(self,path):
  z=np.load(path);self.times=z['time_s'];self.q=z['qpos'];self.release=float(z['release_time_s']);self.offset=z['offset'];self.spline=CubicSpline(self.times,self.q[:,7:],axis=0,bc_type='clamped');self.duration=float(self.times[-1])
  if self.q.shape!=(len(self.times),160) or not np.all(np.diff(self.times)>0) or not np.isfinite(self.q).all() or not 0<=self.release<self.duration or self.offset.shape!=(3,):raise ValueError('Invalid M6 reference')
  if not np.allclose(np.linalg.norm(self.q[:,3:7],axis=1),1,atol=1e-6):raise ValueError('Reference root quaternion must be unit wxyz')
 def target(self,t):
  q=self.q[0].copy();q[7:]=self.spline(np.clip(t,0,self.duration));return q

class Bridge(HumanoidControllerBridge):
 def target(self,t):return self.goal.copy() if hasattr(self,'goal') else super().target(t)


def generate_reference(mode='shoot',duration=2.4):
 """Offline sequential FK/IK. Root translation is a planner/initial-state only."""
 OUT.mkdir(parents=True,exist_ok=True)
 destination=OUT/('reference_'+mode+(f'_{duration:g}s' if duration!=2.4 else '')+'.npz')
 if destination.exists():return destination
 original=mujoco.MjModel.from_xml_path(str(MODELS/'body_01.xml'));base=np.load(ROOT/'outputs/m5_1_motion_redesign/B_face/states.npz')['qpos'][0,:original.nq].copy()
 r=dict(np.load(ROOT/'outputs/m5_1_coordinated_reference/reference/reference.npz'));offset=r['rel0'];p=mujoco.MjData(original);p.qpos[:]=base;mujoco.mj_forward(original,p)
 lower,lowerq,lowerrows=lower_path(original,base,12 if mode=='shoot' else 0)
 start=lowerq[-1].copy();p.qpos[:]=start;mujoco.mj_forward(original,p);cal=calibration(original,p);W0=p.xmat[original.body('R_Wrist').id].reshape(3,3).copy();P0=p.xpos[original.body('R_Wrist').id]+W0@offset
 L0=p.xpos[original.body('L_Wrist').id].copy();WL0=p.xmat[original.body('L_Wrist').id].reshape(3,3).copy();forward=r['forward'];side=r['side'];armnames=[f'R_{b}_{a}' for b in ('Shoulder','Elbow','Wrist') for a in 'xyz'];ra=ids(original,armnames);la=ids(original,[n.replace('R_','L_',1) for n in armnames]);trunk=ids(original,[f'{b}_{a}' for b in ('Torso','Spine','Chest') for a in 'xyz']);head=ids(original,[f'{b}_{a}' for b in ('Neck','Head') for a in 'xyz'])
 # Wrist orientation uses one fixed SO(3) axis; no Euler branch conversion.
 finalR=Rotation.from_rotvec(side*np.radians(-35 if mode=='shoot' else 25 if mode=='D' else 0)).as_matrix()@W0
 release=duration*.78 if mode=='shoot' else 1.45
 total=max(duration+.7,release+2.8);times=np.linspace(0,total,round(total*30)+1);samples=[];rows=[];previous=start.copy()
 for t in times:
  q=start.copy();u=smooth((t-min(.2,.08*duration))/(duration*.68)) if mode=='shoot' else smooth((t-.35)/1.6)
  extension=smooth((t-.4)/(duration*.64)) if mode=='shoot' else 0
  q[lower]=(1-extension)*start[lower]+extension*base[lower]
  # Virtual free-root trajectory is used for FK only; never a root actuator.
  q[:3]=(1-extension)*start[:3]+extension*base[:3]
  q[trunk]=chest_ik(original,q,cal,7*extension if mode=='shoot' else 0,previous[trunk])
  W=Rotation.from_rotvec(side*np.radians(-35 if mode=='shoot' else 25 if mode=='D' else 0)*u).as_matrix()@W0
  delta=np.array([0.,0.,.30*u]) if mode in ('B','D') else .12*u*forward+[0,0,.30*u] if mode=='C' else np.zeros(3)
  if mode=='shoot':
   raising=.48*u;push=smooth((t-duration*.58)/(duration*.28));delta=(.10*u+.17*push)*forward+[0,0,raising+.12*push]
   # Extension then wrist flexion follow-through is continuous, hypothesis not GT.
   W=Rotation.from_rotvec(side*np.radians(-35*u+55*push)).as_matrix()@W0
  center=P0+delta;q,fit=pose_ik(original,q,'R',center-W@offset,W,previous[ra],iterations=30)
  lprogress=u if mode=='shoot' else 0;left_center=center+side*.21-WL0@r['left_offset'];left_goal=(1-lprogress)*L0+lprogress*left_center
  if mode=='shoot':left_goal+=side*.12*smooth((t-release+.1)/.45)
  q,_=pose_ik(original,q,'L',left_goal,WL0,previous[la],iterations=20);q=head_ik(original,q,RIM_CENTER,previous[head]);p.qpos[:]=q;mujoco.mj_forward(original,p)
  rows.append(dict(time_s=float(t),rise_progress=float(u),extension_progress=float(extension),R_knee_deg=np.degrees(knee_bend(original,p,'R')),L_knee_deg=np.degrees(knee_bend(original,p,'L')),trunk_tilt_deg=np.degrees(np.arccos(np.clip(trunk_axis(original,p)[2],-1,1))),ik_position_error_m=float(np.linalg.norm(p.xpos[original.body('R_Wrist').id]+p.xmat[original.body('R_Wrist').id].reshape(3,3)@offset-center))))
  samples.append(q.copy());previous=q.copy()
 np.savez_compressed(destination,time_s=times,qpos=samples,release_time_s=release,offset=offset,kind=mode,source='synthetic coordinated FK reference; not video ground truth')
 from examples.m3_control import write_csv
 write_csv(OUT/('reference_'+mode+'_fk.csv'),rows)
 return destination

class WeldShootingEnv(gym.Env):
 metadata={'render_modes':['human','rgb_array'],'render_fps':30}
 def __init__(self,stage=2,mode='shoot',reference=None,render_mode=None,record=False):
  super().__init__()
  if stage not in (0,1,2):raise ValueError('Stages 0..2 implemented; 3/4 require explicit timing/robustness extensions')
  self.stage=stage;self.mode=mode;self.render_mode=render_mode;self.record=record;self.viewer=None;self.renderer=None
  self.reference=Reference(reference or generate_reference(mode));self.scene=OUT/('scene_'+mode+'.xml')
  if not self.scene.exists():build_holding_xml(MODELS/'body_01.xml',self.scene,offset=self.reference.offset,weld=True)
  self.m=mujoco.MjModel.from_xml_path(str(self.scene));self.m.opt.timestep=1/450;self.d=mujoco.MjData(self.m);self.substeps=15;self.control_dt=1/30
  self.names=[f'{s}_{b}_{a}' for s in ('R','L') for b in ('Hip','Knee') for a in 'xyz']+[f'{b}_{a}' for b in ('Torso','Spine','Chest') for a in 'xyz']+[f'R_{b}_{a}' for b in ('Shoulder','Elbow','Wrist') for a in 'xyz'];self.ids=ids(self.m,self.names);self.dofs=np.array([int(self.m.joint(n).dofadr[0]) for n in self.names]);self.bounds=np.array([self.m.joint(n).range for n in self.names]);self.action_space=spaces.Box(-1,1,(len(self.names),),np.float32)
  self.qa=int(self.m.joint('basketball_free').qposadr[0]);self.va=int(self.m.joint('basketball_free').dofadr[0]);self.wrist=self.m.body('R_Wrist').id;self.detector=ScoringDetector();self.reset();self.observation_space=spaces.Box(-np.inf,np.inf,self.observation().shape,np.float32)
 def reset(self,*,seed=None,options=None):
  super().reset(seed=seed);self.action_space.seed(seed);mujoco.mj_resetData(self.m,self.d);self.d.qpos[:160]=self.reference.q[0];mujoco.mj_forward(self.m,self.d)
  self.bridge=Bridge(MODELS/'body_01.xml',self.m,SOURCE);self.bridge.goal=self.reference.target(0)
  # Release-only hand contact remains disabled deliberately in M6: no finger acceleration.
  self.grip=WeldBallRelease(self.m,offset=self.reference.offset,release_time=self.reference.release,hand_collision='none');self.grip.initialize(self.d);self.detector.reset();self.root0=self.d.qpos[:3].copy();self.residual=np.zeros(len(self.ids));self.last_action=np.zeros(len(self.ids));self.last_qvel=self.d.qvel[self.dofs].copy();self.last_acc=np.zeros(len(self.ids));self.rows=[];self.states=[];self.times=[];self.targets=[];self.score=False;self.finished=False;self.release_event=None;self.freeflight=[];self.flight_contact=False;self.initial_ball=self.d.qpos[self.qa:self.qa+3].copy();self.goal=self.reference.target(0);self.cal=calibration(self.m,self.d);self.joint_log=[];return self.observation(),{'seed':seed,'stage':self.stage,'ball_initialization_writes':1}
 def ball_velocity(self):
  v=np.zeros(6);mujoco.mj_objectVelocity(self.m,self.d,mujoco.mjtObj.mjOBJ_BODY,self.m.body('basketball').id,v,0);return v[3:].copy(),v[:3].copy()
 def com_position(self):
  human=self.d.subtree_com[self.m.body('Pelvis').id]
  if not self.grip.released:
   mass=self.m.body_subtreemass[self.m.body('Pelvis').id];return (mass*human+.6*self.d.qpos[self.qa:self.qa+3])/(mass+.6)
  return human.copy()
 def status(self):
  feet,margin,nonfoot=contacts(self.m,self.d);tilt=np.degrees(np.arccos(np.clip(self.d.xmat[self.m.body('Pelvis').id].reshape(3,3)[2,1],-1,1)));slip,_=foot_contact_slip(self.m,self.d)
  # contacts() counts ball-floor as nonfoot: use pelvis height/tilt for falls.
  if not self.grip.released:margin=support_margin_at(self.m,self.d,self.com_position()[:2])
  return feet,float(margin),bool(self.d.qpos[2]<.65*self.root0[2] or tilt>40),float(np.sqrt((slip['L']['rms_speed']**2+slip['R']['rms_speed']**2)/2))
 def observation(self):
  v,w=self.ball_velocity();wv=np.zeros(6);mujoco.mj_objectVelocity(self.m,self.d,mujoco.mjtObj.mjOBJ_BODY,self.wrist,wv,0);feet,margin,fall,slip=self.status();com=self.com_position();time=float(self.d.time);ph=np.zeros(5);ph[min(4,0 if time<.2 else 1 if time<self.reference.release*.72 else 2 if time<self.reference.release else 3 if time<self.reference.release+.6 else 4)]=1
  result=np.r_[self.d.qpos[self.ids]/np.pi,self.d.qvel[self.dofs]/10,(self.goal[self.ids]-self.d.qpos[self.ids])/np.pi,self.d.qpos[:3]-self.root0,self.d.qpos[3:7],com-self.d.qpos[:3],margin,feet['L']/1000,feet['R']/1000,self.d.xpos[self.wrist],self.d.xquat[self.wrist],wv/10,self.d.qpos[self.qa:self.qa+3],self.d.qpos[self.qa+3:self.qa+7],v/10,w/10,RIM_CENTER-self.d.qpos[self.qa:self.qa+3],ph,float(not self.grip.released),(self.reference.release-time)/3,self.residual/.15]
  return np.asarray(result,np.float32)
 def step(self,action):
  if self.finished:raise RuntimeError('reset required after episode end')
  a=np.asarray(action,float)
  if a.shape!=self.action_space.shape or not np.isfinite(a).all():raise ValueError('Invalid action')
  a=np.clip(a,-1,1);score_event=False;launch_reward=0.
  for _ in range(self.substeps):
   dt=self.m.opt.timestep;self.residual+=np.clip(.15*a-self.residual,-.8*dt,.8*dt);self.goal=self.reference.target(self.d.time+dt);self.goal[self.ids]=np.clip(self.goal[self.ids]+self.residual,self.bounds[:,0]+1e-5,self.bounds[:,1]-1e-5);self.bridge.goal=self.goal
   beforeq=self.d.qpos[self.qa:self.qa+7].copy();beforev=self.d.qvel[self.va:self.va+6].copy();was=self.grip.released;self.grip.update(self.d)
   if not was and self.grip.released:
    self.release_event=dict(self.grip.event);self.release_event['before_qpos']=beforeq.tolist();self.release_event['after_qpos']=self.d.qpos[self.qa:self.qa+7].tolist();self.release_event['before_qvel']=beforev.tolist();self.release_event['after_qvel']=self.d.qvel[self.va:self.va+6].tolist()
    required=(RIM_CENTER-beforeq[:3])/1.1+np.array([0,0,.5*9.81*1.1]);launch_reward=float(np.exp(-np.mean((beforev[:3]-required)**2)/16));self.release_event['ballistic_velocity_diagnostic_m_s']=required.tolist();self.release_event['velocity_alignment_reward']=launch_reward
   self.bridge.control(self.d)
   assert np.array_equal(beforeq,self.d.qpos[self.qa:self.qa+7]) and np.array_equal(beforev,self.d.qvel[self.va:self.va+6]);assert not self.d.xfrc_applied.any() and not self.d.qfrc_applied.any()
   mujoco.mj_step(self.m,self.d);mujoco.mj_forward(self.m,self.d);score_event=self.detector.update(self.d.qpos[self.qa:self.qa+3],self.grip.released) or score_event
   if self.grip.released:
    contact=any(self.m.geom('basketball_geom').id in (self.d.contact[i].geom1,self.d.contact[i].geom2) for i in range(self.d.ncon));self.flight_contact|=contact
    if not self.flight_contact:
     ev=self.release_event;s=self.d.time-ev['actual_time_s'];v=self.d.qvel[self.va:self.va+3];self.freeflight.append(float(np.linalg.norm(v-np.asarray(ev['velocity_world_m_s'])-[0,0,-9.81*s])))
  self.score|=score_event;feet,margin,fall,slip=self.status();v,w=self.ball_velocity();p=self.d.qpos[self.qa:self.qa+3];err=self.d.qpos[self.ids]-self.goal[self.ids];acc=(self.d.qvel[self.dofs]-self.last_qvel)/self.control_dt;jerk=(acc-self.last_acc)/self.control_dt;self.last_qvel=self.d.qvel[self.dofs].copy();self.last_acc=acc
  # Downward crossing prediction, only after true release; misses cannot count as scores.
  dz=p[2]-RIM_CENTER[2];disc=v[2]**2+2*9.81*dz;flight=(v[2]+np.sqrt(disc))/9.81 if disc>=0 else -1;rim_error=float(np.linalg.norm((p+v*max(0,flight)+[0,0,-.5*9.81*max(0,flight)**2])[:2]-RIM_CENTER[:2])) if flight>0 else 10.
  motion=float(np.exp(-20*np.mean(err**2)));balance=float(min(feet.values())>.1 and margin>0);ballistic=float(np.exp(-rim_error**2/2)) if self.grip.released and flight>0 else 0.;direction=float(max(0,np.dot(v[:2],(RIM_CENTER-p)[:2])/max(np.linalg.norm((RIM_CENTER-p)[:2])*np.linalg.norm(v[:2]),1e-8))) if self.grip.released else 0
  limit=np.array([self.bridge.controller.limit[self.bridge.controller.index[n]] for n in self.names]);tau=self.d.qfrc_actuator[self.dofs];sat=float(np.mean(abs(tau)>.98*limit));near=float(np.mean(np.minimum(self.d.qpos[self.ids]-self.bounds[:,0],self.bounds[:,1]-self.d.qpos[self.ids])<.05))
  reward=.5*motion+.2*balance+(2*ballistic+.1*direction+.3*launch_reward+100*score_event if self.stage==2 else 0)-.02*np.mean(a*a)-.005*np.mean((a-self.last_action)**2)-.1*sat-.05*near-.00001*min(float(np.mean(jerk**2)),1e5)-.02*float(np.linalg.norm(self.d.qpos[:2]-self.root0[:2]));self.last_action=a.copy();warnings=sum(int(x.number) for x in self.d.warning);finite=bool(np.isfinite(self.d.qpos).all() and np.isfinite(self.d.qvel).all());terminated=bool(fall or warnings or not finite);truncated=bool(self.d.time>=self.reference.duration-1e-9);self.finished=terminated or truncated
  self.rows.append(dict(time_s=float(self.d.time),R_elbow_bend_deg=np.degrees(elbow_bend(self.m,self.d)),wrist_extension_proxy_deg=np.degrees(extension_angle(self.m,self.d,self.cal)),head_aim_error_deg=np.degrees(np.arccos(np.clip(np.dot(self.d.xmat[self.m.body('Head').id].reshape(3,3)[:,2],(RIM_CENTER-self.d.xpos[self.m.body('Head').id])/np.linalg.norm(RIM_CENTER-self.d.xpos[self.m.body('Head').id])),-1,1))),holding=int(not self.grip.released),score=int(self.score),selected_rmse_rad=float(np.sqrt(np.mean(err**2))),body_rmse_rad=float(np.sqrt(np.mean((self.d.qpos[7:160]-self.goal[7:160])**2))),root_error_m=float(np.linalg.norm(self.d.qpos[:3]-self.reference.target(self.d.time)[:3])),support_margin_m=margin,both_feet=int(min(feet.values())>.1),foot_slip_m_s=slip,torque_saturation=sat,fallen=int(fall),warnings=warnings,ball_x=p[0],ball_y=p[1],ball_z=p[2],vx=v[0],vy=v[1],vz=v[2],wx=w[0],wy=w[1],wz=w[2],rim_prediction_error_m=rim_error,reward=float(reward),R_knee_deg=np.degrees(knee_bend(self.m,self.d,'R')),L_knee_deg=np.degrees(knee_bend(self.m,self.d,'L')),trunk_tilt_deg=np.degrees(np.arccos(np.clip(trunk_axis(self.m,self.d)[2],-1,1))),joint_jerk_rms=float(np.sqrt(np.mean(jerk**2))),weld_error_m=float(np.linalg.norm(p-(self.d.xpos[self.wrist]+self.d.xmat[self.wrist].reshape(3,3)@self.reference.offset))) if not self.grip.released else 0.))
  if self.record:
   self.joint_log.extend(dict(time_s=float(self.d.time),joint=n,target_rad=float(self.goal[adr]),actual_rad=float(self.d.qpos[adr]),qvel_rad_s=float(self.d.qvel[dof]),torque_Nm=float(self.d.qfrc_actuator[dof])) for n,adr,dof in zip(self.names,self.ids,self.dofs))
  if self.record:self.states.append(self.d.qpos.copy());self.times.append(float(self.d.time));self.targets.append(self.goal.copy())
  if self.render_mode=='human':self.render()
  obs=self.observation()
  if not np.isfinite(obs).all() or not np.isfinite(reward):raise FloatingPointError('Nonfinite observation/reward')
  return obs,float(reward),terminated,truncated,{'is_success':bool(self.score),'released':self.grip.released,'fallen':fall,'warning_count':warnings,'rim_error_m':rim_error,'stage':self.stage,'reward_terms':{'motion':motion,'balance':balance,'ballistic':ballistic,'direction':direction,'launch_velocity':launch_reward,'score_event':int(score_event)}}
 def save(self,path):
  from examples.m3_control import write_csv
  path=Path(path);path.mkdir(parents=True,exist_ok=True);write_csv(path/'physics.csv',self.rows)
  if self.record:
   write_csv(path/'joints.csv',self.joint_log)
  if self.record:np.savez_compressed(path/'states.npz',time_s=self.times,qpos=self.states,target_qpos=self.targets)
  summary=dict(stage=self.stage,mode=self.mode,score=self.score,release=self.release_event,duration_s=float(self.d.time),falls=sum(r['fallen'] for r in self.rows),warnings=sum(int(w.number) for w in self.d.warning),body_tracking_rmse_rad=float(np.sqrt(np.mean([r['body_rmse_rad']**2 for r in self.rows]))),both_feet_contact_rate=float(np.mean([r['both_feet'] for r in self.rows])),min_com_margin_m=min(r['support_margin_m'] for r in self.rows),foot_slip_rms_m_s=float(np.sqrt(np.mean([r['foot_slip_m_s']**2 for r in self.rows]))),max_weld_error_m=max(r['weld_error_m'] for r in self.rows),max_freeflight_velocity_residual_m_s=max(self.freeflight,default=0),freeflight_samples=len(self.freeflight),ball_rise_m=max(r['ball_z'] for r in self.rows)-self.initial_ball[2],initial_ball_position=self.initial_ball.tolist(),ball_writes_after_reset=0)
  (path/'summary.json').write_text(json.dumps(summary,indent=2));return summary
 def render(self):
  import mujoco.viewer
  if self.render_mode=='human':
   if self.viewer is None:self.viewer=mujoco.viewer.launch_passive(self.m,self.d);self.viewer.cam.lookat[:]=[1.8,0,1.6];self.viewer.cam.distance=6;self.viewer.cam.azimuth=100;self.viewer.cam.elevation=-10
   self.viewer.sync()
  else:
   if self.renderer is None:self.renderer=mujoco.Renderer(self.m,height=480,width=640)
   self.renderer.update_scene(self.d);return self.renderer.render()
 def close(self):
  if self.viewer:self.viewer.close();self.viewer=None
  if self.renderer:self.renderer.close();self.renderer=None
