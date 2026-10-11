"""Original holding -> upright hand -> forearm-axis roll -> palm towards hoop."""
import json,hashlib
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from examples.m6.core import ROOT,MODELS,Reference,generate_reference,WeldShootingEnv,ids,smooth,lower_path,RIM_CENTER
from examples.m5_bimanual_body_ik import calibration,trunk_axis,head_ik,unit,forearm_swing_twist
from examples.m5_bimanual_reorientation import pose_ik
from examples.m5_coordinated_reference import elbow_bend
from examples.m3_control import write_csv
OUT=ROOT/'outputs/m6_palm_to_hoop_v3'

def blend(a,b,u):return a@Rotation.from_rotvec(float(u)*Rotation.from_matrix(a.T@b).as_rotvec()).as_matrix()
def basis(normal,finger):
 n=unit(normal);f=unit(finger-n*np.dot(finger,n));return np.column_stack([unit(np.cross(n,f)),n,f])
def twist_angle(axis,a,b):
 axis=unit(axis);a=unit(a-axis*np.dot(a,axis));b=unit(b-axis*np.dot(b,axis));return float(np.arctan2(axis@np.cross(a,b),np.clip(a@b,-1,1)))

def make(duration=1.1):
 path=OUT/f'reference_{duration:g}s.npz'
 if path.exists():return path
 m=mujoco.MjModel.from_xml_path(str(MODELS/'body_01.xml'));old=Reference(generate_reference('shoot'));base=np.load(ROOT/'outputs/m5_1_motion_redesign/B_face/states.npz')['qpos'][0,:160].copy();leg,stance,_=lower_path(m,base,25);start=stance[-1].copy();d=mujoco.MjData(m);d.qpos[:]=start;mujoco.mj_forward(m,d);w=m.body('R_Wrist').id;W0=d.xmat[w].reshape(3,3).copy();finger0=unit(d.xpos[m.body('R_Middle1').id]-d.xpos[w]);normal0=-W0[:,1];local=basis(W0.T@normal0,W0.T@finger0);normal_flat=np.r_[normal0[:2],0.];p0=d.xpos[w]+W0@old.offset;forward=unit(np.r_[(RIM_CENTER-p0)[:2],0.]);side=unit(np.cross([0,0,1],forward));normal_flat=unit(normal_flat) if np.linalg.norm(normal_flat)>.1 else side
 upright=basis(normal_flat,[0,0,1])@local.T;center_end=p0+.30*forward+[0,0,.58];palm_goal=unit(RIM_CENTER-center_end);final=basis(palm_goal,[0,0,1])@local.T
 cal=calibration(m,d);axis0=trunk_axis(m,d);goalaxis=unit(np.r_[axis0[:2],0])*np.sin(max(0,np.arccos(np.clip(axis0[2],-1,1))-np.radians(9)))+[0,0,np.cos(max(0,np.arccos(np.clip(axis0[2],-1,1))-np.radians(9)))];tr=ids(m,[f'{b}_{a}' for b in ('Torso','Spine','Chest') for a in 'xyz']);ra=ids(m,[f'R_{b}_{a}' for b in ('Shoulder','Elbow','Wrist') for a in 'xyz']);la=ids(m,[f'L_{b}_{a}' for b in ('Shoulder','Elbow','Wrist') for a in 'xyz']);head=ids(m,[f'{b}_{a}' for b in ('Neck','Head') for a in 'xyz']);ab=np.array([m.jnt_range[m.jnt_qposadr==a][0] for a in ra]);tb=np.array([m.jnt_range[m.jnt_qposadr==a][0] for a in tr]);L0=d.xpos[m.body('L_Wrist').id].copy();WL0=d.xmat[m.body('L_Wrist').id].reshape(3,3).copy();previous=start.copy();samples=[];rows=[];release=duration+.6;total=release+2.8;times=np.linspace(0,total,round(total*60)+1)
 lo=ab[:,0]+1e-5;hi=ab[:,1]-1e-5;lo[6:]=np.maximum(lo[6:],start[ra[6:]]-10);hi[6:]=np.minimum(hi[6:],start[ra[6:]]+10)
 for t in times:
  rise=smooth((t-.06*duration)/(.80*duration));ext=smooth((t-.10*duration)/(.78*duration));stand=smooth((t-.02*duration)/(.45*duration));roll=smooth((t-.28*duration)/(.50*duration));finish=smooth((t-.58*duration)/(.42*duration));q=start.copy();q[leg]=(1-ext)*start[leg]+ext*base[leg];q[:3]=(1-ext)*start[:3]+ext*base[:3];targetaxis=unit((1-ext)*axis0+ext*goalaxis)
  def trunk(x):
   d.qpos[:]=q;d.qpos[tr]=x;mujoco.mj_kinematics(m,d);return np.r_[6*(trunk_axis(m,d)-targetaxis),.03*(x-previous[tr]),.04*(x-start[tr])]
  tl=np.maximum(tb[:,0]+1e-5,start[tr]-.5);th=np.minimum(tb[:,1]-1e-5,start[tr]+.5);q[tr]=least_squares(trunk,np.clip(previous[tr],tl,th),bounds=(tl,th),max_nfev=35).x
  # Measure the forearm axis by FK, never infer flexion from hinge names.
  d.qpos[:]=previous;mujoco.mj_forward(m,d);arm_axis=unit(d.xpos[w]-d.xpos[m.body('R_Elbow').id]);Rstand=blend(W0,upright,stand);normal_stand=-Rstand[:,1];angle=twist_angle(arm_axis,normal_stand,palm_goal);rolled=Rotation.from_rotvec(arm_axis*angle*roll).as_matrix()@Rstand;W=blend(rolled,final,finish)
  center=(1-rise)*p0+rise*center_end;elbow_target=np.radians((1-rise)*52+rise*85-50*finish)
  def arm(x):
   d.qpos[:]=q;d.qpos[ra]=x;mujoco.mj_kinematics(m,d);WR=d.xmat[w].reshape(3,3);return np.r_[30*(d.xpos[w]+WR@old.offset-center),4*Rotation.from_matrix(W@WR.T).as_rotvec(),12*finish*(-WR[:,1]-unit(RIM_CENTER-d.xpos[w])),1.5*(elbow_bend(m,d)-elbow_target),.06*(x-previous[ra])]
  # Bound sample-to-sample hinge motion; avoid fast IK branch changes.
  dt=times[1]-times[0];l=np.maximum(lo,previous[ra]-8*dt);h=np.minimum(hi,previous[ra]+8*dt);q[ra]=least_squares(arm,np.clip(previous[ra],l+1e-8,h-1e-8),bounds=(l,h),max_nfev=45).x
  leftgoal=(1-rise)*L0+rise*(center+side*.22-WL0@old.offset);q,_=pose_ik(m,q,'L',leftgoal,WL0,previous[la],iterations=20);q=head_ik(m,q,RIM_CENTER,previous[head]);d.qpos[:]=q;mujoco.mj_forward(m,d);N=-d.xmat[w].reshape(3,3)[:,1];goal=unit(RIM_CENTER-d.xpos[w]);rows.append(dict(time_s=t,rise=rise,extension=ext,upright=stand,forearm_roll=roll,finish=finish,palm_aim_error_deg=np.degrees(np.arccos(np.clip(N@goal,-1,1))),elbow_bend_deg=np.degrees(elbow_bend(m,d)),trunk_tilt_deg=np.degrees(np.arccos(np.clip(trunk_axis(m,d)[2],-1,1))),fk_ball_error_mm=1000*np.linalg.norm(d.xpos[w]+d.xmat[w].reshape(3,3)@old.offset-center)));samples.append(q.copy());previous=q.copy()
 np.savez_compressed(path,time_s=times,qpos=samples,offset=old.offset,release_time_s=release,source='original holding, upright and forearm roll towards actual scene hoop; palm normal is wrist frame proxy',duration=duration);write_csv(path.with_suffix('.csv'),rows);return path

def run():
 OUT.mkdir(parents=True,exist_ok=True);manifest=OUT/'preservation_before.json'
 if not manifest.exists():manifest.write_text(json.dumps({str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*') if p.is_file() and not any(n in p.parts for n in ('.git','__pycache__','m6_palm_to_hoop_v3')) and p.name!='.DS_Store'},indent=2))
 for duration in [1.1,.75]:
  path=make(duration);e=WeldShootingEnv(reference=path,record=True);e.reset(seed=7);cal=calibration(e.m,e.d);extra=[]
  while True:
   _,_,t,u,_=e.step(np.zeros(e.action_space.shape));N=-e.d.xmat[e.wrist].reshape(3,3)[:,1];goal=unit(RIM_CENTER-e.d.xpos[e.wrist]);swing,twist=forearm_swing_twist(e.m,e.d,cal);extra.append(dict(time_s=float(e.d.time),palm_aim_error_deg=float(np.degrees(np.arccos(np.clip(N@goal,-1,1)))),palm_nx=N[0],palm_ny=N[1],palm_nz=N[2],forearm_swing_deg=swing,forearm_twist_deg=twist))
   if t or u:break
  z=e.save(OUT/f'run_{duration:g}s');write_csv(OUT/f'run_{duration:g}s'/'palm.csv',extra);idx=min(range(len(extra)),key=lambda i:abs(extra[i]['time_s']-z['release']['actual_time_s']));z['palm_error_at_release_deg']=extra[idx]['palm_aim_error_deg'];(OUT/f'run_{duration:g}s'/'summary.json').write_text(json.dumps(z,indent=2));e.close();print(duration,z,flush=True)
 saved=json.loads(manifest.read_text());changed=[p for p,h in saved.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];(OUT/'preservation_audit.json').write_text(json.dumps(dict(protected=len(saved),changed=changed),indent=2));assert not changed
if __name__=='__main__':run()
