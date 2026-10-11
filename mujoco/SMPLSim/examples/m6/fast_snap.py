"""New synchronized knee/trunk/arm extension and reversed cock/downward snap."""
import json,hashlib
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from examples.m6.core import ROOT,MODELS,WeldShootingEnv,Reference,generate_reference,ids,smooth,RIM_CENTER,lower_path
from examples.m5_bimanual_body_ik import calibration,trunk_axis,extension_angle,head_ik,unit
from examples.m5_coordinated_reference import elbow_bend
from examples.m5_bimanual_reorientation import pose_ik
from examples.m3_control import write_csv
OUT=ROOT/'outputs/m6_fast_snap'

def protect():
 path=OUT/'preservation_before.json'
 if path.exists():return
 files=[p for p in ROOT.rglob('*') if p.is_file() and not any(x in p.parts for x in ('.git','__pycache__','m6_fast_snap')) and p.name!='.DS_Store']
 path.write_text(json.dumps({str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))

def make(duration=.75,snap=True):
 path=OUT/f'reference_{duration:g}s_{"snap" if snap else "raise"}.npz'
 if path.exists():return path
 old=Reference(generate_reference('shoot'));m=mujoco.MjModel.from_xml_path(str(MODELS/'body_01.xml'));base=np.load(ROOT/'outputs/m5_1_motion_redesign/B_face/states.npz')['qpos'][0,:160].copy();leg,stance,_=lower_path(m,base,16);start=stance[-1].copy();d=mujoco.MjData(m);d.qpos[:]=start;mujoco.mj_forward(m,d);cal=calibration(m,d);w=m.body('R_Wrist').id;W0=d.xmat[w].reshape(3,3).copy();p0=d.xpos[w]+W0@old.offset;L0=d.xpos[m.body('L_Wrist').id].copy();WL0=d.xmat[m.body('L_Wrist').id].reshape(3,3).copy();fore=d.xmat[m.body('R_Elbow').id].reshape(3,3).copy();forward=unit(np.r_[(RIM_CENTER-p0)[:2],0]);side=unit(np.cross([0,0,1],forward));finger0=unit(d.xpos[m.body('R_Middle1').id]-d.xpos[w]);cockR=Rotation.from_rotvec(side*np.radians(35)).as_matrix()@W0;finger_cock=cockR@(W0.T@finger0);snap_axis=unit(np.cross(finger_cock,[0,0,-1.]));
 # Axis sign is determined by actual FK finger direction toward world down.
 tr=ids(m,[f'{b}_{a}' for b in ('Torso','Spine','Chest') for a in 'xyz']);ra=ids(m,[f'R_{b}_{a}' for b in ('Shoulder','Elbow','Wrist') for a in 'xyz']);la=ids(m,[f'L_{b}_{a}' for b in ('Shoulder','Elbow','Wrist') for a in 'xyz']);head=ids(m,[f'{b}_{a}' for b in ('Neck','Head') for a in 'xyz']);bounds=np.array([m.jnt_range[m.jnt_qposadr==a][0] for a in ra]);tb=np.array([m.jnt_range[m.jnt_qposadr==a][0] for a in tr]);previous=start.copy();samples=[];rows=[];release=.73*duration if snap else .68*duration;total=release+2.8;times=np.linspace(0,total,round(total*60)+1)
 axis0=cal['trunk_axis'];tilt0=np.arccos(np.clip(axis0[2],-1,1));angle=max(0,tilt0-np.radians(9));goalaxis=unit(np.r_[axis0[:2],0])*np.sin(angle)+[0,0,np.cos(angle)]
 for t in times:
  rise=smooth((t-.06*duration)/(.62*duration));extend=smooth((t-.08*duration)/(.58*duration));pulse=smooth((t-.55*duration)/(.24*duration)) if snap else 0
  q=start.copy();q[leg]=(1-extend)*start[leg]+extend*base[leg];q[:3]=(1-extend)*start[:3]+extend*base[:3];desired_axis=unit((1-extend)*axis0+extend*goalaxis)
  def torso(x):
   d.qpos[:]=q;d.qpos[tr]=x;mujoco.mj_kinematics(m,d);return np.r_[6*(trunk_axis(m,d)-desired_axis),.03*(x-previous[tr]),.03*(x-start[tr])]
  lo=np.maximum(tb[:,0]+1e-5,start[tr]-.5);hi=np.minimum(tb[:,1]-1e-5,start[tr]+.5);fit=least_squares(torso,np.clip(previous[tr],lo,hi),bounds=(lo,hi),max_nfev=35);q[tr]=fit.x
  # Counter-direction to previous M6 (-35deg): +35deg while rising.
  cock=Rotation.from_rotvec(side*np.radians(35)*rise).as_matrix()@W0;W=Rotation.from_rotvec(snap_axis*np.radians(65)*pulse).as_matrix()@cock
  center=p0+forward*(.08*rise+.20*pulse)+[0,0,.54*rise+.10*pulse]
  target_elbow=np.radians((1-rise)*np.degrees(elbow_bend(m,probe(m,start)))+rise*85-45*pulse)
  def arm(x):
   d.qpos[:]=q;d.qpos[ra]=x;mujoco.mj_kinematics(m,d);wr=d.xmat[w].reshape(3,3);return np.r_[30*(d.xpos[w]+wr@old.offset-center),2*Rotation.from_matrix(W@wr.T).as_rotvec(),.7*(elbow_bend(m,d)-target_elbow),.1*(extension_angle(m,d,cal)-np.radians(18*rise-40*pulse)),.02*(x-previous[ra])]
  fit=least_squares(arm,np.clip(previous[ra],bounds[:,0]+1e-5,bounds[:,1]-1e-5),bounds=(bounds[:,0]+1e-5,bounds[:,1]-1e-5),max_nfev=45);q[ra]=fit.x
  leftgoal=(1-rise)*L0+rise*(center+side*.22-WL0@old.offset);leftgoal+=side*.15*smooth((t-release)/.25);q,_=pose_ik(m,q,'L',leftgoal,WL0,previous[la],iterations=20);q=head_ik(m,q,RIM_CENTER,previous[head]);d.qpos[:]=q;mujoco.mj_forward(m,d)
  rows.append(dict(time_s=t,rise_progress=rise,extension_progress=extend,snap_progress=pulse,trunk_tilt_deg=np.degrees(np.arccos(np.clip(trunk_axis(m,d)[2],-1,1))),elbow_bend_deg=np.degrees(elbow_bend(m,d)),wrist_relative_proxy_deg=np.degrees(extension_angle(m,d,cal)),fk_ball_error_mm=1000*np.linalg.norm(d.xpos[w]+d.xmat[w].reshape(3,3)@old.offset-center),finger_z=unit(d.xpos[m.body('R_Middle1').id]-d.xpos[w])[2]))
  samples.append(q.copy());previous=q.copy()
 np.savez_compressed(path,time_s=times,qpos=samples,offset=old.offset,release_time_s=release,kind='reversed_raise_downward_snap',duration=duration,snap_axis_world=snap_axis,source='new synthetic coordinated target; no injected ball velocity');write_csv(path.with_suffix('.csv'),rows);return path

def probe(m,q):
 d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d);return d

def run():
 OUT.mkdir(parents=True,exist_ok=True);protect();conditions=[('baseline',None),('reverse_raise',make(.75,False)),('reverse_snap',make(.75,True)),('reverse_snap_fast',make(.5,True))];rows=[]
 for name,path in conditions:
  e=WeldShootingEnv(reference=path,record=True);e.reset(seed=7)
  while True:
   _,_,done,end,_=e.step(np.zeros(e.action_space.shape))
   if done or end:break
  z=e.save(OUT/name);ev=z['release'];rows.append(dict(condition=name,reference=str(path) if path else 'original',ball_rise_mm=z['ball_rise_m']*1000,release_speed_m_s=np.linalg.norm(ev['velocity_world_m_s']) if ev else 0,release_velocity=ev['velocity_world_m_s'] if ev else None,body_rmse_rad=z['body_tracking_rmse_rad'],feet_rate=z['both_feet_contact_rate'],margin_mm=z['min_com_margin_m']*1000,falls=z['falls'],warnings=z['warnings'],score=z['score']));e.close();print(name,rows[-1],flush=True)
 (OUT/'comparison.json').write_text(json.dumps(rows,indent=2));write_csv(OUT/'comparison.csv',rows)
 saved=json.loads((OUT/'preservation_before.json').read_text());changed=[p for p,h in saved.items() if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];(OUT/'preservation_audit.json').write_text(json.dumps(dict(count=len(saved),changed=changed),indent=2));assert not changed
if __name__=='__main__':run()
