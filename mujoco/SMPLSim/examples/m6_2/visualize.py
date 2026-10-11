"""Actual saved MuJoCo physics replay at 1x, phase frames and plots."""
import argparse,json,time
from pathlib import Path
from contextlib import nullcontext
import numpy as np,mujoco,mujoco.viewer
from PIL import Image,ImageDraw
from examples.m6_2.experiment import OUT,ROOT,ShootingEnv,RIM_CENTER
from examples.m4_basketball import trail

def hand_arrows(scene,e):
 w=e.d.xpos[e.wrist];f=e.d.xpos[e.m.body('R_Middle1').id]-w;f/=max(np.linalg.norm(f),1e-9)
 for direction,color in [(f,[0,1,0,1]),(-e.d.xmat[e.wrist].reshape(3,3)[:,1],[0,.6,1,1])]:
  if scene.ngeom>=scene.maxgeom:break
  g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color,np.float32));mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_ARROW,.008,w,w+.18*direction);scene.ngeom+=1

def replay(name,view='side',gui=False,repeats=1):
 p=OUT/'runs'/name;z=json.loads((p/'summary.json').read_text());s=np.load(p/'states.npz');e=ShootingEnv(reference=z['reference']);cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam);cam.lookat[:]=[.35,0,1.15];cam.distance=3.6;cam.azimuth=90 if view=='side' else 140;cam.elevation=-8;renderer=None if gui else mujoco.Renderer(e.m,height=540,width=720);frames=[];ts=s['time_s'];release=z['release']['actual_time_s']
 phase_times=[ts[0],.18,.5,release-.2,release-.04,release,release+.04,release+.25,release+.6,ts[-1]];indices=[int(np.argmin(abs(ts-t))) for t in phase_times];phaseframes={};savedframes=[]
 with mujoco.viewer.launch_passive(e.m,e.d) if gui else nullcontext(None) as viewer:
  if viewer:viewer.cam.lookat[:]=cam.lookat;viewer.cam.distance=cam.distance;viewer.cam.azimuth=cam.azimuth;viewer.cam.elevation=cam.elevation
  repeat=0
  while True:
   start=time.monotonic()
   for i,(t,q) in enumerate(zip(ts,s['qpos'])):
    if viewer and not viewer.is_running():break
    with viewer.lock() if viewer else nullcontext():
     e.d.qpos[:]=q;e.d.eq_active[:]=0;mujoco.mj_forward(e.m,e.d)
     if viewer:viewer.user_scn.ngeom=0;trail(viewer.user_scn,s['qpos'][:i+1:2,e.qa:e.qa+3]);hand_arrows(viewer.user_scn,e)
     if renderer and (i%2==0 or i in indices):
      renderer.update_scene(e.d,camera=cam);trail(renderer.scene,s['qpos'][:i+1:2,e.qa:e.qa+3]);hand_arrows(renderer.scene,e);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);draw.text((12,12),f'M6-2 {name} | {t:.2f}s | '+('WELD' if t<release else 'FREE FLIGHT'),fill='yellow')
      if i%2==0:frames.append(im.quantize(colors=256));savedframes.append(t)
      if i in indices:phaseframes[i]=im
    if viewer:viewer.sync();time.sleep(max(0,t-ts[0]-(time.monotonic()-start)))
   repeat+=1
   if not viewer or not viewer.is_running() or (repeats and repeat>=repeats):break
 if renderer:
  renderer.close();durs=np.r_[np.diff(savedframes),np.median(np.diff(savedframes))]*1000;frames[0].save(OUT/f'{name}_{view}_1x.gif',save_all=True,append_images=frames[1:],duration=durs.astype(int).tolist(),loop=0,optimize=False)
  sheet=Image.new('RGB',(720*2,540*5))
  for j,i in enumerate(indices):sheet.paste(phaseframes[i],((j%2)*720,(j//2)*540));phaseframes[i].save(OUT/f'{name}_{view}_phase_{j:02d}.png')
  sheet.save(OUT/f'{name}_{view}_phases.png')
 e.close()

def analyze():
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import csv
 from examples.m3_control import write_csv
 results=json.loads((OUT/'comparison.json').read_text());rows=[]
 for name,z in results.items():
  p=OUT/'runs'/name;r=list(csv.DictReader((p/'physics.csv').open()));j=list(csv.DictReader((p/'joints.csv').open()));s=np.load(p/'states.npz');q=s['qpos'];ts=s['time_s'];target=s['target_qpos'];vel=np.gradient(q[:,7:160],ts,axis=0);acc=np.gradient(vel,ts,axis=0);jerk=np.gradient(acc,ts,axis=0);jointrows=[]
  for k,n in enumerate(sorted(set(x['joint'] for x in j))):
   js=[x for x in j if x['joint']==n];tt=np.array([float(x['time_s']) for x in js]);actual=np.array([float(x['actual_rad']) for x in js]);goal=np.array([float(x['target_rad']) for x in js]);v=np.array([float(x['qvel_rad_s']) for x in js]);vg=np.gradient(goal,tt)
   jointrows.append(dict(joint=n,tracking_rmse_rad=float(np.sqrt(np.mean((actual-goal)**2))),velocity_rmse_rad_s=float(np.sqrt(np.mean((v-vg)**2))),max_velocity_rad_s=float(abs(v).max()),max_target_velocity_rad_s=float(abs(vg).max()),max_torque_Nm=max(abs(float(x['torque_Nm'])) for x in js)))
  write_csv(p/'joint_evaluation.csv',jointrows)
  idx=int(np.argmin(abs(ts-z['release']['actual_time_s'])));z['release_elbow_bend_deg']=float(r[idx]['R_elbow_bend_deg']);z['release_trunk_tilt_deg']=float(r[idx]['trunk_tilt_deg']);z['initial_knee_deg']=float(r[0]['R_knee_deg']);z['release_knee_deg']=float(r[idx]['R_knee_deg']);z['joint_acceleration_rms_rad_s2']=float(np.sqrt(np.mean(acc**2)));z['joint_jerk_rms_rad_s3']=float(np.sqrt(np.mean(jerk**2)));z['max_target_sample_jump_rad']=float(abs(np.diff(target[:,7:160],axis=0)).max());z['ball_height_at_release_m']=z['release']['position_m'][2];rows.append(dict(condition=name,**{k:z[k] for k in ['release_speed_m_s','forward_m_s','lateral_m_s','vertical_m_s','release_angle_deg','minimum_ballistic_speed_m_s','minimum_rim_distance_m','body_tracking_rmse_rad','max_torque_saturation','release_elbow_bend_deg','release_trunk_tilt_deg','initial_knee_deg','release_knee_deg','both_feet_contact_rate','falls','warnings','score']}))
  fig,ax=plt.subplots(3,2,figsize=(12,10));t=np.array([float(x['time_s']) for x in r]);
  for a,keys in zip(ax.flat,[['R_knee_deg','L_knee_deg'],['trunk_tilt_deg','R_elbow_bend_deg'],['vx','vy','vz'],['ball_x','ball_y','ball_z'],['body_rmse_rad','torque_saturation'],['support_margin_m','foot_slip_m_s']]):
   for key in keys:a.plot(t,[float(x[key]) for x in r],label=key)
   a.axvline(z['release']['actual_time_s'],color='gray',ls='--');a.legend(fontsize=8);a.set_xlabel('time (s)');a.grid(alpha=.2)
  fig.tight_layout();fig.savefig(p/'diagnostics.png');plt.close(fig)
 (OUT/'comparison.json').write_text(json.dumps(results,indent=2));write_csv(OUT/'comparison.csv',rows)
 feasible=[(n,z) for n,z in results.items() if z['above_forward'] and not z['falls'] and not z['warnings'] and z['body_tracking_rmse_rad']<.08 and z['both_feet_contact_rate']>.95]
 # Aim error and minimum rim distance are actual-physics metrics; speed alone is not selection criterion.
 n,z=min(feasible,key=lambda x:x[1]['minimum_rim_distance_m']);(OUT/'recommended.json').write_text(json.dumps(dict(condition=n,reference=z['reference'],velocity_ff=z['velocity_ff'],score=z['score'],success_all_criteria=False),indent=2));print('Recommended physics candidate:',n,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['analyze','render','gui']);p.add_argument('--condition');p.add_argument('--view',choices=['side','front'],default='side');p.add_argument('--repeats',type=int,default=0);a=p.parse_args()
 if a.command=='analyze':analyze()
 else:replay(a.condition or json.loads((OUT/'recommended.json').read_text())['condition'],a.view,a.command=='gui',a.repeats)
