"""Separate timing-only, acceleration-only, then combined Weld launch tests.

Retiming changes only release_time_s; every joint target stays byte-identical.
The ball is initialized once and then moved exclusively by MuJoCo constraints.
"""
import argparse,json,hashlib,csv
from pathlib import Path
import numpy as np,mujoco
from examples.m6_2 import experiment as exp,folded_raise_v4 as generator
from examples.m6_2.refine import StableVelocityEnv
from examples.m6_2.experiment import ROOT,core,RIM_CENTER,unit
OUT=ROOT/'outputs/m6_2_release_acceleration'
BASE=ROOT/'outputs/m6_2_folded_raise/references/folded_right_turn.npz'
core.OUT=OUT

def audit(start=False):
 OUT.mkdir(exist_ok=True);p=OUT/'preservation_before.json'
 if start and not p.exists():p.write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.rglob('*') if f.is_file() and not any(n in f.parts for n in ('.git','__pycache__','m6_2_release_acceleration')) and f.name!='.DS_Store'},indent=2))
 if not start:
  old=json.loads(p.read_text());changed=[f for f,h in old.items() if not Path(f).exists() or hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h];(OUT/'preservation_audit.json').write_text(json.dumps(dict(protected=len(old),changed=changed),indent=2));assert not changed,changed

def retime(source,name,release):
 z={k:np.array(v).copy() for k,v in np.load(source).items()};z['release_time_s']=np.array(float(release));p=OUT/'references'/f'{name}.npz';p.parent.mkdir(exist_ok=True);np.savez_compressed(p,**z)
 original=np.load(source);new=np.load(p)
 for k in original.files:
  if k!='release_time_s':np.testing.assert_array_equal(original[k],new[k])
 return p

def evaluate(name,ref,group):
 exp.OUT=OUT;exp.core.OUT=OUT;exp.ShootingEnv=StableVelocityEnv
 # Capture at the exact physics step before equality deactivation.
 captured=[]
 class MeasuredEnv(StableVelocityEnv):
  def reset(self,**kwargs):
   o=super().reset(**kwargs);update=self.grip.update
   def release(d):
    if not self.grip.released and d.time+1e-10>=self.grip.release_time:
     jp=np.zeros((3,self.m.nv));jr=jp.copy();mujoco.mj_jacBody(self.m,d,jp,jr,self.wrist);v=jp@d.qvel;w=jr@d.qvel;ball=d.qpos[self.qa:self.qa+3].copy();r=ball-d.xpos[self.wrist];spin=np.cross(w,r);bv,_=self.ball_velocity();f=unit(np.r_[(RIM_CENTER-ball)[:2],0]);captured.append(dict(wrist_velocity_m_s=v.tolist(),wrist_omega_rad_s=w.tolist(),rotation_velocity_m_s=spin.tolist(),wrist_forward_m_s=float(v@f),rotation_forward_m_s=float(spin@f),ball_velocity_m_s=bv.tolist(),velocity_relation_residual_m_s=float(np.linalg.norm(bv-v-spin))))
    return update(d)
   self.grip.update=release;return o
 exp.ShootingEnv=MeasuredEnv
 z=exp.run_one(name,ref,.8);p=OUT/'runs'/name;rows=list(csv.DictReader((p/'physics.csv').open()));kin=list(csv.DictReader((p/'kinematics.csv').open()));ts=np.array([float(r['time_s']) for r in rows]);q=np.array([[float(r['ball_'+a]) for a in 'xyz'] for r in rows]);rv=np.array([[float(r['v'+a]) for a in 'xyz'] for r in rows]);s=np.load(ref);push_start=float(s['push_start']);i=int(np.argmin(abs(ts-(push_start-.015))))
 z.update(group=group,push_start_s=push_start,arm_push_seconds=float(s['push_s']),snap_degrees=float(s['snap_deg']),setpoint_time_s=float(ts[i]),setpoint_ball_height_m=float(rows[i]['ball_z']),setpoint_ball_above_head_m=float(rows[i]['ball_z'])-float(kin[i]['Head_z']),setpoint_elbow_bend_deg=float(rows[i]['R_elbow_bend_deg']),setpoint_knee_bend_deg=float(rows[i]['R_knee_deg']),velocity_decomposition=captured[-1] if captured else None)
 first=ts>=z['release']['actual_time_s'];hit=np.flatnonzero(first&(q[:,2]<.14)&(rv[:,2]<0));end=hit[0] if len(hit) else len(ts);mask=first&(np.arange(len(ts))<end);z['first_flight_minimum_rim_distance_m']=float(np.linalg.norm(q[mask]-RIM_CENTER,axis=1).min()) if mask.any() else 100.
 # Gate for the user's folded-arm raise -> coordinated knee/arm extension.
 z['pose_order_pass']=bool(z['setpoint_ball_height_m']>1.6 and z['setpoint_ball_above_head_m']>.1 and z['setpoint_elbow_bend_deg']>90 and z['setpoint_knee_bend_deg']>30)
 z['stable_tracking_pass']=bool(not z['falls'] and not z['warnings'] and z['body_tracking_rmse_rad']<.06 and z['both_feet_contact_rate']>.95)
 z['continuous_release_pass']=bool(all(z['release'][k]==0 for k in ('position_jump_m','velocity_jump_m_s','angular_velocity_jump_rad_s')))
 (p/'summary.json').write_text(json.dumps(z,indent=2));return z

def run():
 audit(True);generator.OUT=OUT;results={}
 # Baseline and timing-only: same motion including unchanged left-guide path.
 results['A_baseline']=evaluate('A_baseline',retime(BASE,'A_baseline',1.70),'baseline')
 for t in [1.50,1.53,1.56,1.59,1.62,1.65]:
  n=f'B_release_{t:g}';results[n]=evaluate(n,retime(BASE,n,t),'timing_only')
 # Acceleration-only keeps the release at the same 80% phase.
 sources={}
 for duration in [.30,.22,.16]:
  n=f'C_arm_{duration:g}';r=generator.make(n,1.2,duration,35,release_fraction=.8,lower_s=duration);sources[(duration,35)]=r;results[n]=evaluate(n,r,'acceleration_only')
 for snap in [0,20,55]:
  n=f'C_wrist_{snap:g}';r=generator.make(n,1.2,.30,snap,release_fraction=.8,lower_s=.30);sources[(.30,snap)]=r;results[n]=evaluate(n,r,'acceleration_only')
 # Independent timing scans of each frozen acceleration reference.
 for (duration,snap),source in sources.items():
  for fraction in [.35,.50,.65]:
   n=f'D_joint_{duration:g}_{snap:g}_{fraction:g}';r=retime(source,n,1.38+duration*fraction);results[n]=evaluate(n,r,'combined')
 (OUT/'comparison.json').write_text(json.dumps(results,indent=2));summarize(results);audit()

def summarize(results=None):
 from examples.m3_control import write_csv
 r=results or json.loads((OUT/'comparison.json').read_text());keys=['release_speed_m_s','forward_m_s','lateral_m_s','vertical_m_s','release_angle_deg','first_flight_minimum_rim_distance_m','minimum_ballistic_speed_m_s','body_tracking_rmse_rad','max_torque_saturation','both_feet_contact_rate','min_com_margin_m','foot_slip_rms_m_s','falls','warnings','score','pose_order_pass','stable_tracking_pass']
 write_csv(OUT/'comparison.csv',[dict(condition=n,group=z['group'],release_time_s=z['release']['actual_time_s'],arm_push_seconds=z['arm_push_seconds'],snap_degrees=z['snap_degrees'],**{k:z[k] for k in keys}) for n,z in r.items()])
 eligible=[(n,z) for n,z in r.items() if z['pose_order_pass'] and z['stable_tracking_pass'] and z['continuous_release_pass'] and z['forward_m_s']>0 and z['vertical_m_s']>0 and abs(z['lateral_m_s'])<z['forward_m_s'] and 20<z['release_angle_deg']<80]
 if not eligible:raise RuntimeError('No safe forward/up launch passes the motion-order gates')
 name,z=min(eligible,key=lambda item:item[1]['first_flight_minimum_rim_distance_m']);(OUT/'recommended.json').write_text(json.dumps(dict(condition=name,reference=z['reference'],score=z['score'],rim_reachable=z['release_speed_m_s']>=z['minimum_ballistic_speed_m_s'],selection='stable motion-order gates + forward/up + lateral constraint + first-flight rim distance'),indent=2));print('Recommended:',name,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['run','summarize','audit','gui','render']);p.add_argument('--condition');p.add_argument('--view',choices=['side','front'],default='side');p.add_argument('--repeats',type=int,default=0);a=p.parse_args()
 if a.command=='run':run()
 elif a.command=='summarize':summarize()
 elif a.command=='audit':audit()
 else:
  from examples.m6_2 import visualize
  visualize.OUT=OUT;visualize.replay(a.condition or json.loads((OUT/'recommended.json').read_text())['condition'],a.view,a.command=='gui',a.repeats)
