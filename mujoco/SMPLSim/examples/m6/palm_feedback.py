"""Reference-only wrist correction for world palm aim; original PD stays unchanged."""
import json,hashlib
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from examples.m6.core import ROOT,WeldShootingEnv,ids,smooth,RIM_CENTER
from examples.m6.palm_to_hoop import basis,blend
from examples.m5_bimanual_body_ik import unit,calibration,forearm_swing_twist
from examples.m3_control import write_csv
OUT=ROOT/'outputs/m6_palm_feedback'
class PalmReference:
 def __init__(self,e):
  self.e=e;self.base=e.reference
  for n in ['times','q','offset','release','duration']:setattr(self,n,getattr(self.base,n))
  self.motion_seconds=float(np.load(self.base_path(e))['duration']);self.ids=ids(e.m,[f'R_Wrist_{a}' for a in 'xyz']);self.bounds=np.array([e.m.jnt_range[e.m.jnt_qposadr==a][0] for a in self.ids]);w=e.wrist;W=e.d.xmat[w].reshape(3,3);f=unit(e.d.xpos[e.m.body('R_Middle1').id]-e.d.xpos[w]);self.local=basis(np.array([0,-1,0]),W.T@f);self.last_time=-1.;self.last_q=None;self.fitted=None
 def base_path(self,e):return e.reference_path
 def target(self,t):
  q=self.base.target(t);alpha=smooth((t-.55*self.motion_seconds)/(.35*self.motion_seconds))
  if t<=0:self.last_time=-1.;self.last_q=None;self.fitted=None;return q
  if alpha<=0:return q
  if self.last_time<0 or t-self.last_time>=1/30-1e-8:
   e=self.e;d=mujoco.MjData(e.m);d.qpos[:]=e.d.qpos;mujoco.mj_forward(e.m,d);W=d.xmat[e.wrist].reshape(3,3);normal=unit(RIM_CENTER-d.xpos[e.wrist]);final=basis(normal,[0,0,1])@self.local.T
   # Only a hand orientation goal is changed. Ball qpos/qvel are untouched.
   Rgoal=blend(W,final,alpha);seed=e.d.qpos[self.ids].copy() if self.last_q is None else self.last_q;dt=1/30 if self.last_time<0 else t-self.last_time;lo=np.maximum(self.bounds[:,0]+1e-5,seed-4*dt);hi=np.minimum(self.bounds[:,1]-1e-5,seed+4*dt)
   def loss(x):
    d.qpos[:]=e.d.qpos;d.qpos[self.ids]=x;mujoco.mj_kinematics(e.m,d);return np.r_[5*Rotation.from_matrix(Rgoal@d.xmat[e.wrist].reshape(3,3).T).as_rotvec(),.02*(x-seed)]
   self.last_q=least_squares(loss,np.clip(seed,lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=30).x;self.fitted=self.last_q.copy();self.last_time=t
  q[self.ids]=self.fitted;return q

def run():
 OUT.mkdir(parents=True,exist_ok=True);mp=OUT/'preservation_before.json'
 if not mp.exists():mp.write_text(json.dumps({str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*') if p.is_file() and not any(n in p.parts for n in ('.git','__pycache__','m6_palm_feedback')) and p.name!='.DS_Store'},indent=2))
 for duration in [1.1,.75]:
  ref=ROOT/f'outputs/m6_palm_to_hoop_v3/reference_{duration:g}s.npz';e=WeldShootingEnv(reference=ref,record=True);e.reference_path=ref;e.reference=PalmReference(e);e.reset(seed=7);extra=[]
  while True:
   _,_,t,u,_=e.step(np.zeros(e.action_space.shape));N=-e.d.xmat[e.wrist].reshape(3,3)[:,1];goal=unit(RIM_CENTER-e.d.xpos[e.wrist]);extra.append(dict(time_s=float(e.d.time),palm_aim_error_deg=float(np.degrees(np.arccos(np.clip(N@goal,-1,1)))),nx=N[0],ny=N[1],nz=N[2]))
   if t or u:break
  path=OUT/f'run_{duration:g}s';z=e.save(path);write_csv(path/'palm.csv',extra);i=min(range(len(extra)),key=lambda k:abs(extra[k]['time_s']-z['release']['actual_time_s']));z['palm_error_at_release_deg']=extra[i]['palm_aim_error_deg'];z['palm_final_error_deg']=extra[-1]['palm_aim_error_deg'];(path/'summary.json').write_text(json.dumps(z,indent=2));print(duration,z,flush=True);e.close()
 old=json.loads(mp.read_text());changed=[p for p,h in old.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];(OUT/'preservation_audit.json').write_text(json.dumps(dict(protected=len(old),changed=changed),indent=2));assert not changed
if __name__=='__main__':run()
