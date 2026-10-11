"""Implicit Stable PD target velocity experiment, no original controller edits."""
import json,numpy as np
from examples.m6_2.experiment import OUT,make,run_one,ShootingEnv,VelocityBridge

class StableVelocityBridge(VelocityBridge):
 def control(self,d):
  self.base.control(d);c=self.base.controller;pd=c.pd;m=self.base.model;p=self.base.data
  _,target=c.control(p,self.base.goal);dt=m.opt.timestep
  vdes=self.ref.spline(np.clip(d.time,0,self.ref.duration),1)
  names=[m.joint(i+1).name for i in range(m.nu)];mask=np.array([any(b in n for b in ('Shoulder','Elbow','Wrist','Torso','Spine','Chest')) for n in names]);vdes=self.amount*vdes*mask
  qerr=np.r_[np.zeros(6),p.qpos[7:]+p.qvel[6:]*dt-target];verr=p.qvel.copy();verr[6:]-=vdes
  kp=np.r_[np.zeros(6),pd.jkp];kd=np.r_[np.zeros(6),pd.jkd]
  acc=pd._compute_desired_accel(qerr,verr,kp,kd,m,p)
  d.ctrl[:]=np.clip(-pd.jkp*qerr[6:]-pd.jkd*(verr[6:]+acc[6:]*dt),-c.limit,c.limit)

class StableVelocityEnv(ShootingEnv):
 def reset(self,**kwargs):
  # Parent initially installs the old additive adapter. Replace it before step.
  result=super().reset(**kwargs)
  if self.velocity_ff:self.bridge=StableVelocityBridge(self.bridge.base,self.reference,self.velocity_ff)
  return result

def run():
 import examples.m6_2.experiment as exp
 exp.ShootingEnv=StableVelocityEnv
 results=json.loads((OUT/'comparison.json').read_text())
 for ps in [.22,.14]:
  for ff in [.25,.5,.8]:
   for frac in [.65,.85]:
    n=f'G_stable_{ps:g}_{ff:g}_{frac:g}';ref=make(n,.85,ps,-25,release_fraction=frac,lower_s=.45);results[n]=run_one(n,ref,ff);results[n]['controller']='implicit stable desired velocity'
 (OUT/'comparison.json').write_text(json.dumps(results,indent=2));exp.audit()
if __name__=='__main__':run()
