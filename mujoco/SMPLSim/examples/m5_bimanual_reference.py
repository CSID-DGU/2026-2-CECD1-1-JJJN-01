"""Body-specific experimental MuJoCo reference for later residual curricula.
Never applies state changes; returns targets only. Not SMPL pose ground truth.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp

class ReferenceSequence:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as f:
            self.times=f['time_s'].astype(float);self.targets=f['target_qpos'].astype(float);self.joint_names=f['joint_names'].tolist()
        if self.times.ndim!=1 or len(self.times)<2 or len(self.targets)!=len(self.times) or not np.isfinite(self.targets).all() or not np.all(np.diff(self.times)>0):raise ValueError('Invalid experimental reference')
        self.rotation=Slerp(self.times,Rotation.from_quat(self.targets[:,3:7][:,[1,2,3,0]]))
    def target(self,t):
        t=float(np.clip(t,self.times[0],self.times[-1]));j=min(np.searchsorted(self.times,t,side='right')-1,len(self.times)-2);a=(t-self.times[j])/(self.times[j+1]-self.times[j]);q=(1-a)*self.targets[j]+a*self.targets[j+1];q[3:7]=self.rotation(t).as_quat()[[3,0,1,2]];return q

def curriculum(summary):
    """Observed outcomes are diagnostics, not fixed pose labels or PPO rewards."""
    return dict(body='body_01',representation='MuJoCo humanoid target qpos; free root remains physical',not_smpl_axis_angle=True,not_ground_truth=True,physical_reference_valid=summary.get('physical_reference_valid',False),failure_reason=summary['reason'],learning_now=False,large_ppo_allowed=False,stages=[dict(name=n,observed_success=summary.get(k,False)) for n,k in [('right_reorientation','geometric_rotation_success'),('wrist_extension','wrist_extension_verified'),('bimanual_contact','bimanual_holding_success'),('trunk_extension','trunk_extension_verified'),('raising','raising_success'),('shooting_ready','shooting_ready_success')]],future_small_experiment='Start from protected stable right holding; learn guide approach before reorientation; terminate on escape/fall; compare against no-learning baseline. Do not replay failed reference as validated motion.')
