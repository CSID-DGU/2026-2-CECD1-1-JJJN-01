"""Candidate with explicit trunk axis and calibrated relative wrist constraint."""
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from examples.m6.core import OUT,MODELS,Reference,ids,smooth,RIM_CENTER,generate_reference
from examples.m5_bimanual_body_ik import calibration,whole_body_ik,head_ik,unit,trunk_axis,extension_angle
from examples.m5_bimanual_reorientation import pose_ik
from examples.m3_control import write_csv

def generate(duration=1.2):
 path=OUT/f'reference_coordinated_{duration:g}s.npz'
 if path.exists():return path
 old=Reference(generate_reference('shoot',duration));m=mujoco.MjModel.from_xml_path(str(MODELS/'body_01.xml'));d=mujoco.MjData(m);d.qpos[:]=old.q[0];mujoco.mj_forward(m,d);cal=calibration(m,d);W0=d.xmat[m.body('R_Wrist').id].reshape(3,3).copy();center0=d.xpos[m.body('R_Wrist').id]+W0@old.offset
 ra=ids(m,[f'R_{b}_{a}' for b in ('Shoulder','Elbow','Wrist') for a in 'xyz']);tr=ids(m,[f'{b}_{a}' for b in ('Torso','Spine','Chest') for a in 'xyz']);combo=np.r_[tr,ra];head=ids(m,[f'{b}_{a}' for b in ('Neck','Head') for a in 'xyz']);side=np.array([0,1.,0]);forward=np.array([1.,0,0]);samples=[];rows=[];previous=old.q[0].copy()
 for t,base in zip(old.times,old.q):
  q=base.copy();u=smooth((t-.2)/(duration*.68));push=smooth((t-duration*.58)/(duration*.28));center=center0+(.10*u+.17*push)*forward+[0,0,.48*u+.12*push];W=Rotation.from_rotvec(side*np.radians(-20*u+30*push)).as_matrix()@W0
  axis0=cal['trunk_axis'];tilt0=np.arccos(np.clip(axis0[2],-1,1));tilt=max(0,tilt0-np.radians(7)*u);axis=unit(np.r_[axis0[:2],0])*np.sin(tilt)+[0,0,np.cos(tilt)];extension=np.radians(18*u-28*push)
  q,info=whole_body_ik(m,q,center-W@old.offset,W,axis,cal,extension,previous[combo],iterations=45);q=head_ik(m,q,RIM_CENTER,previous[head]);d.qpos[:]=q;mujoco.mj_forward(m,d)
  rows.append(dict(time_s=t,position_error_mm=1000*np.linalg.norm(d.xpos[m.body('R_Wrist').id]+d.xmat[m.body('R_Wrist').id].reshape(3,3)@old.offset-center),trunk_tilt_deg=np.degrees(np.arccos(np.clip(trunk_axis(m,d)[2],-1,1))),wrist_extension_proxy_deg=np.degrees(extension_angle(m,d,cal)),desired_wrist_extension_proxy_deg=np.degrees(extension)))
  samples.append(q.copy());previous=q
 # Release within the acceleration window, not after the arm path stops.
 np.savez_compressed(path,time_s=old.times,qpos=samples,offset=old.offset,release_time_s=duration*.70,kind='coordinated',source='synthetic constraint-based reference, not validated natural motion');write_csv(OUT/f'reference_coordinated_{duration:g}s_fk.csv',rows);return path
if __name__=='__main__':print(generate())
