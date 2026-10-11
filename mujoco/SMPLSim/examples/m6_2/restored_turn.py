"""Restore the user-requested large hand turn while raising the ball."""
from examples.m6_2.experiment import *
def make(name,raise_s=1.1,push_s=.4,snap=0.,release_fraction=.55,lower_s=None):
 path=OUT/'references'/f'{name}.npz';path.parent.mkdir(exist_ok=True)
 if path.exists():return path
 oldturn=core.Reference(ROOT/'outputs/m6_palm_to_hoop_v4/reference_0.75s.npz');turn_data=mujoco.MjData(model())
 m=model();base=initial();leg,qs,_=lower_path(m,base,22);start=qs[-1].copy();d=mujoco.MjData(m);d.qpos[:]=start;mujoco.mj_forward(m,d)
 offset=np.load(OLD/'reference_shoot.npz')['offset'];p0,W0,f0=geometry(m,d,offset);forward=unit(np.r_[(RIM_CENTER-p0)[:2],0]);side=unit(np.cross([0,0,1],forward));local=basis(-W0[:,1],f0);local=W0.T@local
 # Set point relative to head, not a long forward reach. Elbow stays below ball.
 upright=mujoco.MjData(m);upright.qpos[:]=base;mujoco.mj_forward(m,upright);head=upright.xpos[m.body('Head').id].copy();shoulder=upright.xpos[m.body('R_Shoulder').id].copy();setpoint=head+.12*forward+.09*side+np.array([0,0,.12]);setpoint[2]=max(head[2]+.12,shoulder[2]+.35)
 # Palm supports below/behind; finger direction lies in the sagittal plane.
 normal=unit(.32*forward+np.array([0,0,.95]));cock=basis(normal,forward)@local.T
 push_end=setpoint+.20*forward+np.array([0,0,.25])
 la=ids(m,[f'L_{b}_{a}' for b in ('Shoulder','Elbow','Wrist') for a in 'xyz']);ra=ids(m,[f'R_{b}_{a}' for b in ('Shoulder','Elbow','Wrist') for a in 'xyz']);tr=ids(m,[f'{b}_{a}' for b in ('Torso','Spine','Chest') for a in 'xyz']);hd=ids(m,[f'{b}_{a}' for b in ('Neck','Head') for a in 'xyz']);bounds=np.array([m.jnt_range[m.jnt_qposadr==a][0] for a in ra]);tl=np.array([m.jnt_range[m.jnt_qposadr==a][0] for a in tr]);L0=d.xpos[m.body('L_Wrist').id].copy();WL=d.xmat[m.body('L_Wrist').id].reshape(3,3).copy();axis0=trunk_axis(m,d);previous=start.copy();samples=[];rows=[]
 t0=.18;push_start=t0+raise_s*.82;release=push_start+push_s*release_fraction;total=max(push_start+push_s+.45,release+2.8);times=np.arange(0,total+1/120,1/120);lower_s=lower_s or raise_s*.85
 for t in times:
  up=smooth((t-t0)/raise_s);ext=smooth((t-t0)/lower_s);push=smooth((t-push_start)/push_s);q=start.copy();q[leg]=(1-ext)*start[leg]+ext*base[leg];q[:3]=(1-ext)*start[:3]+ext*base[:3]
  tilt=(1-ext)*np.arccos(np.clip(axis0[2],-1,1))+ext*np.radians(5);axisgoal=unit(np.r_[axis0[:2],0])*np.sin(tilt)+np.array([0,0,np.cos(tilt)])
  def trunk(x):
   d.qpos[:]=q;d.qpos[tr]=x;mujoco.mj_kinematics(m,d);return np.r_[7*(trunk_axis(m,d)-axisgoal),.04*(x-previous[tr])]
  q[tr]=least_squares(trunk,np.clip(previous[tr],tl[:,0]+1e-6,tl[:,1]-1e-6),bounds=(tl[:,0]+1e-6,tl[:,1]-1e-6),max_nfev=25).x
  center=(1-up)*p0+up*setpoint+push*(push_end-setpoint);turn_progress=np.clip((t-t0)/raise_s,0,1);turn_q=oldturn.target(.75*turn_progress);turn_data.qpos[:]=turn_q;mujoco.mj_forward(m,turn_data);W=turn_data.xmat[m.body('R_Wrist').id].reshape(3,3).copy();W=Rot.from_rotvec(-side*np.radians(20)*up).as_matrix()@W
  # World sagittal snap defined by palm/finger geometry. Sign is probed, not inferred from a joint name.
  snap_progress=smooth((t-push_start-push_s*.15)/(push_s*.7));W=Rot.from_rotvec(side*np.radians(snap)*snap_progress).as_matrix()@W
  elbowgoal=np.radians((1-up)*58+up*95-65*push)
  def arm(x):
   d.qpos[:]=q;d.qpos[ra]=x;mujoco.mj_kinematics(m,d);Wr=d.xmat[m.body('R_Wrist').id].reshape(3,3);e=d.xpos[m.body('R_Elbow').id];c=d.xpos[m.body('R_Wrist').id]+Wr@offset
   return np.r_[35*(c-center),2.5*Rot.from_matrix(W@Wr.T).as_rotvec(),2*(elbow_bend(m,d)-elbowgoal),8*max(0,e[2]-center[2]+.06),.04*(x-previous[ra]),.8*(x[6:]-turn_q[ra[6:]])]
  lo=np.maximum(bounds[:,0]+1e-5,previous[ra]-12/120);hi=np.minimum(bounds[:,1]-1e-5,previous[ra]+12/120)
  q[ra]=least_squares(arm,np.clip(previous[ra],lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=40).x
  left=(1-up)*L0+up*(center+.20*side-WL@offset)+.13*side*smooth((t-release+.12)/.35);q,_=pose_ik(m,q,'L',left,WL,previous[la],iterations=15);q=head_ik(m,q,RIM_CENTER,previous[hd]);d.qpos[:]=q;mujoco.mj_forward(m,d);c,actualW,f=geometry(m,d,offset)
  rows.append(dict(time_s=t,raising=up,lower_extension=ext,arm_push=push,snap=snap_progress,ball_goal_x=center[0],ball_goal_y=center[1],ball_goal_z=center[2],fk_ball_error_mm=1000*np.linalg.norm(c-center),elbow_deg=np.degrees(elbow_bend(m,d)),orientation_error_deg=np.degrees(Rot.from_matrix(W@actualW.T).magnitude())))
  samples.append(q.copy());previous=q.copy()
 np.savez_compressed(path,time_s=times,qpos=samples,offset=offset,release_time_s=release,source='M6-2 restored upright/forearm roll/palm hoop sequence, cock-back and snap; not ground truth',setpoint=setpoint,raise_s=raise_s,push_s=push_s,snap_deg=snap,push_start=push_start)
 write_csv(path.with_suffix('.csv'),rows);return path

if __name__=='__main__':
 from examples.m6_2.refine import StableVelocityEnv
 import examples.m6_2.experiment as e
 e.ShootingEnv=StableVelocityEnv
 for snap in [0,35]:
  n=f'I_restored_turn_{snap:g}'
  ref=make(n,1.0,.22,snap,release_fraction=.65,lower_s=.5)
  z=e.run_one(n,ref,.8);z['controller']='implicit stable desired velocity';z['restored_turn']=True
  (OUT/f'{n}_summary.json').write_text(json.dumps(z,indent=2))
