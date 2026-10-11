"""Separate contact-only bimanual reorientation; protected M4/M5 stay unchanged."""
import sys,json,argparse,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from examples.m5_coordinated_reference import CoordinatedEnv,CoordinatedTargets,lift,MODELS,SOURCE,unit,reserve,scene_rim
from examples.m5_upper_body_ik import joint_ids,minimum_jerk,anatomical_mapping
from examples.m5_shooting_grip_experiment import contact_wrenches
from examples.m5_bimanual_contact import side_contacts,local_ball,hand_geom,gap_to_ball
from examples.m3_control import write_csv
from examples.m5_bimanual_reference import curriculum
from examples.m5_bimanual_body_ik import calibration,extension_angle,forearm_swing_twist,trunk_axis,whole_body_ik,head_ik
OUT=ROOT/'outputs/m5_1_bimanual_reorientation'


def pose_ik(m,q,side,position,orientation,seed=None,iterations=35):
    """Actual MuJoCo Jacobian over shoulder/elbow/wrist; no Euler decomposition."""
    names=[f'{side}_{b}_{a}' for b in ('Shoulder','Elbow','Wrist') for a in 'xyz'];ids=joint_ids(m,names);dofs=[int(m.joint(n).dofadr[0]) for n in names];bounds=np.array([m.jnt_range[m.joint(n).id] for n in names]);lo,hi=bounds[:,0]+1e-5,bounds[:,1]-1e-5
    p=mujoco.MjData(m);p.qpos[:]=q;p.qpos[ids]=np.clip(q[ids] if seed is None else seed,lo,hi);anchor=p.qpos[ids].copy();bid=m.body(side+'_Wrist').id
    for _ in range(iterations):
        mujoco.mj_kinematics(m,p);mujoco.mj_comPos(m,p);dp=position-p.xpos[bid];dr=Rotation.from_matrix(orientation@p.xmat[bid].reshape(3,3).T).as_rotvec()
        if np.linalg.norm(dp)<.0002 and np.linalg.norm(dr)<.003:break
        jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv));mujoco.mj_jacBody(m,p,jp,jr,bid);J=np.vstack((35*jp[:,dofs],2*jr[:,dofs]));error=np.r_[35*dp,2*dr];penalty=np.diag([.015]*6+[.022]*3);step=np.linalg.solve(J.T@J+penalty,J.T@error-penalty@(p.qpos[ids]-anchor));step*=min(1,.18/max(np.linalg.norm(step),1e-10));p.qpos[ids]=np.clip(p.qpos[ids]+step,lo,hi)
    mujoco.mj_kinematics(m,p);report=dict(position_error_mm=float(np.linalg.norm(position-p.xpos[bid])*1000),orientation_error_deg=float(np.degrees(Rotation.from_matrix(orientation@p.xmat[bid].reshape(3,3).T).magnitude())))
    return p.qpos.copy(),report


def palm_basis(m,d,side):
    b=m.body(side+'_Wrist').id;W=d.xmat[b].reshape(3,3);finger=unit(d.xpos[m.body(side+'_Middle1').id]-d.xpos[b]);normal=-W[:,1];local=W.T@finger;local=unit(local-[0,local[1],0]);return np.column_stack((unit(np.cross([0,1,0],local)),[0,1,0],local)),finger,normal


def screen():
    ref=dict(np.load(ROOT/'outputs/m5_1_coordinated_reference/reference/reference.npz'));m=mujoco.MjModel.from_xml_path(str(MODELS/'body_01.xml'));d=mujoco.MjData(m);d.qpos[:]=ref['initial'][:m.nq];mujoco.mj_kinematics(m,d);rows=[];axes=[];normal=-ref['right_rotation'][:,1]
    for n in [f'R_{b}_{a}' for b in ('Shoulder','Elbow','Wrist') for a in 'xyz']:
        j=m.joint(n).id;axes.append(dict(joint=n,axis_x=d.xaxis[j,0],axis_y=d.xaxis[j,1],axis_z=d.xaxis[j,2],lower_rad=m.jnt_range[j,0],upper_rad=m.jnt_range[j,1]))
    for yaw in (-180,-150,150,180):
        for orbit in (0.,.5,1.):
            q=d.qpos.copy();p0=ref['ball_initial']-ref['right_rotation']@ref['rel0']
            for u in np.linspace(0,1,41):
                R=Rotation.from_rotvec(normal*np.radians(yaw)*u).as_matrix()@ref['right_rotation'];position=(1-orbit)*p0+orbit*(ref['ball_initial']-R@ref['rel0']);q,z=pose_ik(m,q,'R',position,R,iterations=80);rows.append(dict(yaw_deg=yaw,orbit=orbit,fraction=u,**z))
    write_csv(OUT/'joint_axes.csv',axes);write_csv(OUT/'ik_screen.csv',rows);print(json.dumps([dict(yaw_deg=r['yaw_deg'],orbit=r['orbit'],**{k:r[k] for k in ('position_error_mm','orientation_error_deg')}) for r in rows if r['fraction']==1]),flush=True)


def body_screen():
    """Pure FK feasibility, explicitly excluded from physical success gates."""
    e=RotationEnv(condition='A');m=e.bridge.model;d=mujoco.MjData(m);d.qpos[:]=e.d.qpos[:m.nq];mujoco.mj_kinematics(m,d);cal=calibration(m,d);w=m.body('R_Wrist').id;p0=d.xpos[w].copy();W0=d.xmat[w].reshape(3,3).copy();q=d.qpos.copy();tilt0=np.arccos(np.clip(cal['trunk_axis'][2],-1,1));rows=[];targets=[];seeds=np.r_[e.adapter.trunk_ids,e.arm_ids]
    try:
        for u in np.linspace(0,1,61):
            R=Rotation.from_rotvec(-W0[:,1]*np.radians(-150)*u).as_matrix()@W0;axis=unit(np.cross(R@cal['finger_local'],-R[:,1]));R=Rotation.from_rotvec(-axis*np.radians(15)*u).as_matrix()@R;tilt=max(0,tilt0-np.radians(8)*u);trunk=unit(unit(np.r_[cal['trunk_axis'][:2],0.])*np.sin(tilt)+[0,0,np.cos(tilt)]);q,z=whole_body_ik(m,q,p0,R,trunk,cal,np.radians(15)*u,seed=q[seeds],iterations=80);d.qpos[:]=q;mujoco.mj_kinematics(m,d);rows.append(dict(fraction=u,kinematic_only=True,actual_fk_extension_deg=float(np.degrees(extension_angle(m,d,cal))),actual_fk_trunk_straightening_deg=float(np.degrees(tilt0-np.arccos(np.clip(trunk_axis(m,d)[2],-1,1)))),**z));targets.append(q.copy())
        write_csv(OUT/'whole_body_fk_screen.csv',rows);np.savez_compressed(OUT/'whole_body_fk_targets.npz',qpos=targets,fraction=np.linspace(0,1,61),kinematic_only=True,physical_reference_valid=False);print(json.dumps(rows[-1]),flush=True)
    finally:e.close()


class RotationTargets(CoordinatedTargets):
    def __init__(self,*args,experiment,**kwargs):
        super().__init__(*args,**kwargs);self.exp=experiment;self.rotation_start=None;self.rotation_finished=None;self.bimanual_since=None;self.phase='A';self.ik_time=None;self.guide_depth=0.;self.force_filtered=0.;self.jac_reports={};self.pulse_dirs=None;self.last_t=0;self.left_lost_since=None
    def target(self,t,actual=None):
        if t<=0 or actual is None:return super().target(t,actual)
        dt=max(t-self.last_t,self.m.opt.timestep);self.last_t=t;c=self.exp;r=self.reference;kind=c['condition'];guide=kind in ('C','D');L=side_contacts(actual.model,actual,'L');Rcontact=side_contacts(actual.model,actual,'R');ball=actual.qpos[int(actual.model.joint('basketball_free').qposadr[0]):][:3]
        if guide and t>1.:
            self.force_filtered+=(1-np.exp(-dt/.06))*(L['normal_force']-self.force_filtered)
            self.guide_depth+=np.clip(.002*(c['guide_force']-self.force_filtered),-.06,.004)*dt;self.guide_depth=np.clip(self.guide_depth,-.04,.035 if c.get('proximity_guide',False) else .025)
            if c.get('proximity_guide',False) and t>2 and L['count']==0:
                closest_gap=min(gap_to_ball(actual.model,actual,actual.model.geom(g).name,ball) for g in range(actual.model.ngeom) if hand_geom(actual.model.geom(g).name or '', 'L'));self.guide_depth=min(.035,self.guide_depth+np.clip(2*closest_gap,0,.012)*dt)
        # Both-hand physical gate, independent of right-only stability.
        stable_guide=guide and t>2 and L['count']>0 and L['normal_force']<3 and Rcontact['count']>0 and np.linalg.norm(ball-r['ball_initial'])<.04
        if stable_guide:
            if self.bimanual_since is None:self.bimanual_since=t
        elif L['count']==0 and self.left_lost_since is not None and t-self.left_lost_since>.04:self.bimanual_since=None
        if L['count']==0:
            if self.left_lost_since is None:self.left_lost_since=t
        else:self.left_lost_since=None
        if self.rotation_start is None:
            if not guide and t>=3:self.rotation_start=t;self.finger_at_rotation_start=unit(actual.xpos[actual.model.body('R_Middle1').id]-actual.xpos[actual.model.body('R_Wrist').id])
            elif guide and self.bimanual_since is not None and t-self.bimanual_since>=.25:self.rotation_start=t;self.finger_at_rotation_start=unit(actual.xpos[actual.model.body('R_Middle1').id]-actual.xpos[actual.model.body('R_Wrist').id])
        elapsed=0 if self.rotation_start is None else max(0,t-self.rotation_start);u=minimum_jerk(elapsed/c['rotation_seconds']) if kind!='A' else 0
        if self.rotation_finished is None and self.rotation_start is not None and elapsed>=c['rotation_seconds']:self.rotation_finished=t
        raise_elapsed=0 if self.rotation_finished is None else t-self.rotation_finished
        self.phase='A' if self.rotation_start is None else 'B' if self.rotation_finished is None else 'C' if raise_elapsed<4 else 'D' if raise_elapsed<5 else 'E'
        if kind=='A':
            raise_elapsed=t-5.;self.phase='A' if t<3 else 'C' if t<9 else 'D' if t<10 else 'E'
        base=lift.LiftingTargets.target(self,0,actual);q=base.copy();align=minimum_jerk(t/2);q[self.trunk_ids]=(1-align)*base[self.trunk_ids]+align*r['aligned'][self.trunk_ids];q[self.head_ids]=(1-align)*base[self.head_ids]+align*r['aligned'][self.head_ids]
        # A compares preserved -15/-5 grip; B/C/D start there before turning.
        if not hasattr(self,'body_calibration'):
            self.body_calibration=calibration(actual.model,actual);self.nominal_trunk=self.body_calibration['trunk_axis'].copy()
        body_blend=minimum_jerk(elapsed/max(c['rotation_seconds']*.7,1))
        initial=Rotation.from_rotvec(r['side']*np.radians(-5)).as_matrix()@Rotation.from_rotvec(-r['right_rotation'][:,1]*np.radians(-15)).as_matrix()@r['right_rotation']
        W=r['right_rotation']@Rotation.from_rotvec(minimum_jerk(t/1.5)*Rotation.from_matrix(r['right_rotation'].T@initial).as_rotvec()).as_matrix()
        if kind!='A':W=Rotation.from_rotvec((-initial[:,1])*np.radians(c['yaw'])*u).as_matrix()@W
        if kind!='A' and c['wrist_extension_deg']>0:
            f_local=self.body_calibration['finger_local'];ext_axis=unit(np.cross(W@f_local,-W[:,1]));W=Rotation.from_rotvec(-ext_axis*np.radians(c['wrist_extension_deg'])*body_blend).as_matrix()@W
        center=r['ball_initial'].copy()
        if kind=='D':
            progress=minimum_jerk(max(elapsed-c['rotation_seconds']*.7,0)/4.);center=(1-progress)*r['ball_initial']+progress*r['ball_goal_shoot'];self.phase='C' if progress>0 and self.rotation_finished is None else self.phase
        elif kind=='A':
            center=r['ball_initial']+(r['ball_goal_shoot']-r['ball_initial'])*minimum_jerk(max(t-3,0)/6.)
        # Planned reposition from the grip geometry, not a static-ball error test.
        reposition=(1-c['orbit'])*(W@r['rel0']-initial@r['rel0'])
        if kind=='A':reposition=np.zeros(3)
        center=center+(1-progress)*reposition if kind=='D' else center+reposition
        self.ball_target=center.copy();self.desired_right_R=W.copy();fixed_wrist=center-initial@r['rel0'];orbit=center-W@r['rel0'];self.cartesian_right=center-W@r['rel0']
        # Right tracks ball via arm displacement only, never writes ball state.
        self.cartesian_right+=np.clip(.35*(center-ball),-.025,.025);self.nominal_cartesian_right=self.cartesian_right.copy();self.feedback_offset=np.zeros(3)
        if not hasattr(self,'left_basis'):
            probe=mujoco.MjData(self.m);probe.qpos[:]=r['initial'][:self.m.nq];mujoco.mj_kinematics(self.m,probe);self.left_basis=palm_basis(self.m,probe,'L')[0];self.left_initial_R=probe.xmat[self.m.body('L_Wrist').id].reshape(3,3).copy();self.left_initial_p=probe.xpos[self.m.body('L_Wrist').id].copy();self.left_mapping=anatomical_mapping(self.m,probe,'L');self.left_capsule_radius=float(self.m.geom('L_Wrist').size[0])
        # Lower-side support increases upward normal component when right turns.
        normal=unit(-r['side']*np.cos(np.radians(c['support_tilt']))+np.array([0,0,np.sin(np.radians(c['support_tilt']))]));finger_guess=np.cos(np.radians(c['guide_finger_elevation']))*r['forward']+np.sin(np.radians(c['guide_finger_elevation']))*np.array([0.,0.,1.]);finger=unit(finger_guess-normal*(finger_guess@normal));local_target=np.column_stack((unit(np.cross(-normal,finger)),-normal,finger));left_W=local_target@self.left_basis.T;approach=minimum_jerk(t/2.);WL=self.left_initial_R@Rotation.from_rotvec(approach*Rotation.from_matrix(self.left_initial_R.T@left_W).as_rotvec()).as_matrix();self.desired_left_R=WL.copy()
        guide_center=(.85*ball+.15*center)-normal*(.121+self.left_capsule_radius+c['clearance']-self.guide_depth);target_left=guide_center-left_W@r['left_offset']
        if guide and self.rotation_start is not None:
            bv=int(actual.model.joint('basketball_free').dofadr[0]);target_left+=np.clip(.2*actual.qvel[bv:bv+3],-.025,.025)
        if not guide:target_left=ball+r['side']*.32-left_W@r['left_offset']
        if self.phase=='E' and self.exp.get('probe',False):target_left+=r['side']*.05*minimum_jerk(raise_elapsed-5)
        self.cartesian_left=(1-approach)*self.left_initial_p+approach*target_left
        q[self.left_finger_ids]=(1-approach)*base[self.left_finger_ids]
        if guide:
            for item in self.left_mapping:
                if item['role']=='flexion':q[int(self.m.joint(item['joint']).qposadr[0])]+=approach*c['finger_flex']*item['sign']
        if self.ik_time is None or t-self.ik_time>=.033-1e-9:
            background=actual.qpos[:self.m.nq].copy();seedR=self.filtered_q[self.adrs] if self.filtered_q is not None else background[self.adrs];seedL=self.filtered_q[self.left_ids] if self.filtered_q is not None else background[self.left_ids]
            if self.rotation_start is not None and (c['trunk_extension_deg']>0 or c['wrist_extension_deg']>0):
                tilt0=np.arccos(np.clip(self.nominal_trunk[2],-1,1));tilt=max(0,tilt0-np.radians(c['trunk_extension_deg'])*body_blend);trunk_goal=unit(unit(np.r_[self.nominal_trunk[:2],0.])*np.sin(tilt)+np.array([0.,0.,np.cos(tilt)]));combo_ids=np.r_[self.trunk_ids,self.adrs];right,zR=whole_body_ik(self.m,background,self.cartesian_right,W,trunk_goal,self.body_calibration,np.radians(c['wrist_extension_deg'])*body_blend,seed=self.filtered_q[combo_ids]);self.desired_trunk_axis=trunk_goal;self.desired_extension_rad=np.radians(c['wrist_extension_deg'])*body_blend;self.fitted_trunk=right[self.trunk_ids];self.fitted_head=head_ik(self.m,right,r['rim_center'],self.filtered_q[self.head_ids])[self.head_ids]
            else:right,zR=pose_ik(self.m,background,'R',self.cartesian_right,W,seedR)
            left_background=background.copy()
            if hasattr(self,'fitted_trunk'):left_background[self.trunk_ids]=self.fitted_trunk
            left,zL=pose_ik(self.m,left_background,'L',self.cartesian_left,WL,seedL);self.fitted_right=right[self.adrs];self.fitted_left=left[self.left_ids];self.jac_reports={'R':zR,'L':zL};self.ik_time=t
        q[self.adrs]=self.fitted_right;q[self.left_ids]=self.fitted_left
        if hasattr(self,'fitted_trunk'):q[self.trunk_ids]=self.fitted_trunk;q[self.head_ids]=self.fitted_head
        if self.phase=='E' and c.get('probe',False):
            if self.pulse_dirs is None:self.pulse_dirs=reserve(actual.model,actual,r['rel0'])
            s=minimum_jerk((raise_elapsed-5)/.6);q[self.adrs[3:6]]+=.06*s*self.pulse_dirs['extension_direction'];q[self.adrs[6:9]]+=.04*s*self.flexion_direction(actual)
        # Limits/actual FK determine the solution; filter prevents branch jumps.
        acceleration=np.clip(64*(q-self.filtered_q)-16*self.filtered_v,-8.,8.);self.filtered_v=np.clip(self.filtered_v+acceleration*dt,-1.5,1.5);self.filtered_q+=self.filtered_v*dt;self.unfiltered=q.copy();return self.filtered_q.copy()


    def flexion_direction(self,actual):
        probe=mujoco.MjData(actual.model);probe.qpos[:]=actual.qpos;mujoco.mj_kinematics(actual.model,probe);base=extension_angle(actual.model,probe,self.body_calibration);derivatives=[]
        for adr in self.adrs[6:9]:
            old=probe.qpos[adr];probe.qpos[adr]=old+1e-5;mujoco.mj_kinematics(actual.model,probe);derivatives.append((extension_angle(actual.model,probe,self.body_calibration)-base)/1e-5);probe.qpos[adr]=old
        return -unit(np.array(derivatives))


class RotationEnv(CoordinatedEnv):
    def __init__(self,condition='C',yaw=-150,orbit=1.,support_tilt=30,guide_force=2.,finger_flex=.25,rotation_seconds=5.,probe=True,gui=False,clearance=.035,guide_finger_elevation=90,trunk_extension_deg=0,wrist_extension_deg=0,proximity_guide=False):
        if condition not in 'ABCD' or not np.isfinite(yaw) or abs(yaw)>180 or not 0<=orbit<=1 or not 0<=support_tilt<=60 or not .2<=guide_force<=6 or not 0<=finger_flex<=.7 or not 2<=rotation_seconds<=8:raise ValueError('Invalid experiment bounds')
        if not 0<=trunk_extension_deg<=15 or not 0<=wrist_extension_deg<=35 or not 0<=guide_finger_elevation<=90 or not 0<=clearance<=.04:raise ValueError('Invalid geometric target bounds')
        self.experiment=dict(condition=condition,yaw=yaw,orbit=orbit,support_tilt=support_tilt,guide_force=guide_force,finger_flex=finger_flex,rotation_seconds=rotation_seconds,probe=probe,clearance=clearance,guide_finger_elevation=guide_finger_elevation,trunk_extension_deg=trunk_extension_deg,wrist_extension_deg=wrist_extension_deg,proximity_guide=proximity_guide);self.telemetry=[];self.points=[];self.frames=[];self.frame_times=[];self.targets=[];self.control_rows=[];self.no_contact_since=None;self.start_finger=None;self.initial_ball_rotation=None
        super().__init__(guide='off',tilt_deg=0,probe=False,ball_feedback=False,prep_seconds=3.5,render_mode='human' if gui else None,record=False);self.max_seconds=18.;self.reset(seed=7)
    def _adapter(self):
        if not self.coord_ready:return super()._adapter()
        self.bridge=lift.HumanoidControllerBridge(MODELS/'body_01.xml',self.m,SOURCE);self._upper_cache_time=None;self.adapter=RotationTargets(self.bridge,self.candidate,self.arm_ids,self.ref,4,self.condition,config=self.config,experiment=self.experiment)
    def stage_success(self,z):return False
    def reset(self,*args,**kwargs):
        out=super().reset(*args,**kwargs)
        if self.coord_ready:
            self.telemetry=[];self.points=[];self.frames=[self.d.qpos.copy()];self.frame_times=[0.];self.targets=[];self.control_rows=[];self.no_contact_since=None;self.start_finger=None;self.initial_landmarks={n:self.d.xpos[self.m.body(n).id].copy() for n in ('R_Shoulder','R_Elbow','R_Wrist')};self.initial_palm=-self.d.xmat[self.wrist].reshape(3,3)[:,1].copy();self.spin_path=0.;self.previous_ball_rotation=None;self.initial_ball_rotation=Rotation.from_quat(self.d.qpos[self.ball_q+3:self.ball_q+7][[1,2,3,0]])
        return out
    def sample(self):
        d,m=self.d,self.m;a=self.adapter;ball=d.qpos[self.ball_q:self.ball_q+3];goal=getattr(a,'ball_target',self.ref['ball_initial']);R=d.xmat[self.wrist].reshape(3,3);finger=unit(d.xpos[m.body('R_Middle1').id]-d.xpos[self.wrist]);normal=-R[:,1]
        if self.start_finger is None and d.time>=2.9:self.start_finger=finger.copy()
        initial=getattr(a,'finger_at_rotation_start',self.start_finger if self.start_finger is not None else palm_basis(m,d,'R')[1]);target_R=getattr(a,'desired_right_R',R);probe=mujoco.MjData(m);probe.qpos[:]=d.qpos;probe.qpos[self.arm_ids]=self.last_target[self.arm_ids];mujoco.mj_kinematics(m,probe);target_finger=target_R@unit(R.T@(d.xpos[m.body('R_Middle1').id]-d.xpos[self.wrist]));w,points=contact_wrenches(m,d);rel,rv=local_ball(m,d,'R');predicted=d.xpos[self.wrist]+R@self.rel0
        z=self.last_metrics;cal=a.body_calibration if hasattr(a,'body_calibration') else calibration(m,d);swing,twist=forearm_swing_twist(m,d,cal);trunk=trunk_axis(m,d);row=dict(wrist_extension_delta_deg=float(np.degrees(extension_angle(m,d,cal))),forearm_swing_deg=swing,forearm_twist_deg=twist,trunk_tilt_deg=float(np.degrees(np.arccos(np.clip(trunk[2],-1,1)))),calibration_used=True);row.update(time_s=float(d.time),phase=a.phase,finger_change_deg=float(np.degrees(np.arccos(np.clip(initial@finger,-1,1)))),finger_target_error_deg=float(np.degrees(np.arccos(np.clip(target_finger@finger,-1,1)))),wrist_orientation_goal_error_deg=float(np.degrees(Rotation.from_matrix(target_R@R.T).magnitude())),right_contacts=w['R']['count'],left_contacts=w['L']['count'],left_normal_N=w['L']['normal'],right_normal_N=w['R']['normal'],palm_up_dot=float(normal[2]),palm_rim_dot=float(normal@unit(scene_rim(m,d)-ball)),ball_path_error_mm=float(np.linalg.norm(ball-goal)*1000),right_local_change_mm=float(np.linalg.norm(rel-self.rel0)*1000),right_local_target_error_mm=float(np.linalg.norm(ball-predicted)*1000),other_contact_force_N=float(np.linalg.norm(w['other']['force'])),ball_rotation_deg=float(np.degrees((Rotation.from_quat(d.qpos[self.ball_q+3:self.ball_q+7][[1,2,3,0]])*self.initial_ball_rotation.inv()).magnitude())),right_slip_m_s=side_contacts(m,d,'R')['slip'],left_slip_m_s=side_contacts(m,d,'L')['slip'],com_margin_mm=z['com_margin_m']*1000,both_feet=bool(z['both_feet']),head_error_deg=z['head_error_deg'],fallen=bool(z['fallen']),warning_count=sum(z['warnings']),right_wrist_tracking_mm=z['R_target_actual_mm'],left_wrist_tracking_mm=z['L_target_actual_mm'])
        actual_rot=Rotation.from_quat(d.qpos[self.ball_q+3:self.ball_q+7][[1,2,3,0]])
        if self.previous_ball_rotation is not None:self.spin_path+=float((actual_rot*self.previous_ball_rotation.inv()).magnitude())
        self.previous_ball_rotation=actual_rot
        row['ball_spin_path_deg']=float(np.degrees(self.spin_path));row['trunk_extension_delta_deg']=float(np.degrees(np.arccos(np.clip(cal['trunk_axis'][2],-1,1)))-row['trunk_tilt_deg']);row['wrist_extension_goal_deg']=float(np.degrees(getattr(a,'desired_extension_rad',0.)))
        row['palm_normal_change_deg']=float(np.degrees(np.arccos(np.clip(self.initial_palm@normal,-1,1))))
        row['arm_tracking_rms_rad']=float(np.sqrt(np.mean((self.last_target[self.arm_ids]-d.qpos[self.arm_ids])**2)))
        limited=m.actuator_ctrllimited.astype(bool);span=np.maximum(abs(m.actuator_ctrlrange).max(axis=1),1e-12);row['actuator_saturation_fraction']=float(np.mean(abs(d.ctrl)>=self.bridge.controller.limit-1e-7))
        jid=[m.joint(n).id for n in self.joint_names];vals=d.qpos[self.ids];limits=m.jnt_range[jid];row['joint_limit_proximity_count']=int(np.sum(np.minimum(vals-limits[:,0],limits[:,1]-vals)<.02))
        for n,p0 in self.initial_landmarks.items():
            row[n+'_translation_mm']=float(np.linalg.norm(d.xpos[m.body(n).id]-p0)*1000)
        for ax,x in zip('xyz',normal):row['palm_normal_'+ax]=float(x)
        for side in ('R','L'):
            for kind in ('force','torque'):
                for ax,x in zip('xyz',w[side][kind]):row[side+'_'+kind+'_'+ax]=float(x)
            row[side+'_ik_position_error_mm']=a.jac_reports.get(side,{}).get('position_error_mm',0.);row[side+'_ik_orientation_error_deg']=a.jac_reports.get(side,{}).get('orientation_error_deg',0.)
        for ax,x in zip('xyz',ball):row['ball_'+ax]=float(x)
        for ax,x in zip('xyz',goal):row['ball_goal_'+ax]=float(x)
        for ax,x in zip('xyz',d.qvel[self.ball_v:self.ball_v+3]):row['ball_velocity_'+ax]=float(x)
        total=sum((x['force'] for x in w.values()),np.zeros(3))+.6*m.opt.gravity;torque=sum((x['torque'] for x in w.values()),np.zeros(3))
        for ax,x in zip('xyz',total):row['net_force_'+ax]=float(x)
        for ax,x in zip('xyz',torque):row['net_torque_'+ax]=float(x)
        self.telemetry.append(row);self.points.extend(points);return row
    def physical_step(self):
        reason=None
        for _ in range(self.substeps):
            before=self.d.qpos[self.ball_q:self.ball_q+7].copy();v=self.d.qvel[self.ball_v:self.ball_v+6].copy();ref=self.adapter.target(self.d.time,self.d);self.last_reference=ref;self.last_target=ref.copy();self.adapter.control(self.d,ref)
            assert np.array_equal(before,self.d.qpos[self.ball_q:self.ball_q+7]) and np.array_equal(v,self.d.qvel[self.ball_v:self.ball_v+6]) and self.m.neq==0 and not self.d.xfrc_applied.any() and not self.d.qfrc_applied.any()
            mujoco.mj_step(self.m,self.d);mujoco.mj_forward(self.m,self.d);self.last_metrics=self.metrics();z=self.sample()
            if z['right_contacts']+z['left_contacts']>0:self.no_contact_since=None
            elif self.no_contact_since is None:self.no_contact_since=self.d.time
            if not np.isfinite(self.d.qpos).all() or not np.isfinite(self.d.qvel).all() or z['warning_count']:reason='nonfinite_or_physics_warning'
            elif z['fallen']:reason='humanoid_fall'
            elif z['ball_path_error_mm']>200:reason='ball_world_path_tracking_error'
            elif self.no_contact_since is not None and self.d.time-self.no_contact_since>.15:reason='both_hands_contact_loss'
            elif self.d.qpos[self.ball_q+2]<self.ref['ball_initial'][2]-.25:reason='ball_drop'
            elif z['left_normal_N']>25 or z['right_normal_N']>45:reason='excessive_contact_load'
            elif self.experiment['condition'] in ('C','D') and self.d.time>=5 and self.adapter.rotation_start is None:reason='bimanual_preparation_gate_failed'
            if reason:break
        self.frames.append(self.d.qpos.copy());self.frame_times.append(float(self.d.time));self.targets.append(self.last_reference.copy());row=dict(time_s=float(self.d.time))
        for n,adr in zip(self.joint_names,self.ids):row[n+'_target']=float(self.last_reference[adr]);row[n+'_actual']=float(self.d.qpos[adr]);row[n+'_torque_Nm']=float(self.d.qfrc_actuator[int(self.m.joint(n).dofadr[0])])
        self.control_rows.append(row)
        if self.render_mode=='human':self.render()
        return reason


def run(name,**cfg):
    folder=OUT/name
    if folder.exists():raise FileExistsError(f'Preserve previous run: {folder}')
    folder.mkdir(parents=True);gui=cfg.pop('gui',False);speed=cfg.pop('speed',1.);e=RotationEnv(gui=gui,**cfg);initial=e.d.qpos.copy();start=time.perf_counter();reason=None
    try:
        while e.d.time<e.max_seconds:
            wall=time.perf_counter();reason=e.physical_step()
            if reason:break
            if e.adapter.rotation_finished is not None and e.d.time>e.adapter.rotation_finished+6.2:reason='sequence_completed';break
            if e.experiment['condition']=='A' and e.d.time>=10.8:reason='sequence_completed';break
            if gui and e.viewer and not e.viewer.is_running():reason='viewer_closed';break
            if gui and speed>0:time.sleep(max(0,e.control_dt/speed-(time.perf_counter()-wall)))
        data=e.telemetry;rot=[r for r in data if r['phase'] in ('B','C')];late=[r for r in data if r['time_s']>e.d.time-.5];geometric=bool(rot and max(r['finger_change_deg'] for r in rot)>=135 and np.mean([r['wrist_orientation_goal_error_deg'] for r in late])<20)
        bimanual=bool(rot and np.mean([r['left_contacts']>0 for r in rot])>=.5 and max(r['left_normal_N'] for r in rot)<12 and not any(r['ball_path_error_mm']>80 for r in rot));raising=bool(e.experiment['condition'] in ('A','D') and reason=='sequence_completed' and max(r['ball_z'] for r in data)-initial[e.ball_q+2]>.4);ready=False
        if reason=='sequence_completed':
            rr=reserve(e.m,e.d,e.ref['rel0']);ready=bool(raising and rr['elbow_extension_reserve_deg']>5 and rr['wrist_rim_rotation_reserve_rad']>.1 and np.mean([r['head_error_deg'] for r in late])<15 and max(r['ball_path_error_mm'] for r in late)<50 and all(r['com_margin_mm']>0 and r['both_feet'] for r in late))
        summary=dict(name=name,config=e.experiment,duration_s=float(e.d.time),reason=reason or 'time_limit',rotation_start_s=e.adapter.rotation_start,rotation_end_s=e.adapter.rotation_finished,geometric_rotation_success=geometric,bimanual_holding_success=bimanual,raising_success=raising,legacy_readiness_gate=ready,shooting_ready_success=bool(ready and geometric and bimanual),probe_executed=e.adapter.pulse_dirs is not None,release_success=False,large_ppo_allowed=False,max_finger_change_deg=max(r['finger_change_deg'] for r in data),final_finger_change_deg=data[-1]['finger_change_deg'],final_finger_error_deg=data[-1]['finger_target_error_deg'],ball_world_path_peak_mm=max(r['ball_path_error_mm'] for r in data),planned_ball_excursion_peak_mm=max(float(np.linalg.norm(np.array([r['ball_goal_'+a] for a in 'xyz'])-initial[e.ball_q:e.ball_q+3]))*1000 for r in data),other_contact_force_peak_N=max(r['other_contact_force_N'] for r in data),ball_local_change_peak_mm=max(r['right_local_change_mm'] for r in data),ball_rise_mm=(max(r['ball_z'] for r in data)-initial[e.ball_q+2])*1000,left_contact_ratio=float(np.mean([r['left_contacts']>0 for r in data])),left_rotation_contact_ratio=float(np.mean([r['left_contacts']>0 for r in rot])) if rot else 0.,left_force_peak_N=max(r['left_normal_N'] for r in data),right_vertical_mean_N=float(np.mean([r['R_force_z'] for r in data])),left_vertical_mean_N=float(np.mean([r['L_force_z'] for r in data])),ball_torque_rms_Nm=float(np.sqrt(np.mean([sum(r['net_torque_'+a]**2 for a in 'xyz') for r in data]))),ball_rotation_peak_deg=max(r['ball_rotation_deg'] for r in data),min_com_margin_mm=min(r['com_margin_mm'] for r in data),both_feet_ratio=float(np.mean([r['both_feet'] for r in data])),fallen=any(r['fallen'] for r in data),warnings=[int(w.number) for w in e.d.warning],neq=e.m.neq,ball_writes_after_reset=0,wall_seconds=time.perf_counter()-start)
        summary.update(wrist_extension_peak_deg=max(r['wrist_extension_delta_deg'] for r in data),trunk_extension_peak_deg=max(r['trunk_extension_delta_deg'] for r in data),forearm_twist_peak_deg=max(abs(r['forearm_twist_deg']) for r in data),palm_normal_change_peak_deg=max(r['palm_normal_change_deg'] for r in data),head_error_late_mean_deg=float(np.mean([r['head_error_deg'] for r in late])),arm_tracking_rmse_rad=float(np.sqrt(np.mean([r['arm_tracking_rms_rad']**2 for r in data]))),actuator_saturation_ratio=float(np.mean([r['actuator_saturation_fraction'] for r in data])),ball_spin_path_deg=data[-1]['ball_spin_path_deg'],left_first_contact_s=next((r['time_s'] for r in data if r['left_contacts']>0),None),left_raising_contact_ratio=float(np.mean([r['left_contacts']>0 for r in data if r['phase']=='C'])) if e.experiment['condition'] in ('A','D') and any(r['phase']=='C' for r in data) else None,physical_reference_valid=bool(ready and geometric and bimanual),R_shoulder_translation_peak_mm=max(r['R_Shoulder_translation_mm'] for r in data),R_elbow_translation_peak_mm=max(r['R_Elbow_translation_mm'] for r in data))
        summary['wrist_extension_verified']=bool(e.experiment['wrist_extension_deg']>0 and summary['wrist_extension_peak_deg']>5);summary['trunk_extension_verified']=bool(e.experiment['trunk_extension_deg']>0 and summary['trunk_extension_peak_deg']>3)
        (folder/'curriculum.json').write_text(json.dumps(curriculum(summary),indent=2))
        write_csv(folder/'physics.csv',data);write_csv(folder/'contacts.csv',e.points);write_csv(folder/'joint_targets_torques.csv',e.control_rows);np.savez_compressed(folder/'states.npz',qpos=e.frames,time_s=e.frame_times);np.savez_compressed(folder/'reference.npz',target_qpos=e.targets,time_s=e.frame_times[1:],fps=30.,joint_names=e.joint_names,source='contact-only experimental target reference; not ground truth');(folder/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True);return summary
    finally:e.close()


def audit():
    before=json.loads((OUT/'preservation_before.json').read_text());changed=[p for p,h in before.items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];z=dict(checked_files=len(before),changed_scientific_files=[p for p in changed if Path(p).name!='.DS_Store'],changed_os_metadata=[p for p in changed if Path(p).name=='.DS_Store']);(OUT/'preservation_audit.json').write_text(json.dumps(z,indent=2));assert not z['changed_scientific_files'];return z


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='manual');p.add_argument('--condition',choices=list('ABCD'),default='C');p.add_argument('--yaw',type=float,default=-150);p.add_argument('--orbit',type=float,default=1);p.add_argument('--proximity-guide',action='store_true');p.add_argument('--trunk-extension',type=float,default=0);p.add_argument('--wrist-extension',type=float,default=0);p.add_argument('--guide-finger-elevation',type=float,default=90);p.add_argument('--clearance',type=float,default=.035);p.add_argument('--support-tilt',type=float,default=30);p.add_argument('--guide-force',type=float,default=2);p.add_argument('--finger-flex',type=float,default=.25);p.add_argument('--rotation-seconds',type=float,default=5);p.add_argument('--no-probe',action='store_true');p.add_argument('--gui',action='store_true');p.add_argument('--speed',type=float,default=1);p.add_argument('--body-screen',action='store_true');p.add_argument('--screen',action='store_true');p.add_argument('--audit',action='store_true');a=p.parse_args()
    if a.body_screen:return body_screen()
    if a.screen:return screen()
    if a.audit:print(audit());return
    if not np.isfinite(a.speed) or a.speed<0:p.error('speed must be finite and >=0')
    run(a.name,proximity_guide=a.proximity_guide,probe=not a.no_probe,condition=a.condition,yaw=a.yaw,orbit=a.orbit,support_tilt=a.support_tilt,guide_force=a.guide_force,finger_flex=a.finger_flex,rotation_seconds=a.rotation_seconds,gui=a.gui,speed=a.speed,clearance=a.clearance,guide_finger_elevation=a.guide_finger_elevation,trunk_extension_deg=a.trunk_extension,wrist_extension_deg=a.wrist_extension)
if __name__=='__main__':main()
