"""Independent continuous contact-only shooting trajectory; old experiments untouched."""
import sys,json,argparse,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from examples.m5_bimanual_reorientation import RotationEnv,palm_basis,pose_ik
from examples.m5_coordinated_reference import CoordinatedTargets,lift,MODELS,SOURCE,unit,reserve
from examples.m5_upper_body_ik import joint_ids,minimum_jerk,anatomical_mapping
from examples.m5_bimanual_contact import side_contacts,hand_geom,gap_to_ball
from examples.m5_bimanual_body_ik import calibration,extension_angle,trunk_axis,head_ik
from examples.m3_control import write_csv
OUT=ROOT/'outputs/m5_1_motion_redesign'

def shooting_progress(t,duration,lag=0.,separated=False):
    if duration<=0 or abs(lag)>duration/2:raise ValueError('Invalid duration/lag')
    return dict(rise=minimum_jerk((t-duration if separated else t)/duration),rotation=minimum_jerk((t-lag)/duration))

def so3_interpolate(start,end,u):
    return start@Rotation.from_rotvec(float(np.clip(u,0,1))*Rotation.from_matrix(start.T@end).as_rotvec()).as_matrix()

def trajectory_body_ik(m,q,position,orientation,trunk_target,cal,extension,seed=None,iterations=35,soft=False):
    names=[f'{b}_{a}' for b in ('Torso','Spine','Chest') for a in 'xyz']+[f'R_{b}_{a}' for b in ('Shoulder','Elbow','Wrist') for a in 'xyz'];ids=joint_ids(m,names);dofs=[int(m.joint(n).dofadr[0]) for n in names];bounds=np.array([m.jnt_range[m.joint(n).id] for n in names]);lo,hi=bounds[:,0]+1e-5,bounds[:,1]-1e-5;lo[:9]=np.maximum(lo[:9],cal['trunk_q']-.4);hi[:9]=np.minimum(hi[:9],cal['trunk_q']+.4);p=mujoco.MjData(m);p.qpos[:]=q;p.qpos[ids]=np.clip(q[ids] if seed is None else seed,lo,hi);anchor=p.qpos[ids].copy();w=m.body('R_Wrist').id;penalty=np.diag([.06]*9+[.018]*6+[.025]*3)
    ew=.15 if cal.get("soft",False) else 1.8
    for _ in range(iterations):
        mujoco.mj_kinematics(m,p);mujoco.mj_comPos(m,p);axis=trunk_axis(m,p);dp=position-p.xpos[w];dr=Rotation.from_matrix(orientation@p.xmat[w].reshape(3,3).T).as_rotvec();de=float((extension-extension_angle(m,p,cal)+np.pi)%(2*np.pi)-np.pi);jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv));mujoco.mj_jacBody(m,p,jp,jr,w)
        js=np.zeros((3,m.nv));jh=np.zeros((3,m.nv))
        for b in ('L_Shoulder','R_Shoulder'):
            temp=np.zeros((3,m.nv));rot=np.zeros((3,m.nv));mujoco.mj_jacBody(m,p,temp,rot,m.body(b).id);js+=temp*.5
        for b in ('L_Hip','R_Hip'):
            temp=np.zeros((3,m.nv));rot=np.zeros((3,m.nv));mujoco.mj_jacBody(m,p,temp,rot,m.body(b).id);jh+=temp*.5
        hs=np.mean([p.xpos[m.body(n).id] for n in ('L_Hip','R_Hip')],axis=0);ss=np.mean([p.xpos[m.body(n).id] for n in ('L_Shoulder','R_Shoulder')],axis=0);ja=(np.eye(3)-np.outer(axis,axis))@(js-jh)/max(np.linalg.norm(ss-hs),1e-8)
        v=np.cross(cal['trunk_axis'],trunk_target);angle=np.arccos(np.clip(cal['trunk_axis']@trunk_target,-1,1));trunk_rotation=(Rotation.from_rotvec(unit(v)*angle).as_matrix() if np.linalg.norm(v)>1e-8 else np.eye(3))@cal['chest_rotation'];chest=m.body('Chest').id;jch=np.zeros((3,m.nv));jcp=np.zeros((3,m.nv));mujoco.mj_jacBody(m,p,jcp,jch,chest);dc=Rotation.from_matrix(trunk_rotation@p.xmat[chest].reshape(3,3).T).as_rotvec()
        # Finite FK derivative of the calibrated extension component, wrist axes only.
        je=np.zeros(len(ids));base_ext=extension_angle(m,p,cal)
        for k in range(15,18):
            old=p.qpos[ids[k]];p.qpos[ids[k]]=old+1e-5;mujoco.mj_kinematics(m,p);je[k]=((extension_angle(m,p,cal)-base_ext+np.pi)%(2*np.pi)-np.pi)/1e-5;p.qpos[ids[k]]=old
        mujoco.mj_kinematics(m,p)
        J=np.vstack((35*jp[:,dofs],2*jr[:,dofs],5*ja[:,dofs],ew*je,2*jch[:,dofs]));err=np.r_[35*dp,2*dr,5*(trunk_target-axis),ew*de,2*dc];step=np.linalg.solve(J.T@J+penalty,J.T@err-penalty@(p.qpos[ids]-anchor));step*=min(1,.18/max(np.linalg.norm(step),1e-12));p.qpos[ids]=np.clip(p.qpos[ids]+step,lo,hi)
        if np.linalg.norm(dp)<.001 and np.linalg.norm(dr)<.02 and abs(de)<.02 and np.linalg.norm(trunk_target-axis)<.02:break
    mujoco.mj_kinematics(m,p);z=dict(position_error_mm=float(np.linalg.norm(position-p.xpos[w])*1000),orientation_error_deg=float(np.degrees(Rotation.from_matrix(orientation@p.xmat[w].reshape(3,3).T).magnitude())),extension_error_deg=float(np.degrees((extension-extension_angle(m,p,cal)+np.pi)%(2*np.pi)-np.pi)),trunk_error_deg=float(np.degrees(np.arccos(np.clip(trunk_target@trunk_axis(m,p),-1,1)))))
    return p.qpos.copy(),z



class TrajectoryTargets(CoordinatedTargets):
    def __init__(self,*args,experiment,**kwargs):
        super().__init__(*args,**kwargs);self.exp=experiment;self.rotation_start=None;self.rotation_finished=None;self.bimanual_since=None;self.phase='A';self.ik_time=None;self.guide_depth=0.;self.force_filtered=0.;self.jac_reports={};self.pulse_dirs=None;self.last_t=0;self.left_lost_since=None
    def target(self,t,actual=None):
        if t<=0 or actual is None:return super().target(t,actual)
        dt=max(t-self.last_t,self.m.opt.timestep);self.last_t=t;c=self.exp;r=self.reference;kind=c['trajectory_condition'];guide=kind=='D';L=side_contacts(actual.model,actual,'L');Rcontact=side_contacts(actual.model,actual,'R');ball=actual.qpos[int(actual.model.joint('basketball_free').qposadr[0]):][:3]
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
            if not guide and t>=c['prep_seconds']:self.rotation_start=t;self.finger_at_rotation_start=unit(actual.xpos[actual.model.body('R_Middle1').id]-actual.xpos[actual.model.body('R_Wrist').id])
            elif guide and self.bimanual_since is not None and t-self.bimanual_since>=.25:self.rotation_start=t;self.finger_at_rotation_start=unit(actual.xpos[actual.model.body('R_Middle1').id]-actual.xpos[actual.model.body('R_Wrist').id])
        elapsed=0 if self.rotation_start is None else max(0,t-self.rotation_start)
        trajectory=shooting_progress(elapsed,c['motion_seconds'],c['lag_seconds'],kind=='A')
        u=trajectory['rotation'];progress=trajectory['rise']
        self.rise_progress=progress;self.rotation_progress=u
        self.phase='A' if self.rotation_start is None else 'C' if progress<1 or u<1 else 'D'
        if self.rotation_finished is None and self.rotation_start is not None and u>=1:self.rotation_finished=t
        raise_elapsed=elapsed
        base=lift.LiftingTargets.target(self,0,actual);q=base.copy();align=minimum_jerk(t/2);q[self.trunk_ids]=(1-align)*base[self.trunk_ids]+align*r['aligned'][self.trunk_ids];q[self.head_ids]=(1-align)*base[self.head_ids]+align*r['aligned'][self.head_ids]
        # A compares preserved -15/-5 grip; B/C/D start there before turning.
        if not hasattr(self,'body_calibration'):
            self.body_calibration=calibration(actual.model,actual);self.nominal_trunk=self.body_calibration['trunk_axis'].copy()
        body_blend=progress if kind in ('C','D') else 0.
        initial=r['right_rotation'].copy()
        if c.get('stable_grip',False):initial=Rotation.from_rotvec(r['side']*np.radians(-5)).as_matrix()@Rotation.from_rotvec(-initial[:,1]*np.radians(-15)).as_matrix()@initial
        prepared=so3_interpolate(r['right_rotation'],initial,minimum_jerk(t/1.5))
        W=Rotation.from_rotvec((-initial[:,1])*np.radians(c['yaw'])*u).as_matrix()@prepared
        if c['wrist_extension_deg']>0:
            f_local=self.body_calibration['finger_local'];ext_axis=unit(np.cross(W@f_local,-W[:,1]));W=Rotation.from_rotvec(-ext_axis*np.radians(c['wrist_extension_deg'])*body_blend).as_matrix()@W
        # Desired ball path only: never assigned to simulation qpos/qvel.
        goal=r['ball_goal_shoot'].copy() if c.get('face_path',False) else r['ball_initial']+c['height_m']*np.array([0.,0.,1.])+c['forward_m']*r['forward']
        center=(1-progress)*r['ball_initial']+progress*goal
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
        if False and self.exp.get('probe',False):target_left+=r['side']*.05*minimum_jerk(raise_elapsed-5)
        self.cartesian_left=(1-approach)*self.left_initial_p+approach*target_left
        q[self.left_finger_ids]=(1-approach)*base[self.left_finger_ids]
        if guide:
            for item in self.left_mapping:
                if item['role']=='flexion':q[int(self.m.joint(item['joint']).qposadr[0])]+=approach*c['finger_flex']*item['sign']
        if self.ik_time is None or t-self.ik_time>=.033-1e-9:
            background=actual.qpos[:self.m.nq].copy();seedR=self.filtered_q[self.adrs] if self.filtered_q is not None else background[self.adrs];seedL=self.filtered_q[self.left_ids] if self.filtered_q is not None else background[self.left_ids]
            if self.rotation_start is not None and (c['trunk_extension_deg']>0 or c['wrist_extension_deg']>0):
                tilt0=np.arccos(np.clip(self.nominal_trunk[2],-1,1));tilt=max(0,tilt0-np.radians(c['trunk_extension_deg'])*body_blend);trunk_goal=unit(unit(np.r_[self.nominal_trunk[:2],0.])*np.sin(tilt)+np.array([0.,0.,np.cos(tilt)]));combo_ids=np.r_[self.trunk_ids,self.adrs];self.body_calibration["soft"]=c.get("soft",False);right,zR=trajectory_body_ik(self.m,background,self.cartesian_right,W,trunk_goal,self.body_calibration,np.radians(c['wrist_extension_deg'])*body_blend,seed=self.filtered_q[combo_ids]);self.desired_trunk_axis=trunk_goal;self.desired_extension_rad=np.radians(c['wrist_extension_deg'])*body_blend;self.fitted_trunk=right[self.trunk_ids];self.fitted_head=head_ik(self.m,right,r['rim_center'],self.filtered_q[self.head_ids])[self.head_ids]
            else:right,zR=pose_ik(self.m,background,'R',self.cartesian_right,W,seedR)
            left_background=background.copy()
            if hasattr(self,'fitted_trunk'):left_background[self.trunk_ids]=self.fitted_trunk
            left,zL=pose_ik(self.m,left_background,'L',self.cartesian_left,WL,seedL);self.fitted_right=right[self.adrs];self.fitted_left=left[self.left_ids];self.jac_reports={'R':zR,'L':zL};self.ik_time=t
        q[self.adrs]=self.fitted_right;q[self.left_ids]=self.fitted_left
        if hasattr(self,'fitted_trunk'):q[self.trunk_ids]=self.fitted_trunk;q[self.head_ids]=self.fitted_head
        if False and c.get('probe',False):
            if self.pulse_dirs is None:self.pulse_dirs=reserve(actual.model,actual,r['rel0'])
            s=minimum_jerk((raise_elapsed-5)/.6);q[self.adrs[3:6]]+=.06*s*self.pulse_dirs['extension_direction'];q[self.adrs[6:9]]+=.04*s*self.flexion_direction(actual)
        # Limits/actual FK determine the solution; filter prevents branch jumps.
        acceleration=np.clip(64*(q-self.filtered_q)-16*self.filtered_v,-8.,8.);self.filtered_v=np.clip(self.filtered_v+acceleration*dt,-1.5,1.5);self.filtered_q+=self.filtered_v*dt;self.unfiltered=q.copy();return self.filtered_q.copy()


    def flexion_direction(self,actual):
        probe=mujoco.MjData(actual.model);probe.qpos[:]=actual.qpos;mujoco.mj_kinematics(actual.model,probe);base=extension_angle(actual.model,probe,self.body_calibration);derivatives=[]
        for adr in self.adrs[6:9]:
            old=probe.qpos[adr];probe.qpos[adr]=old+1e-5;mujoco.mj_kinematics(actual.model,probe);derivatives.append((extension_angle(actual.model,probe,self.body_calibration)-base)/1e-5);probe.qpos[adr]=old
        return -unit(np.array(derivatives))



class TrajectoryEnv(RotationEnv):
    def __init__(self,condition='C',motion_seconds=3.,lag_seconds=.25,height_m=.4,forward_m=.06,yaw=-35.,gui=False,trunk_extension=8.,wrist_extension=10.,soft=False,stable_grip=False,path_limit_mm=200.,face_path=False,guide_tilt=0.,guide_force=.6):
        shooting_progress(0,motion_seconds,lag_seconds)
        if not np.isfinite(path_limit_mm) or not 100<=path_limit_mm<=350:raise ValueError('Path diagnostic limit 100..350 mm')
        if not 0<height_m<=.7 or not 0<=forward_m<=.3:raise ValueError('Path bounds')
        self.trajectory_config=dict(face_path=face_path,path_limit_mm=path_limit_mm,stable_grip=stable_grip,soft=soft,trajectory_condition=condition,motion_seconds=motion_seconds,lag_seconds=lag_seconds,height_m=height_m,forward_m=forward_m,prep_seconds=1.5)
        super().__init__(condition=condition,yaw=yaw,orbit=1.,support_tilt=guide_tilt,guide_force=guide_force,finger_flex=0.,rotation_seconds=3.,probe=False,gui=gui,clearance=.015,guide_finger_elevation=0,trunk_extension_deg=trunk_extension if condition in ('C','D') else 0,wrist_extension_deg=wrist_extension if condition in ('C','D') else 0,proximity_guide=True)
        self.max_seconds=15.
    def _adapter(self):
        if not self.coord_ready:return super()._adapter()
        self.experiment.update(self.trajectory_config)
        self.bridge=lift.HumanoidControllerBridge(MODELS/'body_01.xml',self.m,SOURCE);self._upper_cache_time=None
        self.adapter=TrajectoryTargets(self.bridge,self.candidate,self.arm_ids,self.ref,4,self.condition,config=self.config,experiment=self.experiment)
    def sample(self):
        row=super().sample();row['rise_progress']=getattr(self.adapter,'rise_progress',0.);row['rotation_progress']=getattr(self.adapter,'rotation_progress',0.)
        bid=self.m.body('R_Wrist').id;row['wrist_z']=float(self.d.xpos[bid,2])
        row['left_force_power_W']=sum(row['L_force_'+a]*row['ball_velocity_'+a] for a in 'xyz')
        # Store actual hinge qvel/qacc at control sampling rate in separate records.
        return row
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
            elif z['ball_path_error_mm']>self.experiment['path_limit_mm']:reason='ball_world_path_tracking_error'
            elif self.no_contact_since is not None and self.d.time-self.no_contact_since>.15:reason='both_hands_contact_loss'
            elif self.d.qpos[self.ball_q+2]<self.ref['ball_initial'][2]-.25:reason='ball_drop'
            elif z['left_normal_N']>25 or z['right_normal_N']>45:reason='excessive_contact_load'
            elif self.experiment['condition']=='D' and self.d.time>=5 and self.adapter.rotation_start is None:reason='bimanual_preparation_gate_failed'
            if reason:break
        self.frames.append(self.d.qpos.copy());self.frame_times.append(float(self.d.time));self.targets.append(self.last_reference.copy());row=dict(time_s=float(self.d.time))
        for n,adr in zip(self.joint_names,self.ids):row[n+'_target']=float(self.last_reference[adr]);row[n+'_actual']=float(self.d.qpos[adr]);row[n+'_torque_Nm']=float(self.d.qfrc_actuator[int(self.m.joint(n).dofadr[0])])
        self.control_rows.append(row)
        if self.render_mode=='human':self.render()
        return reason


def run_trajectory(name,**cfg):
    folder=OUT/name
    if folder.exists():raise FileExistsError(folder)
    folder.mkdir();e=TrajectoryEnv(**cfg);reason=None;start=time.perf_counter();initial=e.d.qpos.copy()
    try:
        while e.d.time<e.max_seconds:
            reason=e.physical_step()
            if reason:break
            a=e.adapter
            if a.rotation_start is not None and a.rise_progress>=1 and a.rotation_progress>=1 and e.d.time>a.rotation_start+e.experiment['motion_seconds']*(2 if cfg.get('condition')=='A' else 1)+max(0,e.experiment['lag_seconds'])+.6:
                reason='sequence_completed';break
        data=e.telemetry;p=np.array([r['time_s'] for r in data]);z=np.array([r['ball_z'] for r in data]);finger=np.array([r['finger_change_deg'] for r in data]);vz=np.gradient(z,p);vr=np.gradient(finger,p)
        mask=np.array([r['phase']!='A' for r in data]);overlap=mask&(vz>.01)&(abs(vr)>2.)
        q=np.array([[r[n+'_actual'] for n in e.joint_names] for r in e.control_rows]);target=np.array([[r[n+'_target'] for n in e.joint_names] for r in e.control_rows]);ts=np.array([r['time_s'] for r in e.control_rows]);stats={}
        for label,x in [('actual',q),('target',target)]:
            for order in ('velocity','acceleration','jerk'):
                x=np.gradient(x,ts,axis=0);stats[label+'_'+order+'_rms']=float(np.sqrt(np.mean(x*x)))
        requested_height=float(e.ref['ball_goal_shoot'][2]-e.ref['ball_initial'][2]) if e.experiment['face_path'] else e.experiment['height_m']
        late=data[-min(len(data),225):];rise=float((max(z)-initial[e.ball_q+2])*1000);valid=reason=='sequence_completed' and rise>requested_height*800 and max(r['ball_path_error_mm'] for r in late)<50 and all(r['com_margin_mm']>0 for r in late)
        s=dict(name=name,config=e.experiment,duration_s=float(e.d.time),reason=reason or 'time_limit',motion_start_s=e.adapter.rotation_start,ball_rise_mm=rise,wrist_rise_mm=float((max(r['wrist_z'] for r in data)-e.initial_landmarks['R_Wrist'][2])*1000),finger_change_peak_deg=max(finger),wrist_extension_peak_deg=max(r['wrist_extension_delta_deg'] for r in data),trunk_extension_peak_deg=max(r['trunk_extension_delta_deg'] for r in data),actual_overlap_seconds=float(np.sum(np.diff(p,prepend=p[0])*overlap)),actual_overlap_motion_fraction=float(np.mean(overlap[mask])) if mask.any() else 0.,left_contact_ratio=float(np.mean([r['left_contacts']>0 for r in data if r['phase']!='A'])) if mask.any() else 0.,right_contact_ratio=float(np.mean([r['right_contacts']>0 for r in data])),R_vertical_mean_N=float(np.mean([r['R_force_z'] for r in data])),L_vertical_mean_N=float(np.mean([r['L_force_z'] for r in data])),left_force_peak_N=max(r['left_normal_N'] for r in data),left_work_J=float(np.trapezoid([r['left_force_power_W'] for r in data],p)),path_peak_mm=max(r['ball_path_error_mm'] for r in data),local_movement_peak_mm=max(r['right_local_change_mm'] for r in data),slip_rms_mm_s=float(np.sqrt(np.mean([r['right_slip_m_s']**2 for r in data]))*1000),tracking_rmse_rad=float(np.sqrt(np.mean((q-target)**2))),com_min_mm=min(r['com_margin_mm'] for r in data),both_feet_ratio=float(np.mean([r['both_feet'] for r in data])),fallen=any(r['fallen'] for r in data),warnings=[int(w.number) for w in e.d.warning],ball_writes_after_reset=0,neq=e.m.neq,raising_gate_pass=bool(valid),shooting_ready_success=False,physical_reference_valid=False,large_ppo_allowed=False,wall_seconds=time.perf_counter()-start,**stats)
        write_csv(folder/'physics.csv',data);write_csv(folder/'contacts.csv',e.points);write_csv(folder/'joints.csv',e.control_rows);np.savez_compressed(folder/'states.npz',time_s=e.frame_times,qpos=e.frames);np.savez_compressed(folder/'reference.npz',time_s=e.frame_times[1:],target_qpos=e.targets,fps=30.,physical_reference_valid=False,representation='MuJoCo qpos; not SMPL axis-angle');(folder/'summary.json').write_text(json.dumps(s,indent=2));print(json.dumps(s),flush=True);return s
    finally:e.close()

def audit_new():
    old=json.loads((OUT/'preservation_before.json').read_text());changed=[p for p,h in old.items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];result=dict(files_checked=len(old),changed=changed);(OUT/'preservation_audit.json').write_text(json.dumps(result,indent=2));assert not changed;return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='trial');p.add_argument('--condition',choices=list('ABCD'),default='C');p.add_argument('--duration',type=float,default=3.);p.add_argument('--lag',type=float,default=.25);p.add_argument('--height',type=float,default=.4);p.add_argument('--forward',type=float,default=.06);p.add_argument('--turn',type=float,default=-35.);p.add_argument('--guide-tilt',type=float,default=0.);p.add_argument('--guide-force',type=float,default=.6);p.add_argument('--face-path',action='store_true');p.add_argument('--path-limit',type=float,default=200.);p.add_argument('--stable-grip',action='store_true');p.add_argument('--soft',action='store_true');p.add_argument('--trunk',type=float,default=8.);p.add_argument('--extension',type=float,default=10.);p.add_argument('--gui',action='store_true');p.add_argument('--audit',action='store_true');a=p.parse_args()
    if a.audit:print(audit_new())
    else:run_trajectory(a.name,condition=a.condition,motion_seconds=a.duration,lag_seconds=a.lag,height_m=a.height,forward_m=a.forward,yaw=a.turn,gui=a.gui,trunk_extension=a.trunk,wrist_extension=a.extension,soft=a.soft,stable_grip=a.stable_grip,path_limit_mm=a.path_limit,face_path=a.face_path,guide_tilt=a.guide_tilt,guide_force=a.guide_force)
