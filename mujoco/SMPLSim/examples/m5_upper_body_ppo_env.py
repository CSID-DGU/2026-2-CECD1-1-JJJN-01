"""M5-1 upper-body residual adapter. Existing M5-0 and PD+COM are unchanged."""
import copy
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from examples.m5_residual_ppo_env import BasketballResidualEnv, BASE, MODELS, SOURCE
from examples.m5_bimanual_contact import OUT, derived_scene, side_contacts, local_ball, gap_to_ball
from examples.m5_upper_body_ik import joint_ids, minimum_jerk, fit_wrist
from examples import m4_contact_lifting as lift
from examples.m3_control import FINGERS
from smpl_sim.physics.basketball import RIM_CENTER
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

STAGES={0:'Baseline',1:'Bimanual Holding',2:'Upper-body Alignment',3:'Intermediate Raising',4:'Shooting Position',5:'Shooting Pose Stabilization'}
DURATIONS={0:6.5,1:7.,2:11.,3:21.,4:25.,5:26.}


class UpperTargets(lift.LiftingTargets):
    def __init__(self,b,c,arm_ids,reference,stage,condition):
        ik=np.load(BASE/'ik_path.npz')
        super().__init__(b,c,ik['qpos'],ik['rotation'],arm_ids,'C',1.,4.)
        self.reference=reference;self.stage=stage;self.mode=condition
        self.left_ids=joint_ids(self.m,[n.replace('R_','L_',1) for n in lift.ARM])
        self.left_finger_ids=joint_ids(self.m,[f'L_{f}{part}_{a}' for f in FINGERS for part in (1,2,3) for a in 'xyz'])
        self.shooting_spline=CubicSpline(np.linspace(0,1,len(reference['shooting_path'])),reference['shooting_path'],axis=0,bc_type='clamped')
        self.trunk_ids=reference['trunk_ids'].astype(int);self.head_ids=reference['head_ids'].astype(int)
        probe=mujoco.MjData(self.m);probe.qpos[:]=reference['initial'][:self.m.nq];mujoco.mj_forward(self.m,probe)
        self.initial_left_R=probe.xmat[self.m.body('L_Wrist').id].reshape(3,3).copy()
        self.previous_goal=None;self.goal_velocity=None;self.goal_time=0.;self.unfiltered=None
        self.last_fit_time=None;self.fitted_left=None;self.fitted_right=None;self.cartesian_left=None;self.cartesian_right=None;self.guide_feedback=True;self.guide_clearance=.001;self.guide_start=1.
    def target(self,t,actual=None):
        if self.stage==0:return super().target(t,actual)
        if t<=0 and actual is not None:
            self.initial_left_p=actual.xpos[self.m.body('L_Wrist').id].copy();self.initial_right_p=actual.xpos[self.m.body('R_Wrist').id].copy()
            self.initial_left_R=actual.xmat[self.m.body('L_Wrist').id].reshape(3,3).copy();self.last_fit_time=None
        clock=max(0.,t-9.) if self.stage>=3 else 0.
        q=super().target(clock,actual)
        if self.mode=='A':return q
        r=self.reference
        alpha=minimum_jerk((t-self.guide_start)/4)
        q[self.left_ids]=(1-alpha)*r['initial'][self.left_ids]+alpha*r['approached'][self.left_ids]
        q[self.left_finger_ids]=(1-alpha)*r['initial'][self.left_finger_ids]+alpha*r['approached'][self.left_finger_ids]
        if self.stage>=2:
            s=minimum_jerk((t-6)/3)
            q[self.trunk_ids]=(1-s)*q[self.trunk_ids]+s*r['aligned'][self.trunk_ids]
            q[self.head_ids]=(1-s)*q[self.head_ids]+s*r['aligned'][self.head_ids]
            if self.guide_start<=1:q[self.left_ids]=(1-s)*q[self.left_ids]+s*r['aligned'][self.left_ids]
        if self.stage>=3:
            s=minimum_jerk((t-10)/4)
            if self.guide_start<=1:q[self.left_ids]=(1-s)*q[self.left_ids]+s*r['shooting_path'][0,self.left_ids]
        if self.stage>=3 and t>=15:
            # Stage3 ends at the intermediate midpoint; Stage4/5 traverse to face-above.
            fraction=minimum_jerk((t-15)/8)*(.5 if self.stage==3 else 1.)
            goal=self.shooting_spline(fraction)
            right_ids=self.adrs
            q[right_ids]=goal[right_ids]
            if self.guide_start<=1:q[self.left_ids]=goal[self.left_ids]
        if self.stage>=1 and actual is not None:
            # Preserve world hand directions after trunk/reference displacement.
            for side,ids,world in [('R',self.adrs,self.R),('L',self.left_ids,r['left_rotation'])]:
                if t<=1:continue
                if side=='R' and self.stage<2:continue
                if side=='L':
                    initial_R=self.initial_left_R
                    world=initial_R@Rotation.from_rotvec(minimum_jerk((t-self.guide_start)/4)*Rotation.from_matrix(initial_R.T@world).as_rotvec()).as_matrix()
                bid=self.m.body(side+'_Wrist').id
                parent=actual.xmat[self.m.body_parentid[bid]].reshape(3,3)
                q[ids[6:]]=lift.rotation_angles(parent.T@world,actual.qpos[ids[6:]])
        if self.guide_feedback and actual is not None and t>1:
            alpha=minimum_jerk((t-self.guide_start)/4)
            bq=int(actual.model.joint('basketball_free').qposadr[0])
            center=actual.qpos[bq:bq+3]
            measured=side_contacts(actual.model,actual,'L')['normal_force']
            # Position feedback follows the ball; measured force only adjusts the
            # commanded surface clearance. No external force is applied.
            nominal=self.guide_clearance
            clearance=np.clip(nominal+.002*(measured-.5),-.001,.025)
            side_center=center+[0,.121+float(self.m.geom('L_Wrist').size[0])+clearance,-.02]
            desired_left=(1-alpha)*self.initial_left_p+alpha*(side_center-r['left_rotation']@r['left_offset'])
            left_rotation=self.initial_left_R@Rotation.from_rotvec(alpha*Rotation.from_matrix(self.initial_left_R.T@r['left_rotation']).as_rotvec()).as_matrix()
            self.cartesian_left=desired_left.copy()
            desired_right=self.initial_right_p.copy()
            if self.stage>=3:
                desired_right=self.initial_right_p+[0,0,.25*minimum_jerk((t-10)/4)]
                if t>=15:
                    fraction=minimum_jerk((t-15)/8)*(.5 if self.stage==3 else 1.)
                    if fraction<.5:ball_goal=(1-2*fraction)*r['ball_goal_lift']+2*fraction*r['ball_goal_intermediate']
                    else:ball_goal=(2-2*fraction)*r['ball_goal_intermediate']+(2*fraction-1)*r['ball_goal_shoot']
                    desired_right=ball_goal-self.R@r['rel0']
            self.cartesian_right=desired_right.copy()
            if self.last_fit_time is None or t-self.last_fit_time>=.1-1e-9:
                background=actual.qpos[:self.m.nq].copy()
                left_fit,_=fit_wrist(self.m,background,'L',desired_left,left_rotation)
                self.fitted_left=left_fit[self.left_ids].copy()
                if self.stage>=2 and t>=6:
                    right_fit,_=fit_wrist(self.m,background,'R',desired_right,self.R)
                    self.fitted_right=right_fit[self.adrs].copy()
                self.last_fit_time=t
            q[self.left_ids]=self.fitted_left
            if self.fitted_right is not None:q[self.adrs]=self.fitted_right
        self.unfiltered=q.copy()
        if t<=0 or self.previous_goal is None:
            self.previous_goal=q.copy();self.goal_velocity=np.zeros(len(q));self.goal_time=t
            return q
        dt=max(t-self.goal_time,self.m.opt.timestep);self.goal_time=t
        acceleration=np.clip(36*(q-self.previous_goal)-12*self.goal_velocity,-8.,8.)
        self.goal_velocity=np.clip(self.goal_velocity+acceleration*dt,-2.,2.)
        proposed=self.previous_goal+self.goal_velocity*dt
        self.previous_goal=proposed.copy()
        return proposed


class UpperBodyResidualEnv(BasketballResidualEnv):
    def __init__(self,condition='B',stage=1,reference=None,render_mode=None,record=False,max_seconds=None,success_enabled=False,guide_feedback=True,guide_start=1.):
        if condition not in 'ABCD' or stage not in STAGES:raise ValueError('Unknown condition/stage')
        self.upper_ready=False;self.condition=condition;self.stage=stage;self.guide_feedback=guide_feedback;self.guide_start=float(guide_start)
        self.reference_path=Path(reference or OUT/'references/guide_safe/reference.npz')
        self.ref=dict(np.load(self.reference_path)) if stage else None
        super().__init__(render_mode=render_mode,record=record,use_snapshot=True,max_seconds=max_seconds or DURATIONS[stage],success_enabled=success_enabled)
        if condition!='A' and stage:
            self.m=mujoco.MjModel.from_xml_path(str(derived_scene().resolve()));self.d=mujoco.MjData(self.m)
        self.left_names=[n.replace('R_','L_',1) for n in lift.ARM]
        self.left_ids=joint_ids(self.m,self.left_names)
        self.left_dofs=np.array([int(self.m.joint(n).dofadr[0]) for n in self.left_names])
        self.left_finger_names=[f'L_{f}{p}_{a}' for f in FINGERS for p in (1,2,3) for a in 'xyz']
        self.left_finger_ids=joint_ids(self.m,self.left_finger_names)
        self.trunk_names=[f'{b}_{a}' for b in ('Torso','Spine','Chest') for a in 'xyz']
        self.head_names=[f'{b}_{a}' for b in ('Neck','Head') for a in 'xyz']
        self.trunk_ids=joint_ids(self.m,self.trunk_names);self.head_ids=joint_ids(self.m,self.head_names)
        right_synergy=self.synergy.copy();right_names=self.joint_names.copy()
        self.joint_names=right_names+self.left_names+self.left_finger_names+self.trunk_names+self.head_names
        self.ids=joint_ids(self.m,self.joint_names)
        self.bounds=np.array([self.m.jnt_range[self.m.joint(n).id] for n in self.joint_names])
        self.action_dim={'A':12,'B':23,'C':26,'D':28}[condition]
        self.synergy=np.zeros((len(self.ids),self.action_dim));self.synergy[:54,:12]=right_synergy
        if condition!='A':
            self.synergy[54:63,12:21]=np.diag([.12]*6+[.1]*3)
            mapping=json.loads((self.reference_path.parent/'left_axis_mapping.json').read_text()) if stage else []
            for r in mapping:
                row=self.joint_names.index(r['joint']);thumb='Thumb' in r['joint']
                if r['role']=='flexion' and not thumb:self.synergy[row,21]=.06*r['sign']
                if (r['role']=='abduction' and '1_' in r['joint']) or thumb and r['role']=='flexion':self.synergy[row,22]=.035*r['sign']
            # Distributed world-axis synergies use measured FK axes, not hinge letters.
            probe=mujoco.MjData(self.m);probe.qpos[:]=self.ref['initial'] if self.ref is not None else self.d.qpos;mujoco.mj_forward(self.m,probe)
            if condition in ('C','D'):
                for n in self.trunk_names:
                    axis=probe.xaxis[self.m.joint(n).id]
                    self.synergy[self.joint_names.index(n),23:26]=.025*np.array([axis[1],axis[2],axis[0]])
            if condition=='D':
                for n in self.head_names:
                    axis=probe.xaxis[self.m.joint(n).id]
                    self.synergy[self.joint_names.index(n),26:28]=.04*np.array([axis[2],axis[1]])
        self.action_space=spaces.Box(-1.,1.,(self.action_dim,),np.float32)
        self.observation_space=spaces.Box(-np.inf,np.inf,(147+2*self.action_dim,),np.float32)
        self.upper_ready=True;self.snapshot=None;self.reset(seed=7)
    def _adapter(self):
        if not self.upper_ready:return super()._adapter()
        self.bridge=lift.HumanoidControllerBridge(MODELS/'body_01.xml',self.m,SOURCE)
        if self.stage==0:
            ik=np.load(BASE/'ik_path.npz');self.adapter=lift.LiftingTargets(self.bridge,self.candidate,ik['qpos'],ik['rotation'],self.arm_ids,'C',1.,4.)
        else:
            self.adapter=UpperTargets(self.bridge,self.candidate,self.arm_ids,self.ref,self.stage,self.condition)
            self.adapter.guide_feedback=self.guide_feedback;self.adapter.guide_start=self.guide_start
            self.adapter.guide_clearance=float(json.loads((self.reference_path.parent/'summary.json').read_text())['gap_m'])
    def reset(self,*,seed=None,options=None):
        if not self.upper_ready:return super().reset(seed=seed,options=options)
        # Parent performs exact warmup/snapshot initialization. Observation is deferred
        # until filter arrays are expanded to the selected action dimension.
        self.resetting=True
        try:_,parent_info=super().reset(seed=seed,options=options)
        finally:self.resetting=False
        self.action=np.zeros(self.action_dim);self.applied_action=np.zeros(self.action_dim);self.residual=np.zeros(len(self.ids));self.previous_action=np.zeros(self.action_dim);self.previous_action_delta=np.zeros(self.action_dim);self.residual_velocity=np.zeros(self.action_dim)
        self.prev_goal_distance=float(self.last_metrics['goal_distance_m']);self.last_terms={};self.contact_rows=[];self.stage_stable_since=None
        self.last_metrics=self.metrics();return self.observation(),dict(seed=seed,reset_method=parent_info['reset_method'],stage=self.stage)
    def goal_position(self):
        if not self.upper_ready or self.stage==0:return super().goal_position()
        if self.stage<=2:return self.ref['ball_initial'].copy()
        if self.stage==3:return self.ref['ball_goal_intermediate'].copy()
        return self.ref['ball_goal_shoot'].copy()
    def metrics(self):
        z=super().metrics()
        if not self.upper_ready or not z['finite']:return z
        left=side_contacts(self.m,self.d,'L');right=side_contacts(self.m,self.d,'R')
        head=self.m.body('Head').id;chest=self.m.body('Chest').id
        direction=RIM_CENTER-self.d.xpos[head];direction/=np.linalg.norm(direction)
        forward=self.d.xmat[head].reshape(3,3)[:,2]
        head_error=float(np.degrees(np.arccos(np.clip(np.dot(forward,direction),-1,1))))
        refd=mujoco.MjData(self.m);refd.qpos[:]=self.d.qpos;raw=self.adapter.unfiltered if isinstance(self.adapter,UpperTargets) and self.adapter.unfiltered is not None else self.last_reference;refd.qpos[:len(raw)]=raw;mujoco.mj_forward(self.m,refd)
        targetd=mujoco.MjData(self.m);targetd.qpos[:]=self.d.qpos;targetd.qpos[:len(self.last_target)]=self.last_target;targetd.qpos[:7]=self.d.qpos[:7];mujoco.mj_forward(self.m,targetd)
        z.update(left_contacts=left['count'],left_force=left['force'],left_normal=left['normal_force'],left_max_normal=left['max_normal'],left_fingers=left['fingers'],left_flags=left['finger_flags'],left_slip=left['slip'],right_force=right['force'],head_error_deg=head_error,head_direction_error=forward-direction,
            trunk_error_deg=float(np.degrees(Rotation.from_matrix(refd.xmat[chest].reshape(3,3).T@self.d.xmat[chest].reshape(3,3)).magnitude())),head_reference_matrix=refd.xmat[head].reshape(3,3).copy(),trunk_reference_matrix=refd.xmat[chest].reshape(3,3).copy(),face_gap_m=gap_to_ball(self.m,self.d,'Head',self.d.qpos[self.ball_q:self.ball_q+3]))
        for side in ('R','L'):
            b=self.m.body(side+'_Wrist').id
            desired_p=refd.xpos[b].copy()
            if isinstance(self.adapter,UpperTargets):
                cartesian=self.adapter.cartesian_right if side=='R' else self.adapter.cartesian_left
                if cartesian is not None:desired_p=cartesian
            # FK of the commanded wrist joints against actual upstream state.
            local_target=mujoco.MjData(self.m);local_target.qpos[:]=self.d.qpos
            side_ids=self.arm_ids if side=='R' else self.left_ids
            local_target.qpos[side_ids]=self.last_target[side_ids];mujoco.mj_forward(self.m,local_target)
            target_p=local_target.xpos[b];actual_p=self.d.xpos[b]
            z[side+'_ref_target_mm']=float(np.linalg.norm(desired_p-target_p)*1000)
            z[side+'_target_actual_mm']=float(np.linalg.norm(target_p-actual_p)*1000)
            z[side+'_ref_actual_mm']=float(np.linalg.norm(desired_p-actual_p)*1000)
            z[side+'_orientation_error_deg']=float(np.degrees(Rotation.from_matrix(targetd.xmat[b].reshape(3,3).T@self.d.xmat[b].reshape(3,3)).magnitude()))
        self.ref_probe=refd
        return z
    def observation(self):
        if not self.upper_ready or getattr(self,'resetting',False):return np.zeros(self.observation_space.shape,np.float32) if self.upper_ready else super().observation()
        # Reuse the tested 74D base observation with the first 12 residual channels.
        full=self.applied_action;space=self.observation_space
        self.applied_action=full[:12];self.observation_space=spaces.Box(-np.inf,np.inf,(74,),np.float32)
        try:base=super().observation()
        finally:self.applied_action=full;self.observation_space=space
        z=self.last_metrics;rel,rv=local_ball(self.m,self.d,'L')
        chest=self.m.body('Chest').id;head=self.m.body('Head').id
        velocity=np.zeros(6);mujoco.mj_objectVelocity(self.m,self.d,mujoco.mjtObj.mjOBJ_BODY,chest,velocity,1)
        quat=lambda M:Rotation.from_matrix(M).as_quat()
        right_states=np.linalg.lstsq(self.synergy[9:54,9:12],self.d.qpos[self.finger_ids],rcond=None)[0]
        left_states=np.zeros(2) if self.action_dim==12 else np.linalg.lstsq(self.synergy[63:108,21:23],self.d.qpos[self.left_finger_ids],rcond=None)[0]
        extras=np.r_[self.d.qpos[self.left_ids],self.d.qvel[self.left_dofs],self.last_reference[self.left_ids],rel,rv,z['left_flags'],z['left_fingers'],np.zeros(2) if self.action_dim==12 else full[21:23],quat(self.d.xmat[chest].reshape(3,3)),velocity[:3],quat(z['trunk_reference_matrix']),quat(self.d.xmat[head].reshape(3,3)),quat(z['head_reference_matrix']),z['head_direction_error'],z['right_force'],z['left_force'],z['left_slip'],right_states,left_states,full[12:],self.residual_velocity,self.goal_position()-self.d.qpos[self.ball_q:self.ball_q+3],z['goal_distance_m'],np.clip(self.d.time/self.max_seconds,0,1),self.stage/5.]
        obs=np.r_[base,extras].astype(np.float32)
        if obs.shape!=self.observation_space.shape:raise RuntimeError(f'Observation mismatch: {obs.shape} vs {space.shape}')
        return np.nan_to_num(obs,nan=0.,posinf=1e6,neginf=-1e6)
    def residual_target(self,reference,action):
        acceleration=np.clip(100*(action-self.applied_action)-20*self.residual_velocity,-16.,16.)
        self.residual_velocity=np.clip(self.residual_velocity+acceleration*self.dt,-4.,4.)
        proposed=self.applied_action+self.residual_velocity*self.dt
        crossed=(action-self.applied_action)*(action-proposed)<=0
        proposed[crossed]=action[crossed];self.residual_velocity[crossed]=0
        self.applied_action=np.clip(proposed,-1,1);self.residual=self.synergy@self.applied_action
        target=reference.copy();target[self.ids]=np.clip(reference[self.ids]+self.residual,self.bounds[:,0],self.bounds[:,1])
        return target
    def reward_terms(self,z,action):
        if not z['finite']:
            return {k:0. for k in ('ball_progress','grasp_stability','bimanual_coordination','shooting_pose','head_alignment','balance','reference_tracking','slip','action','jerk','guide_push')}
        held=bool(z['contacts']>0 and z['drift_m']<.05 and z['finite'] and not z['fallen'])
        distance=z['goal_distance_m'];progress=(self.prev_goal_distance-distance)/self.control_dt
        guide=z['left_contacts']>0 and .05<z['left_normal']<8 and z['left_max_normal']<5 and abs(z['left_force'][2])<3
        pose=float(np.exp(-np.mean((self.d.qpos[self.trunk_ids]-self.last_reference[self.trunk_ids])**2)/.1**2))
        # Distinct weights: uncertain wrist/finger channels contribute less.
        err=self.d.qpos[self.ids]-self.last_reference[self.ids];weights=np.ones(len(err));weights[6:54]=.2;weights[60:108]=.2
        delta=action-self.previous_action;jerk=delta-self.previous_action_delta
        return dict(ball_progress=float((np.exp(-distance/.25)+np.clip(progress,-1,1)) * held),grasp_stability=float(np.exp(-(z['drift_m']/.015)**2)*held),bimanual_coordination=float(guide*held),shooting_pose=float(pose*held),head_alignment=float(np.exp(-(z['head_error_deg']/15)**2)*held),balance=float(np.clip(z['com_margin_m']/.1,0,1)*z['both_feet']*held),reference_tracking=float(np.exp(-np.average(err**2,weights=weights)/.15**2)*held),slip=float(np.clip((z['slip_ms']+z['left_slip'])/.1,0,3)),action=float(np.mean(action**2)),jerk=float(np.mean(jerk**2)),guide_push=float(max(0,z['left_normal']-8)/8+max(0,abs(z['left_force'][2])-3)/3))
    def stage_success(self,z):
        stable=z['finite'] and not z['fallen'] and not any(z['warnings']) and z['drift_m']<.02 and z['both_feet'] and z['com_margin_m']>0 and (self.stage<2 or z['head_error_deg']<15)
        if self.stage==0:return z['drift_m']<.05 and z['goal_distance_m']<.04 and z['both_feet'] and not z['fallen'] and not any(z['warnings'])
        guide=(z['left_contacts']>0 and .05<z['left_normal']<8 and z['left_max_normal']<5)
        if self.stage==1:return stable and guide and z['goal_distance_m']<.03
        if self.stage==2:return stable and guide and z['trunk_error_deg']<5 and z['goal_distance_m']<.03
        return stable and guide and z['goal_distance_m']<.03 and z['R_target_actual_mm']<30 and z['R_orientation_error_deg']<10
    def step(self,action):
        if self.finished:raise RuntimeError('Reset after episode end')
        action=np.asarray(action,float)
        if action.shape!=(self.action_dim,) or not np.isfinite(action).all():raise ValueError('Invalid action')
        self.action=np.clip(action,-1,1);reason=None;self.steps+=1
        if not np.isfinite(self.d.qpos).all() or not np.isfinite(self.d.qvel).all():
            self.last_metrics=self.metrics();self.finished=True;return self.observation(),-5.,True,False,dict(termination_reason='nonfinite_or_physics_warning',reward_terms=self.reward_terms(self.last_metrics,self.action))
        for _ in range(self.substeps):
            if self.d.time>=self.max_seconds-1e-10:break
            ref=self.adapter.target(self.d.time,self.d);target=self.residual_target(ref,self.action)
            before=self.d.qpos[self.ball_q:self.ball_q+7].copy();v=self.d.qvel[self.ball_v:self.ball_v+6].copy()
            self.adapter.control(self.d,target)
            assert np.array_equal(before,self.d.qpos[self.ball_q:self.ball_q+7]) and np.array_equal(v,self.d.qvel[self.ball_v:self.ball_v+6])
            assert self.m.neq==0 and not self.d.xfrc_applied.any() and not self.d.qfrc_applied.any()
            self.last_reference=ref;self.last_target=target;mujoco.mj_step(self.m,self.d);mujoco.mj_forward(self.m,self.d);z=self.metrics();self.last_metrics=z
            reason=super().termination(z)
            if reason=='goal_stable':reason=None  # use upper-body stage gate instead
            if self.stage_success(z):
                if getattr(self,'stage_stable_since',None) is None:self.stage_stable_since=z['time_s']
            else:self.stage_stable_since=None
            if self.success_enabled and self.stage_stable_since is not None and z['time_s']-self.stage_stable_since>=1.:reason='stage_success'
            if z['left_normal']>20:reason='excessive_guide_force'
            if self.record:
                row={k:z[k] for k in ('time_s','contacts','force_z_N','drift_m','slip_ms','com_margin_m','both_feet','fallen','wrist_z_m','ball_z_m','goal_distance_m','left_contacts','left_normal','left_max_normal','left_slip','head_error_deg','trunk_error_deg','face_gap_m','R_ref_target_mm','R_target_actual_mm','R_ref_actual_mm','R_orientation_error_deg','L_ref_target_mm','L_target_actual_mm','L_ref_actual_mm','L_orientation_error_deg')}
                for side in ('R','L'):
                    for a,k in enumerate('xyz'):row[side+'_force_'+k+'_N']=z['right_force' if side=='R' else 'left_force'][a]
                for a,k in enumerate('xyz'):row['ball_'+k+'_m']=self.d.qpos[self.ball_q+a];row['ball_v'+k+'_ms']=self.d.qvel[self.ball_v+a]
                row['stage_success']=int(self.stage_success(z));self.rows.append(row)
                if (len(self.rows)-1)%15==0:
                    self.states.append(self.d.qpos.copy());self.state_times.append(float(self.d.time));self.goals.append(dict(time_s=float(self.d.time),**{n+'_reference_rad':ref[j] for n,j in zip(self.joint_names,self.ids)},**{n+'_target_rad':target[j] for n,j in zip(self.joint_names,self.ids)},**{n+'_torque_Nm':float(self.d.qfrc_actuator[int(self.m.joint(n).dofadr[0])]) for n in self.joint_names}))
                    for side in ('R','L'):self.contact_rows.extend([dict(time_s=float(self.d.time),side=side,**r) for r in side_contacts(self.m,self.d,side)['records']])
            if reason:break
        terminated=reason is not None;truncated=not terminated and self.d.time>=self.max_seconds-1e-10
        terms=self.reward_terms(z,self.action);weights=dict(ball_progress=3.,grasp_stability=1.,bimanual_coordination=.5,shooting_pose=.5,head_alignment=.25,balance=.5,reference_tracking=.5,slip=.2,action=.1,jerk=.05,guide_push=1.)
        weighted={k:weights[k]*v*(-1 if k in ('slip','action','jerk','guide_push') else 1) for k,v in terms.items()};reward=sum(weighted.values())+(-5 if terminated and reason!='stage_success' else 5 if reason=='stage_success' else 0)
        self.previous_action_delta=self.action-self.previous_action;self.previous_action=self.action.copy();self.prev_goal_distance=z['goal_distance_m'];self.finished=terminated or truncated;self.last_terms=terms
        info=dict(reward_terms=terms,weighted_reward_terms=weighted,termination_reason=reason if terminated else 'time_limit' if truncated else None,stage=self.stage,stage_success=bool(self.stage_success(z)),metrics={k:v for k,v in z.items() if np.isscalar(v)},ball_state_writes_after_reset=0)
        obs=self.observation()
        if not np.isfinite(obs).all() or not np.isfinite(reward):raise RuntimeError('Nonfinite result')
        if self.render_mode=='human':self.render()
        return obs,float(reward),bool(terminated),bool(truncated),info
