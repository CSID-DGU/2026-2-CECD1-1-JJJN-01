"""M5-0 contact-only Residual Gymnasium adapter over unchanged Lifting C."""
import copy,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/m5_0_residual_ppo';sys.path.insert(0,str(OUT/'vendor'))
import gymnasium as gym
from gymnasium import spaces
import mujoco,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from examples import m4_contact_lifting as lift
from examples.m4_dynamic_grasp import local_ball,contact_records
from examples.m3_control import write_csv,FINGERS
from smpl_sim.motion.physics_metrics import foot_contact_slip
BASE=ROOT/'outputs/m4_2d_2_lifting';MODELS=ROOT.parents[2]/'basketball-r1/SMPLSim/outputs/smplh_physics';SOURCE=ROOT/'outputs/m3_spl_fitting/spl_P0002_T0049.npz'

class BasketballResidualEnv(gym.Env):
    metadata={'render_modes':['human','rgb_array'],'render_fps':30}
    def __init__(self,render_mode=None,max_seconds=6.5,use_snapshot=False,goal_mode='reference',goal_body='Head',goal_offset=(.18,0,.18),success_enabled=False,record=False,reward_weights=None):
        super().__init__();self.render_mode=render_mode;self.max_seconds=float(max_seconds);self.use_snapshot=use_snapshot;self.goal_mode=goal_mode;self.goal_body=goal_body;self.goal_offset=np.asarray(goal_offset,float);self.success_enabled=success_enabled;self.record=record;self.viewer=None;self.renderer=None
        if not np.isfinite(self.max_seconds) or self.max_seconds<=0:raise ValueError('max_seconds must be positive')
        if render_mode not in (None,'human','rgb_array'):raise ValueError('Unknown render_mode')
        if goal_mode not in ('reference','body'):raise ValueError('Unknown goal_mode')
        if self.goal_offset.shape!=(3,) or not np.isfinite(self.goal_offset).all():raise ValueError('Invalid goal offset')
        self.m=mujoco.MjModel.from_xml_path(str((BASE/'contact.xml').resolve()));self.d=mujoco.MjData(self.m)
        if self.m.neq or any('proxy' in (self.m.geom(g).name or '').lower() or 'palm_pad' in (self.m.geom(g).name or '') for g in range(self.m.ngeom)):raise RuntimeError('Forbidden attachment/proxy')
        self.dt=float(self.m.opt.timestep);self.substeps=round(1/(30*self.dt));self.control_dt=self.dt*self.substeps
        if not np.isclose(self.control_dt,1/30,atol=1e-12):raise ValueError('30Hz must use integer unchanged physics steps')
        self.ball_q=int(self.m.joint('basketball_free').qposadr[0]);self.ball_v=int(self.m.joint('basketball_free').dofadr[0]);self.wrist=self.m.body('R_Wrist').id
        self.arm_names=lift.ARM;self.arm_ids=np.array([int(self.m.joint(n).qposadr[0]) for n in self.arm_names]);self.arm_dofs=np.array([int(self.m.joint(n).dofadr[0]) for n in self.arm_names]);self.finger_names=[r['joint'] for r in json.loads((BASE/'axis_mapping.json').read_text())];self.finger_ids=np.array([int(self.m.joint(n).qposadr[0]) for n in self.finger_names]);self.ids=np.r_[self.arm_ids,self.finger_ids]
        self.joint_names=self.arm_names+self.finger_names;self.bounds=np.array([self.m.jnt_range[self.m.joint(n).id] for n in self.joint_names]);self.synergy=np.zeros((len(self.ids),12));self.synergy[:9,:9]=np.diag([.15]*6+[.12]*3)
        mapping=json.loads((BASE/'axis_mapping.json').read_text())
        for i,r in enumerate(mapping,9):
            thumb='Thumb' in r['segment'];sign=r['sign']
            if r['role']=='flexion' and not thumb:self.synergy[i,9]=.08*sign
            if r['role']=='abduction' and r['segment'].endswith('1') and not thumb:self.synergy[i,10]=.05*sign
            if thumb and ((r['segment']=='R_Thumb1' and r['role'] in ('flexion','abduction')) or (r['segment']=='R_Thumb2' and r['role']=='flexion')):self.synergy[i,11]=.06*sign*(.5 if r['segment']=='R_Thumb2' else 1.)
        self.action_space=spaces.Box(-1,1,(12,),np.float32);self.observation_space=spaces.Box(-np.inf,np.inf,(74,),np.float32)
        self.weights={'ball_target':4.,'grasp':1.,'motion':.5,'balance':.5,'action':.1,'slip':.2};
        if reward_weights is not None:
            if set(reward_weights)-set(self.weights):raise ValueError('Unknown reward component')
            self.weights.update(reward_weights)
        if not all(np.isfinite(v) and v>=0 for v in self.weights.values()):raise ValueError('Invalid reward weights')
        self.snapshot=None;self.input_hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in [BASE/'contact.xml',BASE/'ik_path.npz',SOURCE,MODELS/'body_01.xml']}
        self.candidate=next(c for c in json.loads((lift.PREVIOUS/'candidates.json').read_text()) if c['name']=='fk_fit_2')
        self.reset()
    def _adapter(self):
        self.bridge=lift.HumanoidControllerBridge(MODELS/'body_01.xml',self.m,SOURCE);ik=np.load(BASE/'ik_path.npz');self.adapter=lift.LiftingTargets(self.bridge,self.candidate,ik['qpos'],ik['rotation'],self.arm_ids,'C',1.,4.)
    def capture_snapshot(self):
        spec=mujoco.mjtState.mjSTATE_INTEGRATION
        def state(m,d):
            x=np.zeros(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,d,x,spec);return x
        c=self.bridge.controller
        return dict(physics=state(self.m,self.d),bridge=state(self.bridge.model,self.bridge.data),jac=c.jac.copy(),last_correction=c.last_correction.copy(),kp=c.pd.jkp.copy(),kd=c.pd.jkd.copy(),limit=c.limit.copy(),frozen=copy.deepcopy(self.adapter.frozen))
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);self.action_space.seed(seed);self._adapter();use=(options or {}).get('use_snapshot',self.use_snapshot)
        if use and self.snapshot is not None:
            s=self.snapshot;spec=mujoco.mjtState.mjSTATE_INTEGRATION;mujoco.mj_resetData(self.m,self.d);mujoco.mj_setState(self.m,self.d,s['physics'],spec);mujoco.mj_setState(self.bridge.model,self.bridge.data,s['bridge'],spec)
            c=self.bridge.controller
            for key,value in [('kp',c.pd.jkp),('kd',c.pd.jkd),('limit',c.limit)]:np.testing.assert_array_equal(s[key],value)
            c.jac[:]=s['jac'];c.last_correction[:]=s['last_correction'];self.adapter.frozen=copy.deepcopy(s['frozen']);mujoco.mj_forward(self.m,self.d);mujoco.mj_forward(self.bridge.model,self.bridge.data);method='snapshot'
        else:
            lift.initialize(self.m,self.d,self.bridge,self.adapter,self.candidate);self.snapshot=self.capture_snapshot();method='warmup'
        old=json.loads((BASE/'C/initial_state.json').read_text());np.testing.assert_allclose(self.d.qpos[self.ball_q:self.ball_q+7],old['ball_qpos'],atol=1e-10,rtol=0)
        self.rel0,_=local_ball(self.m,self.d);self.root0=float(self.d.qpos[2]);self.goal_initial=self.d.qpos[self.ball_q:self.ball_q+3].copy()+[0,0,.25];self.action=np.zeros(12);self.applied_action=np.zeros(12);self.residual=np.zeros(len(self.ids));self.seen=False;self.loss_start=None;self.stable_since=None;self.finished=False;self.rows=[];self.goals=[];self.states=[];self.state_times=[];self.steps=0;self.last_reference=self.adapter.target(self.d.time,self.d);self.last_target=self.last_reference.copy();self.ball_state_writes_after_reset=0;self.last_metrics=self.metrics();return self.observation(),dict(reset_method=method,seed=seed,physics_dt=self.dt,control_dt=self.control_dt)
    def save_snapshot(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);s=self.snapshot;np.savez_compressed(path,**{k:v for k,v in s.items() if k!='frozen'},metadata=json.dumps(self.input_hashes),frozen_present=s['frozen'] is not None,frozen=np.zeros(3) if s['frozen'] is None else s['frozen'])
    def load_snapshot(self,path):
        x=np.load(path);saved=json.loads(str(x['metadata']))
        if saved!=self.input_hashes:raise ValueError('Snapshot inputs changed')
        self.snapshot={k:x[k].copy() for k in ['physics','bridge','jac','last_correction','kp','kd','limit']};self.snapshot['frozen']=x['frozen'].copy() if bool(x['frozen_present']) else None
    def goal_position(self):
        if self.goal_mode=='reference':return self.goal_initial.copy()
        origin=self.d.xpos[self.m.body(self.goal_body).id];pelvis=Rotation.from_quat(np.roll(self.d.qpos[3:7],-1)).as_matrix();forward=pelvis@np.array([0,0,1.]);forward[2]=0;forward/=max(np.linalg.norm(forward),1e-9);left=np.cross([0,0,1.],forward)
        return origin+forward*self.goal_offset[0]+left*self.goal_offset[1]+np.array([0,0,self.goal_offset[2]])
    def reachability(self):
        goal=self.goal_position();shoulder=self.d.xpos[self.m.body('R_Shoulder').id];distance=float(np.linalg.norm(goal-shoulder));length=sum(np.linalg.norm(self.m.body(n).pos) for n in ('R_Elbow','R_Wrist'))+np.linalg.norm(self.rel0)
        probe=mujoco.MjData(self.m);base=self.d.qpos.copy();W0=self.d.xmat[self.wrist].reshape(3,3).copy();x0=base[self.arm_ids]
        def fun(x):
            probe.qpos[:]=base;probe.qpos[self.arm_ids]=x;mujoco.mj_forward(self.m,probe);W=probe.xmat[self.wrist].reshape(3,3);p=probe.xpos[self.wrist]+W@self.rel0
            return np.r_[20*(p-goal),.01*(x-x0)]
        fit=least_squares(fun,np.clip(x0,self.bounds[:9,0]+1e-5,self.bounds[:9,1]-1e-5),bounds=(self.bounds[:9,0]+1e-5,self.bounds[:9,1]-1e-5),max_nfev=150)
        error=float(np.linalg.norm(fit.fun[:3])/20)
        # Geometric reachability with the M5 residual envelope is stricter than
        # unconstrained XML IK. Use the successful Lifting C endpoint as anchor.
        end=np.load(BASE/'C/states.npz')['qpos'][-1].copy();centre=end[self.arm_ids];radius=np.diag(self.synergy[:9,:9]);lo=np.maximum(centre-radius,self.bounds[:9,0]+1e-5);hi=np.minimum(centre+radius,self.bounds[:9,1]-1e-5)
        def bounded(x):
            probe.qpos[:]=end;probe.qpos[self.arm_ids]=x;mujoco.mj_forward(self.m,probe);W=probe.xmat[self.wrist].reshape(3,3)
            return probe.xpos[self.wrist]+W@self.rel0-goal
        limited=least_squares(bounded,np.clip(centre,lo,hi),bounds=(lo,hi),max_nfev=150);limited_error=float(np.linalg.norm(limited.fun))
        return dict(residual_endpoint_ik_error_m=limited_error,residual_endpoint_candidate_reachable=limited_error<.01,goal_world_m=goal.tolist(),ball_radius_m=.121,upper_reach_bound_m=float(length),shoulder_distance_m=distance,within_distance_bound=bool(distance<=length),ik_position_error_m=error,ik_candidate_reachable=error<.01,collision_and_balance_validated=False)
    def metrics(self):
        finite=bool(np.isfinite(self.d.qpos).all() and np.isfinite(self.d.qvel).all())
        if not finite:
            z=copy.deepcopy(self.last_metrics);z['finite']=False;z['time_s']=float(self.d.time);return z
        n,nf,F,ff,depth,slip=lift.contact.hand_contacts(self.m,self.d);rel,rv=local_ball(self.m,self.d);feet,margin,_=lift.contact.contacts(self.m,self.d);tilt=np.degrees(np.arccos(np.clip(self.d.xmat[self.m.body('Pelvis').id].reshape(3,3)[2,1],-1,1)));per=np.zeros(5)
        for r in contact_records(self.m,self.d):
            for i,name in enumerate(FINGERS):
                if r['geom'].startswith('R_'+name):per[i]+=r['normal_force_N']
        return dict(time_s=float(self.d.time),contacts=n,finger_contacts=nf,force_z_N=float(F[2]),finger_force_z_N=float(ff[2]),slip_ms=slip,drift_m=float(np.linalg.norm(rel-self.rel0)),relative=rel,relative_velocity=rv,finger_forces=per,feet=feet,com_margin_m=margin,both_feet=int(min(feet.values())>.1),fallen=bool(self.d.qpos[2]<.65*self.root0 or tilt>35),finite=finite,warnings=[int(w.number) for w in self.d.warning],wrist_z_m=float(self.d.xpos[self.wrist,2]),ball_z_m=float(self.d.qpos[self.ball_q+2]),goal_distance_m=float(np.linalg.norm(self.goal_position()-self.d.qpos[self.ball_q:self.ball_q+3])))
    def observation(self):
        z=self.last_metrics;W=self.d.xmat[self.wrist].reshape(3,3);phase=np.clip((self.d.time-1)/4,0,1)
        obs=np.r_[self.d.qpos[self.arm_ids],self.d.qvel[self.arm_dofs],self.last_reference[self.arm_ids],self.applied_action,z['relative'],z['relative_velocity'],W.T@self.d.qvel[self.ball_v:self.ball_v+3],self.d.qvel[self.ball_v+3:self.ball_v+6],int(z['contacts']>0),z['finger_forces'],z['force_z_N'],z['slip_ms'],self.d.qpos[3:7],self.d.qvel[3:6],z['com_margin_m'],[int(z['feet'][s]>.1) for s in ('L','R')],phase,np.clip(self.d.time/6.5,0,1),W.T@(self.goal_position()-self.d.qpos[self.ball_q:self.ball_q+3])].astype(np.float32)
        if obs.shape!=self.observation_space.shape:raise RuntimeError('Observation dimension mismatch')
        return np.nan_to_num(obs,nan=0,posinf=1e6,neginf=-1e6) if not z['finite'] else obs
    def residual_target(self,reference,action):
        self.applied_action+=np.clip(action-self.applied_action,-4*self.dt,4*self.dt);self.residual=self.synergy@self.applied_action;target=reference.copy();target[self.ids]=np.clip(reference[self.ids]+self.residual,self.bounds[:,0],self.bounds[:,1]);return target
    def termination(self,z):
        t=z['time_s'];self.seen=self.seen or z['contacts']>0
        if z['contacts']>0:self.loss_start=None
        elif self.loss_start is None:self.loss_start=t
        if not z['finite'] or any(z['warnings']):return 'nonfinite_or_physics_warning'
        if z['fallen']:return 'humanoid_fall'
        if z['drift_m']>.05:return 'ball_relative_drift'
        if self.loss_start is not None and t-self.loss_start>=.05:return 'contact_loss'
        effective_contact=self.seen and (self.loss_start is None or t-self.loss_start<.05)
        stable=z['goal_distance_m']<.025 and effective_contact and z['drift_m']<.02 and z['both_feet'] and z['com_margin_m']>0
        if stable and self.stable_since is None:self.stable_since=t
        if not stable:self.stable_since=None
        if self.success_enabled and self.stable_since is not None and t-self.stable_since>=.5:return 'goal_stable'
        return None
    def reward_terms(self,z,action):
        if not z['finite']:return dict(ball_target=0.,grasp=0.,motion=0.,balance=0.,action=float(np.mean(action**2)),slip=0.)
        held=z['contacts']>0 and z['drift_m']<=.05 and z['finite'] and not z['fallen'];near=np.clip(1-z['goal_distance_m']/.25,0,1)
        return dict(ball_target=float(near**2*held),grasp=float(np.exp(-(z['drift_m']/.015)**2)*held),motion=float(np.exp(-np.mean((self.d.qpos[self.arm_ids]-self.last_reference[self.arm_ids])**2)/.15**2)*held),balance=float(np.clip(z['com_margin_m']/.1,0,1)*z['both_feet']*held),action=float(np.mean(action**2)),slip=float(np.clip(z['slip_ms']/.1,0,2)))
    def step(self,action):
        if self.finished:raise RuntimeError('reset required after done')
        action=np.asarray(action,float)
        if action.shape!=(12,) or not np.isfinite(action).all():raise ValueError('Action must be finite shape (12,)')
        if not np.isfinite(self.d.qpos).all() or not np.isfinite(self.d.qvel).all():
            self.last_metrics=self.metrics();self.finished=True;return self.observation(),-5.,True,False,dict(termination_reason='nonfinite_or_physics_warning',reward_terms=self.reward_terms(self.last_metrics,action),ball_state_writes_after_reset=0)
        self.action=np.clip(action,-1,1);terminated=False;reason=None;self.steps+=1
        for _ in range(self.substeps):
            if self.d.time>=self.max_seconds-1e-10:break
            reference=self.adapter.target(self.d.time,self.d);target=self.residual_target(reference,self.action);bp=self.d.qpos[self.ball_q:self.ball_q+7].copy();bv=self.d.qvel[self.ball_v:self.ball_v+6].copy();self.adapter.control(self.d,target)
            assert np.array_equal(bp,self.d.qpos[self.ball_q:self.ball_q+7]) and np.array_equal(bv,self.d.qvel[self.ball_v:self.ball_v+6]);assert self.m.neq==0 and not self.d.qfrc_applied.any() and not self.d.xfrc_applied.any()
            self.last_reference=reference;self.last_target=target;mujoco.mj_step(self.m,self.d);mujoco.mj_forward(self.m,self.d);z=self.metrics();self.last_metrics=z;reason=self.termination(z)
            if self.record:
                fs,_=foot_contact_slip(self.m,self.d);row={k:z[k] for k in ['time_s','contacts','finger_contacts','force_z_N','finger_force_z_N','slip_ms','drift_m','com_margin_m','both_feet','fallen','wrist_z_m','ball_z_m','goal_distance_m']};row['arm_tracking_rmse_rad']=float(np.sqrt(np.mean((self.d.qpos[self.arm_ids]-target[self.arm_ids])**2)));row['foot_slip_ms']=float(np.sqrt((fs['L']['rms_speed']**2+fs['R']['rms_speed']**2)/2));self.rows.append(row)
                if (len(self.rows)-1)%15==0:
                    self.states.append(self.d.qpos.copy());self.state_times.append(float(self.d.time));r=dict(time_s=float(self.d.time))
                    for n,j in zip(self.joint_names,self.ids):r[n+'_reference_rad']=reference[j];r[n+'_target_rad']=target[j];r[n+'_residual_rad']=target[j]-reference[j];r[n+'_torque_Nm']=float(self.d.qfrc_actuator[int(self.m.joint(n).dofadr[0])])
                    self.goals.append(r)
            if reason:terminated=True;break
        truncated=not terminated and self.d.time>=self.max_seconds-1e-10;self.finished=terminated or truncated;terms=self.reward_terms(self.last_metrics,self.action);reward=sum(self.weights[k]*v*(-1 if k in ('action','slip') else 1) for k,v in terms.items())
        if terminated and reason!='goal_stable':reward-=5.
        if reason=='goal_stable':reward+=5.
        info=dict(reward_terms=terms,weighted_reward_terms={k:self.weights[k]*v*(-1 if k in ('action','slip') else 1) for k,v in terms.items()},termination_reason=reason if terminated else 'time_limit' if truncated else None,metrics={k:v for k,v in self.last_metrics.items() if k not in ('relative','relative_velocity','finger_forces','feet')},ball_state_writes_after_reset=self.ball_state_writes_after_reset,residual_norm_rad=float(np.linalg.norm(self.last_target[self.ids]-self.last_reference[self.ids])))
        obs=self.observation()
        if not np.isfinite(obs).all() or not np.isfinite(reward):raise RuntimeError('Nonfinite observation/reward')
        if self.render_mode=='human':self.render()
        return obs,float(reward),bool(terminated),bool(truncated),info
    def render(self):
        if self.render_mode=='human':
            import mujoco.viewer as mv
            if self.viewer is None:self.viewer=mv.launch_passive(self.m,self.d);self.viewer.cam.lookat[:]=[.2,0,1.25];self.viewer.cam.distance=2.;self.viewer.cam.azimuth=110;self.viewer.cam.elevation=-15
            if self.viewer.is_running():self.viewer.sync()
            return None
        if self.render_mode=='rgb_array':
            if self.renderer is None:self.renderer=mujoco.Renderer(self.m,540,960)
            camera=mujoco.MjvCamera();camera.lookat[:]=[.2,0,1.25];camera.distance=2.;camera.azimuth=110;camera.elevation=-15;self.renderer.update_scene(self.d,camera=camera);return self.renderer.render()
        return None
    def close(self):
        if self.viewer:self.viewer.close();self.viewer=None
        if self.renderer:self.renderer.close();self.renderer=None
