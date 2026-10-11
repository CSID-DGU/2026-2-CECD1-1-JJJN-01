"""Independent A-C shooting reference experiment; existing M4/M5 are read-only.
Candidate timings and poses are configurable priors, not human ground truth.
"""
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from examples.m5_upper_body_ppo_env import UpperBodyResidualEnv,UpperTargets,MODELS,SOURCE
from examples.m5_upper_body_ik import minimum_jerk,joint_ids,fit_wrist
from examples.m5_bimanual_contact import side_contacts,local_ball,gap_to_ball
from examples.m5_residual_ppo_env import OUT as OLD_OUT,BASE,BasketballResidualEnv
from examples import m4_contact_lifting as lift
from examples.m3_control import write_csv
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
import mujoco,numpy as np
OUT=ROOT/'outputs/m5_1_coordinated_reference'


def scene_rim(m,d):
    ids=[g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('rim_')]
    if not ids:raise ValueError('No rim geometries in scene')
    return np.mean(d.geom_xpos[ids],axis=0)


def unit(x):return x/max(np.linalg.norm(x),1e-12)


def elbow_bend(m,d):
    s,e,w=[d.xpos[m.body('R_'+n).id] for n in ('Shoulder','Elbow','Wrist')]
    return float(np.arccos(np.clip(np.dot(unit(e-s),unit(w-e)),-1,1)))


def reserve(m,d,rel):
    """Operational FK reserve inside +/-0.35rad, not anatomical ROM certification."""
    q=d.qpos.copy();probe=mujoco.MjData(m);probe.qpos[:]=q;mujoco.mj_kinematics(m,probe);mujoco.mj_comPos(m,probe)
    elbow=joint_ids(m,[f'R_Elbow_{a}' for a in 'xyz']);wrist=joint_ids(m,[f'R_Wrist_{a}' for a in 'xyz'])
    grad=[]
    for adr in elbow:
        bends=[]
        for sign in (-1,1):
            probe.qpos[:]=q;probe.qpos[adr]+=.005*sign;mujoco.mj_kinematics(m,probe);mujoco.mj_comPos(m,probe);bends.append(elbow_bend(m,probe))
        grad.append((bends[1]-bends[0])/.01)
    extension=-unit(np.array(grad));bounds=np.array([m.jnt_range[m.joint(f'R_Elbow_{a}').id] for a in 'xyz'])
    amounts=np.linspace(0,.35,36);bends=[]
    for amount in amounts:
        probe.qpos[:]=q;probe.qpos[elbow]=np.clip(q[elbow]+extension*amount,bounds[:,0],bounds[:,1]);mujoco.mj_kinematics(m,probe);mujoco.mj_comPos(m,probe);bends.append(elbow_bend(m,probe))
    W=d.xmat[m.body('R_Wrist').id].reshape(3,3);p=d.xpos[m.body('R_Wrist').id]+W@rel
    direction=unit(scene_rim(m,d)-p);jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv))
    mujoco.mj_jac(m,d,jp,jr,p,m.body('R_Wrist').id)
    dofs=[int(m.joint(f'R_Wrist_{a}').dofadr[0]) for a in 'xyz'];flex=unit(jp[:,dofs].T@direction)
    wrist_bounds=np.array([m.jnt_range[m.joint(f'R_Wrist_{a}').id] for a in 'xyz'])
    room=[]
    for value,axis,(lo,hi) in zip(q[wrist],flex,wrist_bounds):
        if abs(axis)>1e-8:room.append((hi-value)/axis if axis>0 else (lo-value)/axis)
    return dict(elbow_bend_deg=np.degrees(elbow_bend(m,d)),elbow_extension_reserve_deg=np.degrees(max(0,elbow_bend(m,d)-min(bends))),wrist_rim_rotation_reserve_rad=max(0,min([.35]+room)),extension_direction=extension,wrist_direction=flex,ball_center_jacobian_rim_gain_m_rad=float(np.linalg.norm(jp[:,dofs].T@direction)))


def generate():
    folder=OUT/'reference';folder.mkdir(parents=True,exist_ok=True)
    old=dict(np.load(ROOT/'outputs/m5_1_upper_body/references/guide_safe/reference.npz'))
    env=UpperBodyResidualEnv(condition='B',stage=1);env.reset(seed=7)
    m,d=env.m,env.d;initial=d.qpos.copy();rim=scene_rim(m,d);forward=unit(np.r_[rim[:2]-d.xpos[m.body('Pelvis').id,:2],0.]);side=unit(np.cross([0,0,1.],forward))
    initial_ball=d.qpos[env.ball_q:env.ball_q+3].copy();R0=np.load(BASE/'ik_path.npz')['rotation'];rel=env.rel0.copy()
    q=env.adapter.target(0,d);q=initial.copy();q[7:env.bridge.model.nq]=env.last_reference[7:]
    trunk=old['trunk_ids'].astype(int);head=old['head_ids'].astype(int)
    # Smaller trunk change than old M5; bounded range remains a tunable prior.
    aligned=q.copy();aligned[trunk]*=.92;probe=mujoco.MjData(m);prior=aligned[head].copy()
    def hfun(x):
        probe.qpos[:]=aligned;probe.qpos[head]=x;mujoco.mj_forward(m,probe);bid=m.body('Head').id
        return np.r_[3*(probe.xmat[bid].reshape(3,3)[:,2]-unit(rim-probe.xpos[bid])),.04*(x-prior)]
    b=np.array([m.jnt_range[m.joint(n).id] for n in [f'{b}_{a}' for b in ('Neck','Head') for a in 'xyz']]);lo=np.maximum(b[:,0]+1e-5,prior-.7);hi=np.minimum(b[:,1]-1e-5,prior+.7)
    aligned[head]=least_squares(hfun,np.clip(prior,lo,hi),bounds=(lo,hi),max_nfev=80).x
    probe.qpos[:]=aligned;mujoco.mj_forward(m,probe);hp=probe.xpos[m.body('Head').id].copy()
    final=hp+.26*forward+[0,0,.13];intermediate=hp+.32*forward+[0,0,.015];lift_goal=initial_ball+[0,0,.25]
    # Save actual geometric axes for future extension; no live ball writes.
    samples=[];rows=[];previous=aligned.copy()
    for u in np.linspace(0,1,61):
        center=(1-u)**2*lift_goal+2*u*(1-u)*intermediate+u*u*final
        previous,fit=fit_wrist(m,previous,'R',center-R0@rel,R0,regularization=.08)
        probe.qpos[:]=previous;mujoco.mj_forward(m,probe);r=reserve(m,probe,rel)
        rows.append(dict(fraction=u,ik_position_error_mm=fit['position_error_mm'],elbow_bend_deg=r['elbow_bend_deg'],elbow_extension_reserve_deg=r['elbow_extension_reserve_deg'],wrist_reserve_rad=r['wrist_rim_rotation_reserve_rad'],head_gap_m=gap_to_ball(m,probe,'Head',center)));samples.append(previous.copy())
    np.savez_compressed(folder/'reference.npz',initial=initial,aligned=aligned,shooting_path=samples,right_rotation=R0,left_rotation=old['left_rotation'],left_offset=old['left_offset'],rel0=rel,trunk_ids=trunk,head_ids=head,ball_initial=initial_ball,ball_goal_lift=lift_goal,ball_goal_intermediate=intermediate,ball_goal_shoot=final,approached=old['approached'],left_origin=old['left_origin'],rim_center=rim,forward=forward,side=side)
    # Parent constructor reads mappings; these are derived reports, not SMPL assets.
    mapping=json.loads((ROOT/'outputs/m5_1_upper_body/references/guide_safe/left_axis_mapping.json').read_text());(folder/'left_axis_mapping.json').write_text(json.dumps(mapping,indent=2));(folder/'summary.json').write_text(json.dumps(dict(gap_m=.012,ik_physics_success=False,maximum_position_error_mm=max(x['ik_position_error_mm'] for x in rows),configuration_is_prior=True),indent=2));write_csv(folder/'ik_readiness.csv',rows)
    env.close();return folder/'reference.npz'


class CoordinatedTargets(UpperTargets):
    def __init__(self,*args,config,**kwargs):
        super().__init__(*args,**kwargs);self.cfg=config;self.feedback_time=None;self.filtered_q=None;self.filtered_v=None;self.last_t=None;self.clearance=.012;self.phase='A';self.pulse_dirs=None
    def target(self,t,actual=None):
        # Exact original warmup and initial hand pose; no reference ball state used.
        if t<=0 or actual is None:
            q=lift.LiftingTargets.target(self,0,actual);self.filtered_q=q.copy();self.filtered_v=np.zeros_like(q);self.last_t=0;return q
        c=self.cfg;duration=c['duration_scale'];clock=t/duration;A=c.get('prep_seconds',1.5);B=A+4.;C=B+3.5;end=C+1.
        self.phase='A' if clock<A else 'B' if clock<B else 'C'
        q=lift.LiftingTargets.target(self,0,actual);r=self.reference
        align=minimum_jerk(clock/2.)
        q[self.trunk_ids]=(1-align)*q[self.trunk_ids]+align*r['aligned'][self.trunk_ids]
        q[self.head_ids]=(1-align)*q[self.head_ids]+align*r['aligned'][self.head_ids]
        # Shared continuous Cartesian curve; wrist tilt changes only in set-point phase.
        if clock<B:center=r['ball_initial']+(r['ball_goal_lift']-r['ball_initial'])*minimum_jerk((clock-A)/(B-A))
        else:
            u=minimum_jerk((clock-B)/(C-B));center=(1-u)**2*r['ball_goal_lift']+2*u*(1-u)*r['ball_goal_intermediate']+u*u*r['ball_goal_shoot']
        tilt=np.radians(c['tilt_deg'])*minimum_jerk((clock-B)/(C-B));W=Rotation.from_rotvec(r['side']*tilt).as_matrix()@r['right_rotation']
        self.cartesian_right=center-W@r['rel0'];self.nominal_cartesian_right=self.cartesian_right.copy();self.feedback_offset=np.zeros(3)
        if c.get('ball_feedback',False):
            bq=int(actual.model.joint('basketball_free').qposadr[0]);error=center-actual.qpos[bq:bq+3]
            self.feedback_offset=minimum_jerk((clock-B)/1.)*np.clip(.5*error,-.025,.025);self.cartesian_right+=self.feedback_offset
        measured=side_contacts(actual.model,actual,'L')
        withdraw=minimum_jerk((clock-(5.5 if c['guide']=='feather' else A+1.5 if c['guide']=='light' else 7.2))/1.5) if c['guide'] in ('phase','feather','light') else 0
        goal_clearance=.012 if c['guide']=='fixed' else .014+.01*np.clip(measured['normal_force']-.5,0,4)+.045*withdraw
        if c['guide']=='light':goal_clearance=.035+.03*np.clip(measured['normal_force']-.3,0,3)+.07*withdraw
        if c['guide']=='feather':goal_clearance=.060+.025*np.clip(measured['normal_force']-.4,0,4)+.05*withdraw
        if c['guide']=='off':goal_clearance=.12
        dt=max(t-(self.last_t or 0),self.m.opt.timestep);self.last_t=t
        self.clearance+=np.clip(goal_clearance-self.clearance,-.02*dt,.08*dt)
        approach=minimum_jerk(clock/1.5)
        ball=actual.qpos[int(actual.model.joint('basketball_free').qposadr[0]):][:3]
        left_R=r['left_rotation'];initial_R=self.initial_left_R;left_rotation=initial_R@Rotation.from_rotvec(approach*Rotation.from_matrix(initial_R.T@left_R).as_rotvec()).as_matrix()
        # Side follows scene rim frame; open guide fingers do not clamp the ball.
        target_left=ball+r['side']*(.121+float(self.m.geom('L_Wrist').size[0])+self.clearance)-[0,0,.015]-left_R@r['left_offset']
        initial_left=r['initial'] # actual initial FK probe is independent of live state
        probe=mujoco.MjData(self.m);probe.qpos[:]=initial_left[:self.m.nq];mujoco.mj_forward(self.m,probe)
        self.cartesian_left=(1-approach)*probe.xpos[self.m.body('L_Wrist').id]+approach*target_left
        # Zero/open left finger targets; preserve right fk_fit_2 grasp throughout A-C.
        q[self.left_finger_ids]=(1-approach)*q[self.left_finger_ids]
        if self.feedback_time is None or t-self.feedback_time>=.1-1e-9:
            background=actual.qpos[:self.m.nq].copy()
            right,_=fit_wrist(self.m,background,'R',self.cartesian_right,W,regularization=.08)
            left,_=fit_wrist(self.m,background,'L',self.cartesian_left,left_rotation,regularization=.08)
            self.fitted_right=right[self.adrs];self.fitted_left=left[self.left_ids];self.feedback_time=t
        q[self.adrs]=self.fitted_right;q[self.left_ids]=self.fitted_left
        # Optional diagnostic tiny extension/rotation AFTER A-C; not a Release.
        if c['probe'] and clock>=C+2.5:
            if self.pulse_dirs is None:
                candidate=reserve(actual.model,actual,r['rel0']);actual_center=actual.qpos[int(actual.model.joint('basketball_free').qposadr[0]):][:3];actual_rel,_=local_ball(actual.model,actual,'R')
                if getattr(self,'ready_since',None) is not None and t-self.ready_since>=.5 and np.linalg.norm(actual_center-r['ball_goal_shoot'])<.03 and np.linalg.norm(actual_rel-r['rel0'])<.02 and side_contacts(actual.model,actual,'R')['count']>0 and candidate['elbow_extension_reserve_deg']>5:self.pulse_dirs=candidate
            if self.pulse_dirs is not None:
                s=minimum_jerk((clock-C-2.5)/.4)
                q[self.adrs[3:6]]+=.06*s*self.pulse_dirs['extension_direction']
                q[self.adrs[6:9]]+=.04*s*self.pulse_dirs['wrist_direction']
        acceleration=np.clip(64*(q-self.filtered_q)-16*self.filtered_v,-8.,8.)
        self.filtered_v=np.clip(self.filtered_v+acceleration*dt,-2.,2.)
        self.filtered_q+=self.filtered_v*dt;self.unfiltered=q.copy();return self.filtered_q.copy()


class CoordinatedEnv(UpperBodyResidualEnv):
    def __init__(self,guide='phase',tilt_deg=8.,duration_scale=1.,probe=False,condition='B',render_mode=None,record=True,ball_feedback=False,prep_seconds=1.5):
        if guide not in ('off','fixed','phase','feather','light') or not np.isfinite(tilt_deg) or not 0<=tilt_deg<=20 or not np.isfinite(duration_scale) or not .5<=duration_scale<=3 or not np.isfinite(prep_seconds) or not 1<=prep_seconds<=3.5:raise ValueError('Invalid candidate bounds')
        self.coord_ready=False;self.config=dict(guide=guide,tilt_deg=float(tilt_deg),duration_scale=float(duration_scale),probe=probe,ball_feedback=ball_feedback,prep_seconds=float(prep_seconds))
        super().__init__(condition=condition,stage=4,reference=ROOT/'outputs/m5_1_upper_body/references/guide_safe/reference.npz',max_seconds=(prep_seconds+10.6 if probe else prep_seconds+10.)*duration_scale,record=record,render_mode=render_mode)
        self.ref=dict(np.load(OUT/'reference/reference.npz'));self.reference_path=OUT/'reference/reference.npz'
        self.coord_ready=True;self.reset(seed=7)
    def _adapter(self):
        if not self.coord_ready:return super()._adapter()
        self.bridge=lift.HumanoidControllerBridge(MODELS/'body_01.xml',self.m,SOURCE)
        self._upper_cache_time=None
        self.adapter=CoordinatedTargets(self.bridge,self.candidate,self.arm_ids,self.ref,4,self.condition,config=self.config)
    def metrics(self):
        if not self.coord_ready:return super().metrics()
        # Full wrist FK telemetry at control rate; physical safety/contact remains 450Hz.
        previous=getattr(self,'_upper_cache_time',None)
        if previous is None or self.d.time-previous>=self.control_dt-1e-9 or self.d.time<previous:
            z=super().metrics();self._upper_cache=dict(z);self._upper_cache_time=self.d.time
        else:
            z=dict(self._upper_cache);z.update(BasketballResidualEnv.metrics(self))
            left=side_contacts(self.m,self.d,'L');right=side_contacts(self.m,self.d,'R')
            z.update(left_contacts=left['count'],left_force=left['force'],left_normal=left['normal_force'],left_max_normal=left['max_normal'],left_fingers=left['fingers'],left_flags=left['finger_flags'],left_slip=left['slip'],right_force=right['force'])
        if not z['finite']:return z
        # Reference trunk frame uses actual free-root state; source-root mismatch removed.
        p=mujoco.MjData(self.m);p.qpos[:]=self.d.qpos;p.qpos[self.trunk_ids]=self.last_reference[self.trunk_ids];mujoco.mj_kinematics(self.m,p)
        chest=self.m.body('Chest').id;z['trunk_reference_matrix']=p.xmat[chest].reshape(3,3).copy();z['trunk_error_deg']=np.degrees(Rotation.from_matrix(z['trunk_reference_matrix'].T@self.d.xmat[chest].reshape(3,3)).magnitude())
        return z
    def stage_success(self,z):
        # Weak/absent Guide is allowed; force safety matters rather than permanent contact.
        if not self.coord_ready:return super().stage_success(z)
        effective=z['contacts']>0 or (self.seen and self.loss_start is not None and z['time_s']-self.loss_start<.05)
        passed=bool(z['finite'] and not z['fallen'] and not any(z['warnings']) and effective and z['drift_m']<.02 and z['goal_distance_m']<.03 and z['head_error_deg']<15 and z['both_feet'] and z['com_margin_m']>0 and z['left_normal']<3)
        if passed:
            if getattr(self.adapter,'ready_since',None) is None:self.adapter.ready_since=z.get('time_s',self.d.time)
        else:self.adapter.ready_since=None
        return passed


def main():
    p=argparse.ArgumentParser();p.add_argument('--generate',action='store_true');a=p.parse_args()
    if a.generate:print(generate())
if __name__=='__main__':main()
