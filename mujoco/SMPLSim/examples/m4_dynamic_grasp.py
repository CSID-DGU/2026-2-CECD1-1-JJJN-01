"""M4-2D-3B: bounded contact feedback target adapters; unchanged PD/COM.
No ball state writes after initialization, equality, proxy, release or external force.
"""
import argparse,csv,hashlib,json,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from examples import m4_spl_contact_integration as prior
from examples.m3_control import write_csv,FINGERS
lift=prior.lift
OUT=ROOT/'outputs/m4_2d_3b_dynamic'


def norm_clip(x,limit):
    return x*min(1.,limit/max(np.linalg.norm(x),1e-12))


def continuous_rotvec(rotations):
    """Lift an SO(3) path through the pi cut using equivalent 2*pi branches."""
    result=[];previous=None
    for raw in Rotation.from_matrix(rotations).as_rotvec():
        angle=np.linalg.norm(raw)
        axis=raw/angle if angle>1e-9 else (previous/max(np.linalg.norm(previous),1e-12) if previous is not None else np.array([1.,0.,0.]))
        options=[raw+2*np.pi*k*axis for k in range(-2,3)]
        value=raw if previous is None else min(options,key=lambda x:np.linalg.norm(x-previous))
        result.append(value);previous=value
    return np.array(result)


class ContinuousGoals:
    """Discrete acceleration-bounded servo; velocity remains continuous.
    This exports goal velocities but the existing position PD does not consume them.
    """
    def __init__(self,q,vmax,amax):
        self.q=q.copy();self.v=np.zeros_like(q);self.vmax=np.asarray(vmax);self.amax=np.asarray(amax)
    def step(self,desired,dt):
        velocity=np.clip(12*(desired-self.q),-self.vmax,self.vmax)
        self.v+=np.clip(velocity-self.v,-self.amax*dt,self.amax*dt)
        self.q+=self.v*dt
        return self.q.copy()


def local_ball(m,d):
    w=m.body('R_Wrist').id;q=int(m.joint('basketball_free').qposadr[0]);v=int(m.joint('basketball_free').dofadr[0]);R=d.xmat[w].reshape(3,3)
    rel=R.T@(d.qpos[q:q+3]-d.xpos[w]);jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv))
    mujoco.mj_jac(m,d,jp,jr,d.qpos[q:q+3],w)
    # Point Jacobian includes omega cross offset: rotating-frame derivative.
    vel=R.T@(d.qvel[v:v+3]-jp@d.qvel)
    return rel,vel


class DynamicTargets(prior.IntegrationTargets):
    def __init__(self,b,c,condition,speed):
        super().__init__(b,c,1,2,'C',15);self.kind=condition;self.speed=speed;self.last_time=None
        self.arm=np.array([int(self.m.joint(n).qposadr[0]) for n in lift.ARM]);self.names=lift.ARM+list(c['angles']);self.ids=np.array([int(self.m.joint(n).qposadr[0]) for n in self.names]);self.dofs=np.array([int(self.m.joint(n).dofadr[0]) for n in self.names])
        self.goal_servo=None;self.rel0=None;self.finger_G=None;self.feedback=np.zeros(3);self.R_command=self.R.copy()
        self.mapping=json.loads((lift.PREVIOUS/'axis_mapping.json').read_text())
        self.variables=[r for r in self.mapping if r['role']=='flexion' or (r['role']=='abduction' and r['segment'].endswith('1')) or r['segment']=='R_Thumb1']
        self.var_ids=np.array([r['qposadr'] for r in self.variables]);self.t0=None
        rotations=[]
        for frame in range(len(b.motion.poses)):
            self.probe.qpos[:]=b.target(frame/b.motion.fps);mujoco.mj_forward(self.m,self.probe);rotations.append(self.R.T@self.probe.xmat[self.m.body('R_Wrist').id].reshape(3,3))
        self.source_vectors=continuous_rotvec(np.array(rotations))
    def calibrate(self,d):
        self.rel0,_=local_ball(d.model,d);q=d.qpos[:self.m.nq].copy();self.probe.qpos[:]=q;mujoco.mj_forward(self.m,self.probe)
        w=self.m.body('R_Wrist').id;center=self.probe.xpos[w]+self.probe.xmat[w].reshape(3,3)@self.rel0
        self.gaps0=np.array([r['surface_gap_mm']/1000 for r in lift.hand.distances(self.m,self.probe,center)])
        G=[]
        for adr in self.var_ids:
            pair=[]
            for change in [-.01,.01]:
                self.probe.qpos[:]=q;self.probe.qpos[adr]+=change;mujoco.mj_forward(self.m,self.probe)
                pair.append(np.array([r['surface_gap_mm']/1000 for r in lift.hand.distances(self.m,self.probe,center)]))
            G.append((pair[1]-pair[0])/.02)
        self.finger_G=np.array(G).T
    def target(self,t,actual=None):
        # Parent uses simulation t; transformed clock retains 1s hold and 2s blend.
        clock=t if t<=3 else 3+(t-3)*self.speed
        raw=super().target(clock,actual)
        if self.kind=='A' or actual is None or self.rel0 is None:return raw
        dt=float(t-self.last_time) if self.last_time is not None else self.m.opt.timestep
        self.last_time=t;dt=max(dt,1e-8)
        if self.goal_servo is None:
            self.goal_servo=ContinuousGoals(raw[self.ids],np.r_[np.full(6,6.),np.full(3,6.),np.full(45,2.)],np.r_[np.full(6,30.),np.full(3,40.),np.full(45,10.)])
        index=min(self.source_time*self.bridge.motion.fps,len(self.source_vectors)-1);i=min(int(index),len(self.source_vectors)-2);weight=index-i
        vector=(1-weight)*self.source_vectors[i]+weight*self.source_vectors[i+1]
        self.R_goal=self.R@Rotation.from_rotvec(prior.smooth((t-1)/2)*norm_clip(vector,np.radians(15))).as_matrix()
        desired=raw.copy();rel,rv=local_ball(actual.model,actual);w=self.m.body('R_Wrist').id;W=actual.xmat[w].reshape(3,3)
        enabled=t>=1 and self.rel0 is not None and np.linalg.norm(rel-self.rel0)<.10
        if self.kind in ('C','D') and enabled:
            error=rel-self.rel0;pred=norm_clip(error+.06*rv,.035)
            horizontal=W@pred;horizontal[2]=0
            tilt=norm_clip(np.cross(W@np.array([0.,-1.,0.]),-4*horizontal),np.radians(8))
            self.feedback+=(tilt-self.feedback)*min(1.,dt/.12)
            self.R_goal=Rotation.from_rotvec(self.feedback).as_matrix()@self.R_goal
            self.probe.qpos[:]=actual.qpos[:self.m.nq];mujoco.mj_forward(self.m,self.probe)
            future=actual.xpos[w]+W@(rel+.04*rv)
            gaps=np.array([r['surface_gap_mm']/1000 for r in lift.hand.distances(self.m,self.probe,future)])
            weights=np.exp(-np.maximum(self.gaps0,0)/.02);G=self.finger_G*weights[:,None]
            correction=-.35*np.linalg.solve(G.T@G+.002**2*np.eye(G.shape[1]),G.T@((gaps-self.gaps0)*weights))
            desired[self.var_ids]=np.array([self.candidate['angles'][r['joint']] for r in self.variables])+np.clip(correction,-.12,.12)
            if self.kind=='D':
                jp,jr=np.zeros((3,actual.model.nv)),np.zeros((3,actual.model.nv));mujoco.mj_jacBody(actual.model,actual,jp,jr,w)
                J=jp[:,self.dofs[:6]];world=norm_clip(W@(error+.04*rv),.025)
                dq=J.T@np.linalg.solve(J@J.T+.03**2*np.eye(3),.4*world)
                desired[self.arm[:6]]+=np.clip(dq,-.12,.12)
        # Incremental SO(3) IK avoids Euler branch selection entirely.
        self.probe.qpos[:]=desired;self.probe.qpos[self.wrist]=self.goal_servo.q[6:9];mujoco.mj_forward(self.m,self.probe)
        parent=actual.xmat[self.m.body_parentid[w]].reshape(3,3);local=Rotation.from_euler('XYZ',self.goal_servo.q[6:9]).as_matrix();current=parent@local
        err=norm_clip(Rotation.from_matrix(self.R_goal@current.T).as_rotvec(),.6)
        # Transform nominal hinge axes to the actual parent frame.
        nominal_parent=self.probe.xmat[self.m.body_parentid[w]].reshape(3,3)
        J=parent@nominal_parent.T@np.column_stack([self.probe.xaxis[self.m.joint(n).id] for n in lift.ARM[6:]])
        dq=J.T@np.linalg.solve(J@J.T+.08**2*np.eye(3),err)
        desired[self.wrist]=self.goal_servo.q[6:9]+np.clip(dq,-.5,.5)
        low=np.array([self.m.jnt_range[self.m.joint(n).id,0] for n in self.names]);high=np.array([self.m.jnt_range[self.m.joint(n).id,1] for n in self.names])
        desired[self.ids]=np.clip(desired[self.ids],low+.001,high-.001)
        desired[self.ids]=self.goal_servo.step(desired[self.ids],dt)
        self.R_command=self.R_goal.copy();return desired


class GuideTargets(DynamicTargets):
    """Separate left side-contact trial, not a clamp or external constraint.
    Targets a small positive gap; excessive measured guide forces fail evaluation.
    """
    def __init__(self,b,c,speed,center):
        super().__init__(b,c,'C',speed)
        self.left_names=[n.replace('R_','L_',1) for n in lift.ARM];self.left_ids=np.array([int(self.m.joint(n).qposadr[0]) for n in self.left_names]);self.guide_center=np.array(center);self.guide_q=None;self.guide_servo=None;self.guide_last=None;self.guide_last_fit=None;self.ik_error=None
        self.fit_guide(self.guide_center);self.initial_ik_error=self.ik_error
    def fit_guide(self,center,actual=None):
        from scipy.optimize import least_squares
        q=actual.qpos[:self.m.nq].copy() if actual is not None else lift.hand.GraspTargets(self.bridge,self.candidate).target(0);base=self.base[self.left_ids]
        previous=self.guide_q.copy() if self.guide_q is not None else base.copy();g=self.m.geom('L_Wrist').id;w=self.m.body('L_Wrist').id;r=float(self.m.geom_size[g,0]);desired=center+[0,.121+r+.0005,0]
        def residual(x):
            self.probe.qpos[:]=q;self.probe.qpos[self.left_ids]=x;mujoco.mj_forward(self.m,self.probe)
            normal=-self.probe.xmat[w].reshape(3,3)[:,1]
            return np.r_[30*(self.probe.geom_xpos[g]-desired),1.5*(normal-[0,-1,0]),.04*(x-previous)]
        bounds=np.array([self.m.jnt_range[self.m.joint(n).id] for n in self.left_names]);lo=np.maximum(base-1.8,bounds[:,0]+.001);hi=np.minimum(base+1.8,bounds[:,1]-.001)
        fit=least_squares(residual,np.clip(previous,lo,hi),bounds=(lo,hi),max_nfev=25,diff_step=1e-4)
        self.guide_q=fit.x;self.ik_error=float(np.linalg.norm(fit.fun[:3])/30)
    def target(self,t,actual=None):
        q=super().target(t,actual)
        if actual is None or self.rel0 is None:q[self.left_ids]=self.guide_q;return q
        dt=max(t-(self.guide_last if self.guide_last is not None else t-self.m.opt.timestep),1e-8);self.guide_last=t
        if self.guide_servo is None:self.guide_servo=ContinuousGoals(self.guide_q,np.full(9,3.),np.full(9,15.))
        if self.guide_last_fit is None or t-self.guide_last_fit>=.05:
            ball=int(actual.model.joint('basketball_free').qposadr[0]);center=actual.qpos[ball:ball+3]
            if np.linalg.norm(center-self.guide_center)<.6:self.fit_guide(center,actual)
            self.guide_last_fit=t
        q[self.left_ids]=self.guide_servo.step(self.guide_q,dt);return q


def left_force(m,d):
    ball=m.geom('basketball_geom').id;force=np.zeros(3);count=0;maximum=0.
    for i in range(d.ncon):
        ct=d.contact[i]
        if ball not in (ct.geom1,ct.geom2):continue
        other=int(ct.geom2 if ct.geom1==ball else ct.geom1);name=m.geom(other).name
        if not (name=='L_Wrist' or (name.startswith('L_') and any(f in name for f in FINGERS))):continue
        f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
        if f[0]>.01:count+=1;force+=ct.frame.reshape(3,3).T@f[:3]*(-1 if ct.geom1==ball else 1);maximum=max(maximum,float(f[0]))
    return count,force,maximum

def protect(out):
    paths=[]
    for folder in ['examples','tests','smpl_sim']:
        paths+=list((ROOT/folder).rglob('*.py'))
    paths+=list(ROOT.glob('M*.md'))
    for folder in ['m4_2d_contact','m4_2d_1_hand','m4_2d_2_lifting','m4_2d_3a_integration']:
        paths+=list((ROOT/'outputs'/folder).rglob('*'))
    paths+=list((ROOT.parents[2]/'basketball-r1/SMPLSim/outputs/smplh_physics').glob('*'))
    paths.append(ROOT/'outputs/m3_spl_fitting/spl_P0002_T0049.npz')
    paths=[p for p in paths if p.is_file() and 'dynamic_grasp' not in p.name.lower()]
    (out/'preservation_before.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))


def candidates():
    base=next(c for c in json.loads((lift.PREVIOUS/'candidates.json').read_text()) if c['name']=='fk_fit_2');base=json.loads(json.dumps(base));base['name']='fk_fit_2'
    result=[base];mapping=json.loads((lift.PREVIOUS/'axis_mapping.json').read_text())
    for name,flex,spread,thumb,cx in [('spread',0,.07,.06,0),('cupped',.06,.04,.08,0),('back_support',.02,-.04,-.06,.006),('open',-.04,.06,.04,-.006)]:
        c=json.loads(json.dumps(base));c['name']=name;c['center_x']+=cx
        for r in mapping:
            delta=flex if r['role']=='flexion' else (spread if r['role']=='abduction' and r['segment'].endswith('1') else 0)
            if r['segment']=='R_Thumb1' and r['role']=='abduction':delta=thumb
            c['angles'][r['joint']]+=r['sign']*delta
        result.append(c)
    return result


def prepare(args):
    args.output.mkdir(parents=True,exist_ok=True);protect(args.output);items=candidates();(args.output/'candidates.json').write_text(json.dumps(items,indent=2));results=[]
    for c in items:
        out=args.output/'candidates'/c['name'];opts=argparse.Namespace(**vars(args));opts.output=out;opts.gui=False
        lift.hand.setup(opts);(out/'axis_mapping.json').write_text((lift.PREVIOUS/'axis_mapping.json').read_text())
        s=lift.hand.trial(opts,c,3.5);s['static_gate']=bool(s['held_3s'] and s['max_relative_drift_m']<.02 and not s['fallen'] and not any(s['warnings']) and s['both_feet_ratio']>=.99 and s['min_com_margin_m']>=0)
        if s['static_gate']:
            # Existing Lifting routine reads this conventional static summary path.
            conventional=out/'trials/fk_fit_2';conventional.mkdir(exist_ok=True,parents=True);(conventional/'summary.json').write_text(json.dumps(s))
            m=mujoco.MjModel.from_xml_path(str(args.models/'body_01.xml'));b=prior.HumanoidControllerBridge(args.models/'body_01.xml',m,args.source)
            lift.arm_path(b.model,lift.hand.GraspTargets(b,c).target(0),.25,out)
            opts.start=1.;opts.ramp=4.;opts.end_hold=1.5;opts.height=.25
            ls=lift.simulate(opts,'C',c);s['lifting_gate']=ls['success'];s['lifting_relative_drift_m']=ls['max_relative_drift_m']
        else:s['lifting_gate']=False
        results.append(s)
    (args.output/'candidate_gates.json').write_text(json.dumps(results,indent=2));valid=[s for s in results if s['static_gate'] and s['lifting_gate']]
    if not valid:raise RuntimeError('No grasp passed both gates; no dynamic motion')
    # Conservative measured objective: static + lifting maximum relative displacement.
    best=min(valid,key=lambda s:s['max_relative_drift_m']+s['lifting_relative_drift_m']);selected=next(c for c in items if c['name']==best['candidate'])
    (args.output/'selected.json').write_text(json.dumps(selected,indent=2));return selected


def contact_records(m,d):
    ball=m.geom('basketball_geom').id;records=[]
    for i in range(d.ncon):
        ct=d.contact[i]
        if ball not in (ct.geom1,ct.geom2):continue
        other=int(ct.geom2 if ct.geom1==ball else ct.geom1);name=m.geom(other).name
        if name not in ('R_Wrist','L_Wrist') and not (name.startswith(('R_','L_')) and any(f in name for f in FINGERS)):continue
        f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
        if f[0]<=.01:continue
        sign=-1 if ct.geom1==ball else 1;normal=ct.frame[:3]*sign;force=ct.frame.reshape(3,3).T@f[:3]*sign
        row=dict(geom=name,gap_mm=ct.dist*1000,normal_force_N=f[0],force_z_N=force[2])
        for a,k in enumerate('xyz'):row['position_'+k+'_m']=ct.pos[a];row['normal_'+k]=normal[a]
        records.append(row)
    return records


def run(args,condition,speed,c):
    baseline=condition=='A';c=next(x for x in candidates() if x['name']=='fk_fit_2') if baseline else c
    folder=args.output/f'{condition}_{speed:g}';folder.mkdir(exist_ok=True);scene=args.output/'candidates'/c['name']/'contact.xml'
    if condition in ('E','F'):
        import xml.etree.ElementTree as ET
        tree=ET.parse(scene)
        for geom in tree.findall('.//geom'):
            name=geom.get('name','')
            if name=='L_Wrist' or (name.startswith('L_') and any(f in name for f in FINGERS)):geom.set('contype',str(int(geom.get('contype','1'))|32))
        scene=folder/'guide_contact.xml';tree.write(scene,encoding='unicode')
    m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m);b=prior.HumanoidControllerBridge(args.models/'body_01.xml',m,args.source)
    center=json.loads((args.output/'candidates'/c['name']/'trials'/c['name']/'summary.json').read_text())['initial_position_m']
    a=GuideTargets(b,c,speed,center) if condition=='E' else DynamicTargets(b,c,'C' if condition=='F' else condition,speed)
    shift=2.5 if condition in ('E','F') else 0.
    q,v=lift.initialize(m,d,b,a,c);a.calibrate(d);rel0=a.rel0.copy();root0=d.qpos[2];rows=[];states=[];times=[];joints=[];conrows=[];seen=False;loss=None;loss_start=None
    viewer=None
    if args.gui:
        from mujoco import viewer as mv
        viewer=mv.launch_passive(m,d);viewer.cam.lookat[:]=[.25,0,1.1];viewer.cam.distance=2.5;viewer.cam.azimuth=110;viewer.cam.elevation=-10
    duration=3+shift+b.motion.duration/speed+.25;started=time.monotonic();lastgoal=None;maxstep=0.;lastvel=None;maxacc=0.
    for step in range(round(duration/m.opt.timestep)):
        goal=a.target(d.time-shift,d);source=b.target(a.source_time);bp=d.qpos[q:q+7].copy();bv=d.qvel[v:v+6].copy();a.control(d,goal)
        assert np.array_equal(bp,d.qpos[q:q+7]) and np.array_equal(bv,d.qvel[v:v+6]);assert m.neq==0 and not d.qfrc_applied.any() and not d.xfrc_applied.any()
        assert np.isfinite(goal).all()
        goal_vel=np.zeros(len(a.ids)) if lastgoal is None else (goal[a.ids]-lastgoal)/m.opt.timestep
        if lastgoal is not None:
            gv=(goal[a.ids]-lastgoal)/m.opt.timestep;maxstep=max(maxstep,float(np.max(abs(goal[a.ids]-lastgoal))))
            if lastvel is not None:maxacc=max(maxacc,float(np.max(abs(gv-lastvel))/m.opt.timestep))
            lastvel=gv
        lastgoal=goal[a.ids].copy();mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
        assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
        n,nf,F,ff,depth,slip=lift.contact.hand_contacts(m,d);rel,rv=local_ball(m,d);t=float(d.time);w=m.body('R_Wrist').id;W=d.xmat[w].reshape(3,3);feet,margin,_=lift.contact.contacts(m,d)
        seen=seen or n>0
        if n:loss_start=None
        elif seen and loss_start is None:loss_start=t
        if loss is None and loss_start is not None and t-loss_start>=.05:loss=loss_start
        ln,lf,lmax=left_force(m,d)
        if condition=='E' and 3.49<t<3.5 and (loss is not None or np.linalg.norm(rel-rel0)>.02 or lmax>20 or any(r['left_max_normal_force_N']>20 for r in rows) or np.mean([r['left_contacts']>0 for r in rows if r['time_s']>=2.5])<.2):break
        tilt=np.degrees(np.arccos(np.clip(d.xmat[m.body('Pelvis').id].reshape(3,3)[2,1],-1,1)));tau=d.ctrl[:b.model.nu];sat=np.abs(tau)>=.99*b.controller.limit
        rows.append(dict(time_s=t,source_time_s=a.source_time,contacts=n,force_z_N=F[2],slip_ms=slip,drift_m=np.linalg.norm(rel-rel0),relative_speed_ms=np.linalg.norm(rv),ball_x_m=d.qpos[q],ball_y_m=d.qpos[q+1],ball_z_m=d.qpos[q+2],ball_vx_ms=d.qvel[v],ball_vy_ms=d.qvel[v+1],ball_vz_ms=d.qvel[v+2],wrist_z_m=d.xpos[w,2],orientation_error_deg=np.degrees(Rotation.from_matrix(a.R_goal.T@W).magnitude()),source_orientation_error_deg=np.degrees(Rotation.from_matrix(a.R_source.T@W).magnitude()),arm_rmse_rad=np.sqrt(np.mean((d.qpos[a.arm]-goal[a.arm])**2)),source_arm_rmse_rad=np.sqrt(np.mean((d.qpos[a.arm]-source[a.arm])**2)),com_margin_m=margin,both_feet=int(min(feet.values())>.1),fallen=int(d.qpos[2]<.65*root0 or tilt>35),left_contacts=ln,left_force_norm_N=np.linalg.norm(lf),left_max_normal_force_N=lmax,torque_saturation_fraction=np.mean(sat),arm_torque_saturation_fraction=np.mean(sat[a.ids[:9]-7]),max_finger_torque_Nm=np.max(abs(tau[a.ids[9:]-7]))))
        if step%15==0:
            states.append(d.qpos.copy());times.append(t);record=dict(time_s=t,source_time_s=a.source_time)
            for i,name in enumerate(a.names):
                adr=a.ids[i];record[name+'_goal_rad']=goal[adr];record[name+'_goal_velocity_rad_s']=goal_vel[i];record[name+'_error_rad']=d.qpos[adr]-goal[adr];record[name+'_source_error_rad']=d.qpos[adr]-source[adr];record[name+'_torque_Nm']=tau[adr-7]
            if condition in ('E','F'):
                for k,name in enumerate([n.replace('R_','L_',1) for n in lift.ARM]):
                    adr=int(m.joint(name).qposadr[0]);record[name+'_goal_rad']=goal[adr];record[name+'_error_rad']=d.qpos[adr]-goal[adr];record[name+'_source_error_rad']=d.qpos[adr]-source[adr];record[name+'_torque_Nm']=tau[adr-7]
            joints.append(record)
            conrows.extend([dict(time_s=t,source_time_s=a.source_time,**r) for r in contact_records(m,d)])
        if viewer:
            if not viewer.is_running():break
            viewer.sync();time.sleep(max(0,t-(time.monotonic()-started)))
    if viewer:viewer.close()
    write_csv(folder/'telemetry.csv',rows);write_csv(folder/'joint_tracking_torques.csv',joints)
    if conrows:write_csv(folder/'contacts.csv',conrows)
    np.savez_compressed(folder/'states.npz',qpos=states,time_s=times);(folder/'scene_path.txt').write_text(str(scene))
    candidate_time=2.4166666666666665;end=3+shift+candidate_time/speed
    def metrics(limit):
        x=[r for r in rows if 1<=r['time_s']<=limit];gap=longest=0
        for r in x:gap=gap+1 if not r['contacts'] else 0;longest=max(gap,longest)
        return dict(success=bool(seen and (loss is None or loss>limit) and max(r['drift_m'] for r in x)<.05 and not any(r['fallen'] for r in x) and not any(w.number for w in d.warning) and min(r['com_margin_m'] for r in x)>=0 and np.mean([r['both_feet'] for r in x])>=.99 and d.time>=limit-m.opt.timestep),max_relative_drift_m=max(r['drift_m'] for r in x),contact_ratio=float(np.mean([r['contacts']>0 for r in x])),max_contact_gap_s=longest*m.opt.timestep,mean_force_z_N=float(np.mean([r['force_z_N'] for r in x])),slip_rms_ms=float(np.sqrt(np.mean([r['slip_ms']**2 for r in x if r['contacts']]))),arm_rmse_rad=float(np.sqrt(np.mean([r['arm_rmse_rad']**2 for r in x]))),source_arm_rmse_rad=float(np.sqrt(np.mean([r['source_arm_rmse_rad']**2 for r in x]))),max_orientation_error_deg=max(r['orientation_error_deg'] for r in x),min_com_margin_m=min(r['com_margin_m'] for r in x),both_feet_ratio=float(np.mean([r['both_feet'] for r in x])),arm_saturation_ratio=float(np.mean([r['arm_torque_saturation_fraction'] for r in x])),fallen=bool(any(r['fallen'] for r in x)))
    s=dict(wrist_goal_velocity_limit_rad_s=6. if condition!='A' else None,wrist_goal_acceleration_limit_rad_s2=40. if condition!='A' else None,guide_initial_ik_position_error_m=a.initial_ik_error if condition=='E' else None,guide_ik_position_error_m=a.ik_error if condition=='E' else None,guide_static_gate=bool(condition!='E' or (loss is None or loss>3.5) and d.time>=3.5 and max(r['left_max_normal_force_N'] for r in rows if r['time_s']<=3.5)<20 and np.mean([r['left_contacts']>0 for r in rows if 2.5<=r['time_s']<=3.5])>=.2),max_left_contact_force_N=max(r['left_max_normal_force_N'] for r in rows),left_contact_ratio=float(np.mean([r['left_contacts']>0 for r in rows])),condition=condition,speed=speed,candidate=c['name'],release_candidate_source_s=candidate_time,release_candidate_is_ground_truth=False,first_loss_sim_s=loss,first_loss_source_s=max(0,(loss-3-shift)*speed) if loss is not None else None,pre_candidate=metrics(end),full=metrics(duration-m.opt.timestep),max_goal_step_rad=maxstep,max_goal_acceleration_rad_s2=maxacc,warnings=[int(w.number) for w in d.warning],neq=m.neq,ball_state_writes_after_initialization=0,release_implemented=False)
    (folder/'summary.json').write_text(json.dumps(s,indent=2));print(json.dumps(s),flush=True);return s


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT);p.add_argument('--models',type=Path,default=ROOT.parents[2]/'basketball-r1/SMPLSim/outputs/smplh_physics');p.add_argument('--source',type=Path,default=ROOT/'outputs/m3_spl_fitting/spl_P0002_T0049.npz');p.add_argument('--prepare',action='store_true');p.add_argument('--condition',choices=['all','A','B','C','D','E','F'],default='all');p.add_argument('--speed',type=float,choices=[.25,.5,1.],default=1.);p.add_argument('--all-speeds',action='store_true');p.add_argument('--gui',action='store_true');args=p.parse_args()
    if args.prepare:prepare(args);return
    c=json.loads((args.output/'selected.json').read_text());gates=json.loads((args.output/'candidate_gates.json').read_text())
    protected=json.loads((args.output/'preservation_before.json').read_text())
    required=[args.source,args.models/'body_01.xml',ROOT/'examples/m3_control.py',ROOT/'smpl_sim/physics/basketball.py']
    for path in required:
        if str(path.resolve()) not in protected or hashlib.sha256(path.read_bytes()).hexdigest()!=protected[str(path.resolve())]:raise RuntimeError('Protected input changed since gates: '+str(path))
    for name in ['fk_fit_2',c['name']]:
        if not any(x['candidate']==name and x['static_gate'] and x['lifting_gate'] for x in gates):raise RuntimeError('Candidate did not pass static/lifting gate')
    for speed in [.25,.5,1.] if args.all_speeds else [args.speed]:
        for condition in ['A','B','C','D'] if args.condition=='all' else [args.condition]:run(args,condition,speed,c)

if __name__=='__main__':main()
