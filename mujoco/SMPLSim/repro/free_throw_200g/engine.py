"""Portable M6 reproduction. Controller adapted from SMPLSim (BSD-3-Clause).
See licenses/CODE_LICENSE.txt. No import of old experiments or local datasets.
"""
from pathlib import Path
import copy,hashlib,tomllib,xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.interpolate import CubicSpline
from scipy.linalg import cho_factor,cho_solve
HERE=Path(__file__).resolve().parent

def settings():return tomllib.loads((HERE/'settings.toml').read_text())
def smooth(u):
    u=np.clip(u,0,1);return u*u*u*(10+u*(-15+6*u))
def fingerprint(m,fields):
    h=hashlib.sha256()
    for k in fields:h.update(k.encode());h.update(np.ascontiguousarray(getattr(m,k)).tobytes())
    h.update(m.names);return h.hexdigest()

def load_models(humanoid):
    cfg=settings();human=mujoco.MjModel.from_xml_path(str(humanoid))
    if fingerprint(human,cfg['fingerprint_fields'])!=cfg['model_physics_sha256']:
        raise ValueError('Different humanoid physics. Use the matching body_01 XML (male, 16 zero betas). See README.md; do not treat a different body as the same experiment.')
    tree=ET.parse(HERE/'environment.xml');root=tree.getroot();external=ET.parse(humanoid).getroot();pelvis=external.find("worldbody/body[@name='Pelvis']");actuator=external.find('actuator')
    if pelvis is None or actuator is None:raise ValueError('Expected standalone M2 body_01.xml')
    root.find('worldbody').insert(2,copy.deepcopy(pelvis));root.insert(4,copy.deepcopy(actuator))
    m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));m.opt.timestep=cfg['physics_dt_s'];human.opt.timestep=m.opt.timestep;d=mujoco.MjData(m)
    b=m.body('basketball').id;m.body_inertia[b]*=cfg['ball_mass_kg']/m.body_mass[b];m.body_mass[b]=cfg['ball_mass_kg'];mujoco.mj_setConst(m,d)
    assert (human.nq,human.nv,human.nu)==(160,159,153) and (m.nq,m.nv,m.nu)==(167,165,153)
    return human,m,d,cfg

class Motion:
    def __init__(self,cfg):
        p=HERE/'data/reference.csv'
        if hashlib.sha256(p.read_bytes()).hexdigest()!=cfg['reference_sha256']:raise ValueError('Reference CSV checksum mismatch')
        s=np.loadtxt(p,delimiter=',',skiprows=1);self.times=s[:,0];self.q=s[:,1:];self.duration=float(self.times[-1]);self.base=CubicSpline(self.times,self.q[:,7:],axis=0,bc_type='clamped');self.arm=np.array(cfg['follow_arm_ids'])-7;self.end=np.array(cfg['follow_arm_goal']);self.start=cfg['follow_start_s'];self.ramp=cfg['follow_duration_s']
    def spline(self,t,nu=0):
        t=np.asarray(t);u=np.clip((t-self.start)/self.ramp,0,1);a=smooth(u);out=self.base(t,nu).copy();q=self.base(t,0)[...,self.arm]
        if nu==0:out[...,self.arm]=(1-a[...,None])*q+a[...,None]*self.end
        elif nu==1:
            da=30*u*u*(1-u)*(1-u)/self.ramp;out[...,self.arm]=(1-a[...,None])*out[...,self.arm]+da[...,None]*(self.end-q)
        return out
    def target(self,t):
        q=self.q[0].copy();q[7:]=self.spline(np.clip(t,0,self.duration));return q

class PD:
    """Same implicit Stable PD + COM corrections + desired-velocity weights."""
    def __init__(self,m,cfg):
        self.m=m;self.d=mujoco.MjData(m);names=[m.joint(int(j)).name for j in m.actuator_trnid[:,0]];self.index={n:i for i,n in enumerate(names)};kp,kd,limit=np.full(m.nu,300.),np.full(m.nu,30.),np.full(m.nu,150.)
        for i,n in enumerate(names):
            if any(k in n for k in ('Hip','Knee','Ankle','Toe','Torso','Spine','Chest')):kp[i],kd[i],limit[i]=1000.,100.,500.
            if any(k in n for k in ('Index','Middle','Pinky','Ring','Thumb')):kp[i],kd[i],limit[i]=5.,.5,2.
        leg=np.mean([sum(np.linalg.norm(m.body(n).pos) for n in (s+'_Knee',s+'_Ankle')) for s in ('L','R')]);scale=float(m.body_mass.sum()/80*(leg/.82)**2);self.kp=kp*scale;self.kd=kd*scale;self.limit=limit*scale;self.jac=np.zeros((3,m.nv));self.weights=np.array([cfg['upper_velocity_weight'] if any(b in n for b in ('Shoulder','Elbow','Wrist','Torso','Spine','Chest')) else cfg['lower_velocity_weight'] if any(b in n for b in ('Hip','Knee')) else 0. for n in names])
    def control(self,integrated,goal,velocity):
        m,p=self.m,self.d;p.qpos[:]=integrated.qpos[:160];p.qvel[:]=integrated.qvel[:159];p.ctrl[:]=integrated.ctrl;p.time=integrated.time;mujoco.mj_forward(m,p)
        target=goal[7:].copy();feet=np.mean([p.xpos[m.body(n).id] for n in ('L_Ankle','R_Ankle')],axis=0);error=p.subtree_com[1,:2]-feet[:2]-[.02,0];mujoco.mj_jacSubtreeCom(m,p,self.jac,1);v=(self.jac@p.qvel)[:2];correction=np.clip(8*error+2*v,-.6,.6)
        for side in ('L','R'):
            target[self.index[side+'_Ankle_x']]+=correction[0];target[self.index[side+'_Ankle_z']]-=correction[1];target[self.index[side+'_Hip_z']]+=.5*correction[1]
        dt=m.opt.timestep;qerr=np.r_[np.zeros(6),p.qpos[7:]+p.qvel[6:]*dt-target];verr=p.qvel.copy();verr[6:]-=velocity*self.weights;kp=np.r_[np.zeros(6),self.kp];kd=np.r_[np.zeros(6),self.kd];M=np.empty((m.nv,m.nv))
        if hasattr(p,'qM'):mujoco.mj_fullM(m,M,p.qM)
        else:mujoco.mj_fullM(m,p,M)
        C=p.qfrc_bias.copy();Kp=np.diag(kp);Kd=np.diag(kd);acc=cho_solve(cho_factor(M+Kd*dt,overwrite_a=True,check_finite=False),-C[:,None]-Kp.dot(qerr[:,None])-Kd.dot(verr[:,None]),overwrite_b=True,check_finite=False).squeeze()
        integrated.ctrl[:]=np.clip(-self.kp*qerr[6:]-self.kd*(verr[6:]+acc[6:]*dt),-self.limit,self.limit)

class Detector:
    """Original M6 full-sphere, downward aperture crossing test."""
    def __init__(self):self.center=np.array([4.572,0,3.04]);self.radius=.121;self.aperture=.237*np.cos(np.pi/64)-.010;self.previous=None;self.armed=False;self.entered=False;self.scored=False
    def update(self,p,released):
        p=np.asarray(p,float);old=self.previous;self.previous=p.copy()
        if not released:self.armed=False;self.entered=False;return False
        if self.scored:return False
        z=self.center[2];top=z+self.radius;bottom=z-self.radius
        if p[2]>=top:self.armed=True
        if old is None:return False
        if old[2]>=top:self.armed=True
        descending=p[2]<old[2]
        if self.armed and descending and old[2]>=top>p[2]:
            cross=old+(p-old)*((old[2]-top)/(old[2]-p[2]));self.entered=np.linalg.norm(cross[:2]-self.center[:2])<=self.aperture
        if self.entered:
            points=list(np.linspace(old,p,9))
            for plane in (top,z,bottom):
                if descending and old[2]>=plane>=p[2]:points.append(old+(p-old)*((old[2]-plane)/(old[2]-p[2])))
            invalid=False
            for point in points:
                dz=point[2]-z
                if abs(dz)<=self.radius:invalid|=np.linalg.norm(point[:2]-self.center[:2])+np.sqrt(max(0,self.radius**2-dz**2))>self.aperture+1e-10
            if not descending or invalid:self.entered=False;self.armed=False
            elif old[2]>=bottom>p[2]:self.scored=True;return True
        return False

class Simulation:
    def __init__(self,human,m,d,cfg):
        self.human,self.m,self.d,self.cfg=human,m,d,cfg;self.motion=Motion(cfg);self.pd=PD(human,cfg);self.wrist=m.body('R_Wrist').id;self.qa=int(m.joint('basketball_free').qposadr[0]);self.va=int(m.joint('basketball_free').dofadr[0]);self.eq=m.equality('hand_ball_weld').id;self.ball=m.geom('basketball_geom').id;self.floor=m.geom('floor').id;self.detector=Detector();self.release=None;self.floor_hit=False;self.scored=False;self.fallen=False;self.times=[];self.states=[];self.flight=[]
        names=[f'{s}_{b}_{a}' for s in ('R','L') for b in ('Hip','Knee') for a in 'xyz']+[f'{b}_{a}' for b in ('Torso','Spine','Chest') for a in 'xyz']+[f'R_{b}_{a}' for b in ('Shoulder','Elbow','Wrist') for a in 'xyz'];self.ids=np.array([int(m.joint(n).qposadr[0]) for n in names]);self.bounds=np.array([m.joint(n).range for n in names]);mujoco.mj_resetData(m,d);d.qpos[:160]=self.motion.q[0];mujoco.mj_forward(m,d);self.root0=d.qpos[:3].copy();self.initialize_ball()
    def masks(self):
        self.m.geom_contype[self.ball]=8;self.m.geom_conaffinity[self.ball]=16;b=int(self.m.geom_bodyid[self.ball]);self.m.body_contype[b]=8;self.m.body_conaffinity[b]=16
    def initialize_ball(self):
        m,d=self.m,self.d;d.eq_active[self.eq]=0;m.geom_contype[self.ball]=0;m.geom_conaffinity[self.ball]=0;b=int(m.geom_bodyid[self.ball]);m.body_contype[b]=0;m.body_conaffinity[b]=0;R=d.xmat[self.wrist].reshape(3,3);position=d.xpos[self.wrist]+R@np.array(self.cfg['offset_m']);q=np.empty(4);mujoco.mju_mat2Quat(q,R.ravel());jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,jr,position,self.wrist);d.qpos[self.qa:self.qa+7]=np.r_[position,q];d.qvel[self.va:self.va+6]=np.r_[jp@d.qvel,R.T@(jr@d.qvel)];self.masks();d.eq_active[self.eq]=1;mujoco.mj_forward(m,d)
    def step(self):
        m,d=self.m,self.d
        for _ in range(self.cfg['control_substeps']):
            goal=self.motion.target(d.time+m.opt.timestep);goal[self.ids]=np.clip(goal[self.ids],self.bounds[:,0]+1e-5,self.bounds[:,1]-1e-5);beforeq=d.qpos[self.qa:self.qa+7].copy();beforev=d.qvel[self.va:self.va+6].copy()
            if self.release is None and d.time+1e-10>=self.cfg['release_time_s']:
                v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,m.body('basketball').id,v,0);self.release=dict(time_s=float(d.time),position_m=beforeq[:3].tolist(),velocity_m_s=v[3:].tolist());d.eq_active[self.eq]=0;self.masks();mujoco.mj_forward(m,d)
            if self.release is not None:
                floor_contact=any({int(d.contact[i].geom1),int(d.contact[i].geom2)}=={self.ball,self.floor} for i in range(d.ncon));self.floor_hit|=floor_contact
                if not self.floor_hit:self.scored|=self.detector.update(d.qpos[self.qa:self.qa+3],True)
                self.flight.append(np.r_[d.time,d.qpos[self.qa:self.qa+3],int(floor_contact)])
            self.pd.control(d,goal,self.motion.spline(np.clip(d.time,0,self.motion.duration),1))
            assert np.array_equal(beforeq,d.qpos[self.qa:self.qa+7]) and np.array_equal(beforev,d.qvel[self.va:self.va+6]);assert not d.xfrc_applied.any() and not d.qfrc_applied.any()
            mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
        self.times.append(float(d.time));self.states.append(d.qpos.copy());tilt=np.degrees(np.arccos(np.clip(d.xmat[m.body('Pelvis').id].reshape(3,3)[2,1],-1,1)));self.fallen|=bool(d.qpos[2]<.65*self.root0[2] or tilt>40)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or any(w.number for w in d.warning):raise RuntimeError('Nonfinite state or MuJoCo warning')
        return d.time>=self.motion.duration-1e-9
