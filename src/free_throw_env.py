"""Online bounded residual free throw. Root/ball states are never policy actions."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
from pathlib import Path
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from scipy.interpolate import CubicSpline
from humanoid_model import ROOT
from contact_shot import contact_scene,attached_model,ball_reaction
from contact_reference import GRIP_OFFSET
from shoot_spl import HOOP,RADIUS,ScoreDetector
from track_spl import Controller,reference_at
from prepare_residual import ACTION_LIMITS,OPTIONS


def quintic(x):
    if x<=0:return np.array([0.,0.,0.])
    if x>=1:return np.array([1.,0.,0.])
    return np.array([10*x**3-15*x**4+6*x**5,30*x*x*(1-x)**2,60*x-180*x*x+120*x**3])


class FreeThrowEnv(gym.Env):
    metadata={'render_modes':[]}

    def __init__(self,bundle=ROOT/'runs/spl_residual_ready',record=False):
        self.bundle=Path(bundle);self.record=record
        self.ref=dict(np.load(self.bundle/'nominal_reference.npz'))
        self.config=json.loads((self.bundle/'nominal_reference.json').read_text(encoding='utf-8'))
        scene=self.bundle/'scene.xml'
        self.xml=scene.read_text(encoding='utf-8') if scene.exists() else contact_scene()
        self.m=mujoco.MjModel.from_xml_string(self.xml);self.d=mujoco.MjData(self.m)
        self.loaded=attached_model(self.xml)
        # The portable bundle carries the same scene. Remove its free ball for
        # the unloaded inverse-dynamics model, preserving exact joint ordering.
        import xml.etree.ElementTree as ET
        tree=ET.fromstring(self.xml);tree.remove(tree.find('equality'))
        world=tree.find('worldbody');world.remove(world.find("body[@name='basketball']"))
        contact=tree.find('contact')
        for pair in list(contact):
            if pair.tag=='pair' and 'ball_geom' in (pair.get('geom1'),pair.get('geom2')):contact.remove(pair)
        self.unloaded=mujoco.MjModel.from_xml_string(ET.tostring(tree,encoding='unicode'))
        self.nq=self.unloaded.nq;self.nv=self.unloaded.nv
        self.target=mujoco.MjData(self.unloaded);self.future=mujoco.MjData(self.unloaded)
        self.J=np.zeros((3,self.nv));self.J2=np.zeros_like(self.J);self.Jr=np.zeros_like(self.J)
        self.wrists=['right_wrist','left_wrist'];points={n:[] for n in self.wrists}
        for q in self.ref['qpos']:
            self.target.qpos[:]=q;mujoco.mj_forward(self.unloaded,self.target)
            for n in points:points[n].append(self.target.body(n).xpos.copy())
        self.curves={n:CubicSpline(self.ref['times'],p,axis=0) for n,p in points.items()}
        self.groups=[]
        for name in ['right_elbow_z','right_wrist_x','right_forearm_roll']:
            self.groups.append([self.unloaded.joint(name).id])
        self.groups.append([j for j in range(self.unloaded.njnt) if self.unloaded.joint(j).name.startswith('right_')
                            and self.unloaded.joint(j).name.endswith('_x') and any(f in self.unloaded.joint(j).name for f in ['index','middle','ring','pinky'])])
        self.ballid=self.m.geom('ball_geom').id;self.floor=self.m.geom('floor').id
        self.hands=set(np.flatnonzero(self.m.geom_contype==4))
        self.feet={self.m.geom(s+'_'+p+'_geom').id for s in ['left','right'] for p in ['ankle','foot']}
        self.action_space=spaces.Box(-1.,1.,(5,),dtype=np.float32)
        # Fixed physical scaling first, running observation normalization in PPO.
        self.obs_size=2*self.nv+3+3+3+3+3+8+5
        self.observation_space=spaces.Box(-np.inf,np.inf,(self.obs_size,),dtype=np.float32)
        self.settle=.7;self.duration=float(self.ref['times'][-1]);self.total=self.settle+self.duration+2.
        self.dt=float(self.m.opt.timestep);self.policy_stride=round(.05/self.dt);self.control_stride=round(.005/self.dt)
        self.detach0=float(self.config['detach_motion_time_s'])
        self.start=float(self.config['adaptation']['push_start_s']);self.end=self.start+float(self.config['adaptation']['push_duration_s'])
        self.reset()

    def reset(self,seed=None,options=None):
        super().reset(seed=seed);mujoco.mj_resetData(self.m,self.d)
        self.d.qpos[:self.nq]=self.ref['qpos'][0];mujoco.mj_forward(self.m,self.d)
        self.d.qpos[self.nq:self.nq+3]=self.d.body('right_wrist').xpos+self.d.body('right_wrist').xmat.reshape(3,3)@GRIP_OFFSET
        self.d.qpos[self.nq+3:self.nq+7]=self.d.body('right_wrist').xquat;mujoco.mj_forward(self.m,self.d)
        self.cm=self.loaded;self.cd=mujoco.MjData(self.cm)
        self.cd.qpos[:]=self.d.qpos[:self.nq];mujoco.mj_forward(self.cm,self.cd)
        self.ctrl=Controller(self.cm,self.cd,**self.config.get('controller',OPTIONS))
        self.x=np.zeros(4);self.xv=np.zeros(4);self.command=np.zeros(5);self.detach=self.detach0;self.latched=False
        self.off=False;self.seen_contact=False;self.release=None;self.candidate=None;self.step_count=0;self.done=False
        self.hand_contact=False;self.ankle0=[self.d.body(s+'_ankle').xpos.copy() for s in ['left','right']]
        self.slip=0.;self.handpen=0.;self.grasp_force=0.;self.external=0.;self.torque_fraction=0.;self.fell=False
        self.detector=ScoreDetector();self.apex=-np.inf;self.closest=float('inf');self.first_ground=False
        self.error_square=np.zeros(2);self.error_count=0;self.hold_errors=[];self.motion_error=np.zeros(2)
        self.fail_reason=None;self.ball_history=[];self.records=[];self.applied_history=[];self.contacts=[];self.reward_sum=0.
        return self._obs(),{}

    def _window(self,t):
        a=quintic((t-(self.start-.22))/.22);a[1]/=.22;a[2]/=.22**2
        b=quintic((t-self.end)/.30);b=np.array([1-b[0],-b[1]/.30,-b[2]/.30**2])
        return np.array([a[0]*b[0],a[1]*b[0]+a[0]*b[1],a[2]*b[0]+2*a[1]*b[1]+a[0]*b[2]])

    def _targets(self):
        t=self.d.time-self.settle;q,v,a=reference_at(self.ref,t)
        moving=0<=t<=self.duration
        if not moving:v[:]=0;a[:]=0
        omega=16.;xdd=omega**2*(self.command[:4]-self.x)-2*omega*self.xv
        self.xv+=.005*xdd;self.x+=.005*self.xv
        self.x=np.clip(self.x,-1,1)
        w,wd,wdd=self._window(t)
        for k,group in enumerate(self.groups):
            dq=ACTION_LIMITS[k]*w*self.x[k]
            dv=ACTION_LIMITS[k]*(wd*self.x[k]+w*self.xv[k])
            da=ACTION_LIMITS[k]*(wdd*self.x[k]+2*wd*self.xv[k]+w*xdd[k])
            for j in group:
                qa,va=self.unloaded.jnt_qposadr[j],self.unloaded.jnt_dofadr[j]
                candidate=q[qa]+dq;bounded=np.clip(candidate,*self.unloaded.jnt_range[j]);q[qa]=bounded
                if bounded==candidate:v[va]+=dv;a[va]+=da
                else:v[va]=0.;a[va]=0.
        tasks=[];tc=float(np.clip(t,0,self.duration))
        if w==0 or not np.any(self.x):
            for name in self.wrists:
                curve=self.curves[name];tasks.append({'body':name,'local':np.zeros(3),'position':curve(tc),
                    'velocity':curve(tc,1) if moving else np.zeros(3),'acceleration':curve(tc,2) if moving else np.zeros(3),'weight':80.})
        else:
            self.target.qpos[:]=q;mujoco.mj_forward(self.unloaded,self.target)
            self.future.qpos[:]=q;mujoco.mj_integratePos(self.unloaded,self.future.qpos,v,1e-5);mujoco.mj_forward(self.unloaded,self.future)
            for name in self.wrists:
                body=self.unloaded.body(name).id;position=self.target.body(name).xpos.copy()
                mujoco.mj_jac(self.unloaded,self.target,self.J,self.Jr,position,body)
                mujoco.mj_jac(self.unloaded,self.future,self.J2,self.Jr,self.future.body(name).xpos,body)
                tasks.append({'body':name,'local':np.zeros(3),'position':position,'velocity':self.J@v,
                              'acceleration':self.J@a+((self.J2-self.J)/1e-5)@v,'weight':80.})
        return q,v,a,tasks

    def _obs(self):
        q,v,_=reference_at(self.ref,self.d.time-self.settle)
        delta=np.zeros(self.nv);mujoco.mj_differentiatePos(self.unloaded,delta,1.,self.d.qpos[:self.nq],q)
        hand=self.d.body('right_wrist').xpos;ball=self.d.qpos[self.nq:self.nq+3]
        state=[self.d.time/self.total,float(self.off),float(self.hand_contact),float(self.latched),self.detach-self.detach0]
        result=np.r_[delta,(self.d.qvel[:self.nv]-v)/10,(ball-hand),self.d.qvel[self.nv:self.nv+3]/10,
                     (HOOP-ball)/5,self.d.body('pelvis').xmat.reshape(3,3)[:,2],
                     self.d.subtree_com[1]-self.d.body('pelvis').xpos,self.x,self.xv/16,state]
        return result.astype(np.float32)

    def step(self,action):
        if self.done:raise RuntimeError('Call reset before stepping a completed episode')
        action=np.asarray(action,dtype=float)
        if action.shape!=(5,) or not np.isfinite(action).all() or np.any(abs(action)>1.000001):raise ValueError('Invalid residual action')
        previous=self.command.copy();self.command=np.clip(action,-1,1)
        reward=0.
        for _ in range(self.policy_stride):
            t=self.d.time-self.settle
            if not self.latched and t>=self.start-.28:
                self.detach=self.detach0+ACTION_LIMITS[4]*self.command[4];self.latched=True
            if not self.off and t>=self.detach-1e-9:
                qbefore=self.d.qpos.copy();vbefore=self.d.qvel.copy();self.d.eq_active[:]=False;mujoco.mj_forward(self.m,self.d)
                assert np.array_equal(qbefore,self.d.qpos) and np.array_equal(vbefore,self.d.qvel)
                self.off=True;self.cm=self.unloaded;self.cd=mujoco.MjData(self.cm)
                self.ctrl.model=self.cm;self.ctrl.data=self.cd;self.ctrl.lastJ=None;self.ctrl.lasttime=None;self.ctrl.task_history.clear()
            if self.step_count%self.control_stride==0:
                self.cd.qpos[:]=self.d.qpos[:self.nq];self.cd.qvel[:]=self.d.qvel[:self.nv];self.cd.time=self.d.time;mujoco.mj_forward(self.cm,self.cd)
                q,v,a,tasks=self._targets()
                reaction=ball_reaction(self.m,self.d,self.ballid)[:self.nv] if self.off else None
                self.d.ctrl[:]=self.ctrl.command(q,v,a,cartesian=tasks,interaction=reaction)
            before=self.d.qpos[self.nq:self.nq+3].copy();mujoco.mj_step(self.m,self.d);mujoco.mj_forward(self.m,self.d);self.step_count+=1
            if not np.isfinite(self.d.qpos).all() or np.any(self.d.warning.number):
                self.fail_reason='numerical_instability';self.done=True;break
            p=self.d.qpos[self.nq:self.nq+3].copy();vel=self.d.qvel[self.nv:self.nv+3].copy()
            self.detector.update(before,p,float(self.d.time));self.hand_contact=False;nonfoot=False
            for c in self.d.contact:
                if self.ballid in (c.geom1,c.geom2):
                    other=c.geom2 if c.geom1==self.ballid else c.geom1
                    if other in self.hands:self.hand_contact=True;self.handpen=max(self.handpen,-float(c.dist))
                    if self.record and self.step_count%25==0:self.contacts.append({'time_s':float(self.d.time),'geom':self.m.geom(other).name,'penetration_m':max(0.,-float(c.dist))})
                    if other==self.floor and self.off:self.first_ground=True
                elif self.floor in (c.geom1,c.geom2):
                    other=c.geom2 if c.geom1==self.floor else c.geom1
                    if other not in self.feet:nonfoot=True
            if self.off and self.hand_contact:self.seen_contact=True;self.release=None;self.candidate=None
            elif self.off and self.seen_contact and self.release is None:
                if self.candidate is None:self.candidate=(float(self.d.time),p.copy(),vel.copy())
                if self.d.time-self.candidate[0]>=.02:self.release=self.candidate
            if self.release is not None and not self.first_ground:
                self.apex=max(self.apex,float(p[2]));self.closest=min(self.closest,float(np.linalg.norm(p-HOOP)))
            rootup=self.d.body('pelvis').xmat.reshape(3,3)[2,2]
            if self.d.qpos[2]<.55 or rootup<.5 or nonfoot:self.fell=True;self.fail_reason='fall'
            for k,s in enumerate(['left','right']):self.slip=max(self.slip,float(np.linalg.norm(self.d.body(s+'_ankle').xpos[:2]-self.ankle0[k][:2])))
            if not self.off:
                ef=np.where(self.d.efc_type==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY),self.d.efc_force,0.);gen=np.zeros(self.m.nv)
                mujoco.mj_mulJacTVec(self.m,self.d,gen,ef);self.grasp_force=max(self.grasp_force,float(np.linalg.norm(gen[self.nv:self.nv+3])))
            self.external=max(self.external,float(np.max(abs(self.d.qfrc_applied))),float(np.max(abs(self.d.xfrc_applied))))
            self.torque_fraction=max(self.torque_fraction,float(np.max(abs(self.d.actuator_force)/self.m.actuator_forcerange[:,1])))
            if self.step_count%8==1:
                base,_,_=reference_at(self.ref,self.d.time-self.settle);self.target.qpos[:]=base;mujoco.mj_forward(self.unloaded,self.target)
                errors=np.array([np.linalg.norm(self.d.body(n).xpos-self.target.body(n).xpos) for n in ['left_wrist','right_wrist']])
                self.motion_error=errors
                if self.settle<=self.d.time<=self.settle+self.duration:self.error_square+=errors**2;self.error_count+=1
                if self.d.time>self.total-.3:self.hold_errors.append(errors)
                if self.record:self.records.append((float(self.d.time),self.d.qpos.copy(),self.d.qvel.copy(),errors.copy(),not self.off))
            if self.record:
                self.ball_history.append((float(self.d.time),p.copy(),vel.copy(),self.hand_contact));self.applied_history.append(self.x.copy())
            window=self._window(self.d.time-self.settle)[0]
            reward-=self.dt*(8*np.sum(self.motion_error**2)+.05*window*np.sum(self.x**2)+30*self.slip**2)
            if self.slip>.04:self.fail_reason='foot_slip'
            if self.handpen>.006:self.fail_reason='contact_penetration'
            if self.grasp_force>250:self.fail_reason='grasp_force'
            if self.ctrl.failures>5:self.fail_reason='qp_failure'
            if self.fail_reason or self.d.time>=self.total-1e-8:self.done=True;break
        reward-=.002*self._window(self.d.time-self.settle)[0]*float(np.sum((self.command[:4]-previous[:4])**2))
        info={}
        if self.done:
            metrics=self.metrics();reward+=metrics['terminal_reward'];info['metrics']=metrics
        self.reward_sum+=reward
        obs=self._obs()
        if not np.isfinite(obs).all():obs=np.nan_to_num(obs,nan=0,posinf=0,neginf=0)
        return obs,float(reward),self.done,False,info

    def metrics(self):
        rms=np.sqrt(self.error_square/max(self.error_count,1));hold=np.mean(self.hold_errors,axis=0) if self.hold_errors else np.full(2,1.)
        valid=self.release is not None and self.seen_contact
        velocity_error=20.;trajectory=0.
        if valid:
            _,p,v=self.release;T=np.linspace(.45,1.3,100)
            desired=(HOOP-p)[None,:]/T[:,None];desired[:,2]+=.5*9.81*T
            velocity_error=float(np.min(np.linalg.norm(desired-v,axis=1)))
            trajectory=12*np.exp(-.5*(velocity_error/3)**2)+4*np.exp(-.5*(self.closest/2)**2)
        gates={'balance':not self.fell and self.slip<.015,'tracking':bool(np.max(rms)<.08),'followthrough':bool(np.max(hold)<.06),
               'contact':valid and self.handpen<.003,'grasp_force':self.grasp_force<150,
               'actuation':self.external==0 and self.torque_fraction<=1.000001,'solver':self.ctrl.failures==0 and not any(self.d.warning.number)}
        terminal=trajectory+20*float(self.detector.scored and all(gates.values()))-3*sum(not x for x in gates.values())
        if not valid:terminal-=10
        if self.fail_reason:terminal-=15
        return {'scored':bool(self.detector.scored),'valid_success':bool(self.detector.scored and all(gates.values())),
                'gates':gates,'accepted':all(gates.values()),'failure':self.fail_reason,'duration_s':float(self.d.time),
                'apex_m':float(self.apex) if np.isfinite(self.apex) else 0.,'closest_hoop_m':float(self.closest) if np.isfinite(self.closest) else 20.,
                'release_velocity_error_m_s':velocity_error,'release_velocity_m_s':self.release[2].tolist() if valid else None,
                'release_position_m':self.release[1].tolist() if valid else None,'release_time_s':self.release[0] if valid else None,
                'wrist_rms_m':rms.tolist(),'final_hold_wrist_m':hold.tolist(),'foot_slip_m':self.slip,'hand_penetration_m':self.handpen,
                'grasp_force_n':self.grasp_force,'external_force':self.external,'torque_fraction':self.torque_fraction,
                'qp_fallbacks':self.ctrl.failures,'detach_motion_time_s':self.detach,'trajectory_reward':float(trajectory),'terminal_reward':float(terminal)}
