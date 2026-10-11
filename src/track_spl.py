"""Floating-base inverse-dynamics tracking with friction-constrained contact forces.

The QP only commands bounded joint motor torques. No root forces, welds, or state
teleportation are used during the rollout. Contact forces are predictions, not
forces injected into MuJoCo. Actual ground contacts are solved by MuJoCo.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
import osqp
import mujoco
from humanoid_model import ROOT,MODEL_PATH


def motor_model():
    tree=ET.fromstring(MODEL_PATH.read_text(encoding='utf-8'))
    for actuator in tree.find('actuator'):
        actuator.tag='motor'
        actuator.attrib.pop('kp',None);actuator.attrib.pop('kv',None)
        actuator.set('ctrlrange',actuator.get('forcerange'))
    for key in tree.find('keyframe'):key.set('ctrl',' '.join(['0']*len(tree.find('actuator'))))
    return mujoco.MjModel.from_xml_string(ET.tostring(tree,encoding='unicode'))


def foot_points(model,data):
    result=[]
    for side in ['left','right']:
        for part,xsign in [('ankle',-1),('foot',1)]:
            geom=model.geom(side+'_'+part+'_geom')
            for ysign in [-1,1]:
                local=geom.pos+np.array([xsign*geom.size[0],ysign*geom.size[1],-geom.size[2]])
                bid=int(geom.bodyid[0])
                world=data.xpos[bid]+data.xmat[bid].reshape(3,3)@local
                target=world.copy();target[2]=0.
                result.append((bid,local,target))
    return result


class Controller:
    def __init__(self,model,data,arm_gain=90.,arm_weight=1.,foot_weight=30.,acceleration_limit=250.):
        self.model=model;self.data=data
        self.arm_gain=arm_gain;self.arm_weight=arm_weight;self.foot_weight=foot_weight
        self.acceleration_limit=acceleration_limit
        self.arm_dofs=np.array([model.jnt_dofadr[j] for j in range(model.njnt)
                               if any(part in model.joint(j).name for part in ['shoulder_','elbow_','wrist_','forearm_'])])
        self.contacts=foot_points(model,data)
        self.nv=model.nv;self.nf=3*len(self.contacts);self.n=self.nv+self.nf
        self.dofs=model.jnt_dofadr[model.actuator_trnid[:,0]]
        self.maxforce=model.actuator_forcerange[:,1].copy()
        self.M=np.zeros((self.nv,self.nv))
        self.jac=np.zeros((3,self.nv));self.rotjac=np.zeros((3,self.nv))
        self.lastJ=None;self.lasttime=None
        self.failures=0;self.statuses={};self.lastsolution=np.zeros(self.n)
        self.solver=None
        self.task_last_jac=None;self.task_last_time=None
        self.task_history={}
        self.a_rows=self.nv+self.nf+4*len(self.contacts)
        # Fixed dense CSC sparsity lets OSQP reuse symbolic factorization.
        self.Ppattern=sparse.csc_matrix(np.triu(np.ones((self.n,self.n))))
        self.Apattern=sparse.csc_matrix(np.ones((self.a_rows,self.n)))
        self.Pindices=(self.Ppattern.indices,np.repeat(np.arange(self.n),np.diff(self.Ppattern.indptr)))
        self.Aindices=(self.Apattern.indices,np.repeat(np.arange(self.n),np.diff(self.Apattern.indptr)))

    def command(self,qref,vref,aref,cartesian=None,interaction=None):
        m,d=self.model,self.data
        mujoco.mj_fullM(m,d,self.M)
        delta=np.zeros(m.nv);mujoco.mj_differentiatePos(m,delta,1.,d.qpos,qref)
        kp=np.full(m.nv,90.);kd=np.full(m.nv,19.)
        kp[:6]=100.;kd[:6]=20.
        if self.arm_gain!=90.:
            kp[self.arm_dofs]=self.arm_gain;kd[self.arm_dofs]=2*np.sqrt(self.arm_gain)
        desired=np.clip(aref+kp*delta+kd*(vref-d.qvel),-self.acceleration_limit,self.acceleration_limit)
        J=[];pos=[];targets=[]
        for bid,local,target in self.contacts:
            point=d.xpos[bid]+d.xmat[bid].reshape(3,3)@local
            mujoco.mj_jac(m,d,self.jac,self.rotjac,point,bid)
            J.append(self.jac.copy());pos.append(point);targets.append(target)
        J=np.concatenate(J);pos=np.concatenate(pos);targets=np.concatenate(targets)
        elapsed=d.time-self.lasttime if self.lasttime is not None else 0
        jdotv=((J-self.lastJ)/elapsed)@d.qvel if elapsed>1e-7 else np.zeros(self.nf)
        jdotv=np.clip(jdotv,-100,100)
        self.lastJ=J.copy();self.lasttime=d.time
        contact_acc=np.clip(600*(targets-pos)-45*(J@d.qvel)-jdotv,-150,150)
        # Joint acceleration objective plus high-priority foot stabilization.
        weights=np.ones(m.nv);weights[:6]=3.
        weights[self.arm_dofs]=self.arm_weight
        P=np.eye(self.n)*1e-5;P[:m.nv,:m.nv]=np.diag(weights)
        cost=np.zeros(self.n);cost[:m.nv]=-weights*desired
        P[:m.nv,:m.nv]+=self.foot_weight*(J.T@J)
        cost[:m.nv]-=self.foot_weight*J.T@contact_acc
        # Encourage the horizontal COM to remain over the foot support polygon.
        mujoco.mj_jacSubtreeCom(m,d,self.jac,1)
        support=np.mean(np.array([c[2] for c in self.contacts]),axis=0)
        com_acc=80*(support[:2]-d.subtree_com[1,:2])-18*(self.jac[:2]@d.qvel)
        C=self.jac[:2]
        P[:m.nv,:m.nv]+=20*C.T@C
        cost[:m.nv]-=20*C.T@com_acc
        tasks=[] if cartesian is None else (cartesian if isinstance(cartesian,list) else [cartesian])
        for cartesian in tasks:
            if cartesian['weight']<=0:continue
            bid=m.body(cartesian['body']).id
            point=d.xpos[bid]+d.xmat[bid].reshape(3,3)@cartesian['local']
            mujoco.mj_jac(m,d,self.jac,self.rotjac,point,bid)
            task_jac=self.jac.copy()
            key=(cartesian['body'],tuple(cartesian['local']))
            previous,previous_time=self.task_history.get(key,(None,None))
            task_dt=d.time-previous_time if previous_time is not None else 0.
            drift=((task_jac-previous)/task_dt)@d.qvel if task_dt>1e-7 else np.zeros(3)
            self.task_history[key]=(task_jac.copy(),d.time)
            self.task_last_jac=task_jac;self.task_last_time=d.time
            task_acc=np.clip(cartesian['acceleration']+400*(cartesian['position']-point)
                             +40*(cartesian['velocity']-task_jac@d.qvel)-drift,-150,150)
            weight=cartesian['weight']
            P[:m.nv,:m.nv]+=weight*task_jac.T@task_jac
            cost[:m.nv]-=weight*task_jac.T@task_acc
        dynamics=np.c_[self.M,-J.T]
        A=np.zeros((self.a_rows,self.n))
        lower=np.full(self.a_rows,-np.inf);upper=np.full(self.a_rows,np.inf)
        A[:m.nv]=dynamics
        # Passive joint damping is a real internal force in MuJoCo.
        bias=d.qfrc_bias-d.qfrc_passive
        if interaction is not None:
            interaction=np.asarray(interaction)
            if interaction.shape!=(m.nv,) or not np.isfinite(interaction).all():
                raise ValueError('Interaction must be a finite generalized force vector')
            # Known contact reaction on the humanoid. This changes inverse
            # dynamics predictions; it does not inject forces into simulation.
            bias=bias-interaction
        lower[:6]=upper[:6]=-bias[:6]
        force_by_dof=np.zeros(m.nv);force_by_dof[self.dofs]=self.maxforce
        lower[6:m.nv]=-force_by_dof[6:]-bias[6:]
        upper[6:m.nv]=force_by_dof[6:]-bias[6:]
        A[m.nv:m.nv+self.nf,m.nv:]=np.eye(self.nf)
        for i in range(len(self.contacts)):
            zrow=m.nv+3*i+2
            lower[zrow]=0;upper[zrow]=700
            for k,(sx,sy) in enumerate([(1,1),(1,-1),(-1,1),(-1,-1)]):
                row=m.nv+self.nf+4*i+k
                A[row,m.nv+3*i:m.nv+3*i+3]=[sx,sy,-.7]
                upper[row]=0
        Pdata=P[self.Pindices];Adata=A[self.Aindices]
        if self.solver is None:
            self.Ppattern.data[:]=Pdata;self.Apattern.data[:]=Adata
            self.solver=osqp.OSQP()
            self.solver.setup(P=self.Ppattern,q=cost,A=self.Apattern,l=lower,u=upper,
                              verbose=False,eps_abs=1e-3,eps_rel=1e-3,max_iter=300,
                              polishing=False,warm_starting=True)
        else:self.solver.update(Px=Pdata,Ax=Adata,q=cost,l=lower,u=upper)
        result=self.solver.solve(raise_error=False)
        status=result.info.status
        self.statuses[status]=self.statuses.get(status,0)+1
        if result.x is not None and result.info.status_val in (1,2):
            solution=result.x;self.lastsolution=solution
            torque=(dynamics@solution+bias)[self.dofs]
        else:
            self.failures+=1
            # Bounded joint PD fallback, never root actuation.
            torque=(40*delta-5*d.qvel)[self.dofs]
        return np.clip(torque,-self.maxforce,self.maxforce)


def reference_at(ref,time):
    times=ref['times'];time=np.clip(time,times[0],times[-1]);i=min(np.searchsorted(times,time,side='right')-1,len(times)-2)
    alpha=(time-times[i])/(times[i+1]-times[i])
    q=(1-alpha)*ref['qpos'][i]+alpha*ref['qpos'][i+1]
    q[3:7]/=np.linalg.norm(q[3:7])
    v=(1-alpha)*ref['qvel'][i]+alpha*ref['qvel'][i+1]
    a=(1-alpha)*ref['qacc'][i]+alpha*ref['qacc'][i+1]
    return q,v,a


def track(reference_path,output,settle=.7,hold=.5):
    reference_path=Path(reference_path)
    info=json.loads(reference_path.with_suffix('.json').read_text(encoding='utf-8'))
    if info['model_sha256']!=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest():
        raise ValueError('Model changed after retargeting; rebuild the reference first')
    ref=np.load(reference_path)
    m=motor_model();d=mujoco.MjData(m);target=mujoco.MjData(m)
    d.qpos[:]=ref['qpos'][0];d.qvel[:]=0;mujoco.mj_forward(m,d)
    controller=Controller(m,d)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    total=settle+float(ref['times'][-1])+hold
    duration=float(ref['times'][-1]);data=[];refs=[];times=[];metrics=[];states=[]
    tracked=['left_wrist','right_wrist','left_elbow','right_elbow','pelvis']
    fall_time=None;max_root_force=0.;peak_torque=0.;max_penetration=0.
    max_foot_slip=0.;min_root_up=1.;max_torque_fraction=0.;nonfoot_floor_contacts=set()
    floor_id=m.geom('floor').id
    foot_ids={m.geom(side+'_'+part+'_geom').id for side in ['left','right'] for part in ['ankle','foot']}
    ankle_initial={side:d.body(side+'_ankle').xpos.copy() for side in ['left','right']}

    for step in range(int(np.ceil(total/m.opt.timestep))):
        if step%5==0:
            t=d.time-settle
            q,v,a=reference_at(ref,t)
            if t<0 or t>duration:v=np.zeros(m.nv);a=np.zeros(m.nv)
            d.ctrl[:]=controller.command(q,v,a)
        mujoco.mj_step(m,d)
        if not np.isfinite(d.qpos).all() or np.any(d.warning.number):raise RuntimeError('Unstable simulation')
        max_root_force=max(max_root_force,float(np.max(abs(d.qfrc_applied[:6]))),float(np.max(abs(d.xfrc_applied))))
        peak_torque=max(peak_torque,float(np.max(abs(d.actuator_force))))
        for c in d.contact:max_penetration=max(max_penetration,-float(c.dist))
        max_torque_fraction=max(max_torque_fraction,float(np.max(abs(d.actuator_force)/m.actuator_forcerange[:,1])))
        for c in d.contact:
            if c.geom1==floor_id and c.geom2 not in foot_ids:nonfoot_floor_contacts.add(m.geom(c.geom2).name)
            if c.geom2==floor_id and c.geom1 not in foot_ids:nonfoot_floor_contacts.add(m.geom(c.geom1).name)
        root_up=float(d.body('pelvis').xmat.reshape(3,3)[2,2]);min_root_up=min(min_root_up,root_up)
        for side in ankle_initial:
            max_foot_slip=max(max_foot_slip,float(np.linalg.norm(d.body(side+'_ankle').xpos[:2]-ankle_initial[side][:2])))
        if (d.qpos[2]<.55 or root_up<.5 or nonfoot_floor_contacts) and fall_time is None:fall_time=float(d.time)
        if step%17==0:
            mujoco.mj_forward(m,d)
            q,v,a=reference_at(ref,d.time-settle)
            target.qpos[:]=q;mujoco.mj_forward(m,target)
            data.append(d.qpos.copy());refs.append(q.copy());times.append(float(d.time))
            metrics.append([float(np.linalg.norm(d.body(name).xpos-target.body(name).xpos)) for name in tracked])
            states.append(d.subtree_com[1].copy())
    np.savez_compressed(output/'tracking.npz',qpos=np.array(data),reference_qpos=np.array(refs),times=times,
                        errors=np.array(metrics),error_names=np.array(tracked),com=np.array(states),settle=settle)
    motion_mask=(np.array(times)>=settle)&(np.array(times)<=settle+duration)
    errors=np.array(metrics)[motion_mask]
    report={'status':'completed','fell':fall_time is not None,'fall_time_s':fall_time,
            'duration_s':float(d.time),'motion_duration_s':duration,'settle_s':settle,'hold_s':hold,
            'rms_error_m':{name:float(np.sqrt(np.mean(errors[:,i]**2))) for i,name in enumerate(tracked)},
            'max_error_m':{name:float(np.max(errors[:,i])) for i,name in enumerate(tracked)},
            'minimum_pelvis_height_m':float(np.min(np.array(data)[:,2])),
            'maximum_contact_penetration_m':max_penetration,'peak_joint_torque_nm':peak_torque,
            'maximum_ankle_horizontal_displacement_m':max_foot_slip,
            'minimum_root_up_dot':min_root_up,'nonfoot_floor_contacts':sorted(nonfoot_floor_contacts),
            'maximum_torque_limit_fraction':max_torque_fraction,
            'reference_sha256':hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            'source_selection':info['selection'],
            'qp_status_counts':controller.statuses,'qp_fallback_count':controller.failures,
            'max_external_root_or_body_force':max_root_force,'mujoco_warning_counts':d.warning.number.tolist(),
            'controller':'Floating-base constrained inverse dynamics QP; bounded joint motor torque only.',
            'not_implemented':['Ball contact and shooting','Reinforcement learning','Measured finger pose tracking'],
            'reference':str(reference_path)}
    (output/'tracking_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,default=ROOT/'data/motions/processed/free_throw_reference.npz')
    p.add_argument('--output',type=Path,default=ROOT/'runs/spl_tracking')
    a=p.parse_args();track(a.reference,a.output)

