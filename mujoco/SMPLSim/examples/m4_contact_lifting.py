"""M4-2D-2: slow contact-only lifting with unchanged M3 PD + COM controller."""
import argparse,json,hashlib,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from examples import m4_hand_kinematics as hand
from examples import m4_smpl_contact as contact
from examples.m3_control import write_csv
from smpl_sim.physics.basketball import HumanoidControllerBridge
from smpl_sim.motion.physics_metrics import foot_contact_slip

PREVIOUS=ROOT/'outputs/m4_2d_1_hand'
ARM=[f'R_{b}_{a}' for b in ['Shoulder','Elbow','Wrist'] for a in 'xyz']


def rotation_angles(R,reference):
    angles=Rotation.from_matrix(R).as_euler('XYZ')
    other=np.array([angles[0]+np.pi,np.pi-angles[1],angles[2]+np.pi]);other=(other+np.pi)%(2*np.pi)-np.pi
    return min((angles,other),key=lambda a:np.linalg.norm(a-reference))


def arm_path(m,q,height,out):
    d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d)
    wrist=m.body('R_Wrist').id;p0=d.xpos[wrist].copy();R0=d.xmat[wrist].reshape(3,3).copy()
    ids=[m.joint(n).id for n in ARM];adrs=np.array([int(m.jnt_qposadr[j]) for j in ids]);rows=[]
    for j,n in zip(ids,ARM):
        p=d.xpos[wrist].copy();R=d.xmat[wrist].reshape(3,3).copy()
        for delta in [-.2,.2]:
            d.qpos[:]=q;d.qpos[int(m.jnt_qposadr[j])]+=delta;mujoco.mj_forward(m,d)
            dp=d.xpos[wrist]-p;dr=Rotation.from_matrix(d.xmat[wrist].reshape(3,3)@R.T).as_rotvec()
            rows.append(dict(joint=n,delta_rad=delta,dx_m=dp[0],dy_m=dp[1],dz_m=dp[2],rot_x_rad=dr[0],rot_y_rad=dr[1],rot_z_rad=dr[2]))
        d.qpos[:]=q;mujoco.mj_forward(m,d)
    write_csv(out/'arm_fk_perturbations.csv',rows)
    jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv));mujoco.mj_jacBody(m,d,jp,jr,wrist)
    dofs=[int(m.jnt_dofadr[j]) for j in ids];J=np.vstack((jp[:,dofs],jr[:,dofs]));sv=np.linalg.svd(J,compute_uv=False)
    (out/'arm_fk.json').write_text(json.dumps({'rank':int(np.linalg.matrix_rank(J)),'singular_values':sv.tolist(),'initial_wrist_world_m':p0.tolist(),'initial_rotation':R0.tolist()},indent=2))
    base=q[adrs[:6]].copy();previous=base.copy();targets=[];ikrows=[]
    for u in np.linspace(0,1,81):
        desired=p0+[0,0,height*u]
        def residual(x):
            d.qpos[:]=q;d.qpos[adrs[:6]]=x;mujoco.mj_forward(m,d)
            return np.r_[30*(d.xpos[wrist]-desired),.025*(x-base),.04*(x-previous)]
        fit=least_squares(residual,previous,bounds=(base-1.6,base+1.6),xtol=1e-10,ftol=1e-10,gtol=1e-10,max_nfev=80)
        target=q.copy();target[adrs[:6]]=fit.x;d.qpos[:]=target;mujoco.mj_forward(m,d)
        target[adrs[6:]]=rotation_angles(d.xmat[m.body_parentid[wrist]].reshape(3,3).T@R0,q[adrs[6:]])
        d.qpos[:]=target;mujoco.mj_forward(m,d)
        error=np.linalg.norm(d.xpos[wrist]-desired)
        if error>.005:raise RuntimeError(f'IK path position error {error:g}m')
        targets.append(target);ikrows.append(dict(fraction=u,desired_z_m=desired[2],fk_z_m=d.xpos[wrist,2],position_error_mm=error*1000,orientation_error_deg=np.degrees(Rotation.from_matrix(R0.T@d.xmat[wrist].reshape(3,3)).magnitude()),**{n:target[a] for n,a in zip(ARM,adrs)}));previous=fit.x.copy()
    write_csv(out/'ik_path.csv',ikrows);np.savez_compressed(out/'ik_path.npz',qpos=targets,fraction=np.linspace(0,1,81),rotation=R0,height_m=height)
    return np.asarray(targets),R0,adrs


class LiftingTargets(hand.GraspTargets):
    def __init__(self,bridge,candidate,path,R,adrs,condition,start,ramp):
        super().__init__(bridge,candidate);self.path=path;self.R=R;self.adrs=adrs;self.condition=condition;self.start=start;self.ramp=ramp;self.frozen=None
    def target(self,t,actual=None):
        q=super().target(0,actual)
        if t<self.start:return q
        if self.frozen is None:self.frozen=q[self.adrs[6:]].copy()
        s=np.clip((t-self.start)/self.ramp,0,1);s=s**3*(10-15*s+6*s*s)
        at=s*(len(self.path)-1);i=min(int(at),len(self.path)-2);a=at-i
        goal=(1-a)*self.path[i]+a*self.path[i+1];q[self.adrs[:6]]=goal[self.adrs[:6]]
        if self.condition=='B':q[self.adrs[6:]]=self.frozen
        else:
            self.probe.qpos[:]=q;mujoco.mj_forward(self.m,self.probe)
            data=actual if actual is not None else self.probe;bid=self.m.body('R_Wrist').id
            parent=data.xmat[self.m.body_parentid[bid]].reshape(3,3)
            q[self.adrs[6:]]=rotation_angles(parent.T@self.R,data.qpos[self.adrs[6:]])
        return q


def initialize(m,d,bridge,adapter,c):
    bridge.initialize(m,d);warm=mujoco.MjData(bridge.model);warm.qpos[:]=adapter.target(0);mujoco.mj_forward(bridge.model,warm)
    for _ in range(round(1.5/m.opt.timestep)):
        adapter.control(warm,adapter.target(0,warm));mujoco.mj_step(bridge.model,warm);mujoco.mj_forward(bridge.model,warm)
    d.qpos[:bridge.model.nq]=warm.qpos;d.qvel[:bridge.model.nv]=0;mujoco.mj_forward(m,d)
    wrist=m.body('R_Wrist').id;W=d.xmat[wrist].reshape(3,3);origin=d.xpos[wrist]+W@np.array([c['center_x'],c.get('center_y',0),c['center_z']])
    q=int(m.joint('basketball_free').qposadr[0]);v=int(m.joint('basketball_free').dofadr[0]);bg=m.geom('basketball_geom').id;lo,hi=.121,.35
    for _ in range(40):
        h=(lo+hi)/2;d.qpos[q:q+7]=np.r_[origin+[0,0,h],1,0,0,0];mujoco.mj_forward(m,d)
        if any(bg in (x.geom1,x.geom2) and x.dist<0 for x in d.contact):lo=h
        else:hi=h
    d.qpos[q:q+7]=np.r_[origin+[0,0,hi+.0002],1,0,0,0];d.qvel[v:v+6]=0;mujoco.mj_forward(m,d)
    return q,v


def simulate(args,condition,candidate):
    out=args.output/condition;out.mkdir(parents=True,exist_ok=True)
    m=mujoco.MjModel.from_xml_path(str(args.output/'contact.xml'));d=mujoco.MjData(m)
    b=HumanoidControllerBridge(args.models/'body_01.xml',m,args.source)
    ik=np.load(args.output/'ik_path.npz');path=ik['qpos'];R=ik['rotation'];adrs=np.array([int(b.model.joint(n).qposadr[0]) for n in ARM])
    adapter=LiftingTargets(b,candidate,path,R,adrs,condition,args.start,args.ramp)
    q,v=initialize(m,d,b,adapter,candidate)
    static=json.loads((args.output/'trials/fk_fit_2/summary.json').read_text())
    if not np.allclose(d.qpos[q:q+3],static['initial_position_m'],atol=1e-10,rtol=0):raise RuntimeError('Initial ball placement differs from reproduced A')
    w=m.body('R_Wrist').id;W=d.xmat[w].reshape(3,3);rel0=W.T@(d.qpos[q:q+3]-d.xpos[w]);root0=d.qpos[2]
    initial={'ball_qpos':d.qpos[q:q+7].tolist(),'human_qpos':d.qpos[:b.model.nq].tolist()};(out/'initial_state.json').write_text(json.dumps(initial))
    rows=[];joint_rows=[];contact_rows=[];states=[];times=[];seen=False;loss_start=None;loss=None;viewer=None
    if args.gui:
        from mujoco import viewer as mv
        viewer=mv.launch_passive(m,d);viewer.cam.lookat[:]=[.2,0,1.25];viewer.cam.distance=1.8;viewer.cam.azimuth=110;viewer.cam.elevation=-15
    begun=time.monotonic();duration=args.start+args.ramp+args.end_hold
    for step in range(round(duration/m.opt.timestep)):
        target=adapter.target(d.time,d);p=d.qpos[q:q+7].copy();vel=d.qvel[v:v+6].copy();adapter.control(d,target)
        assert np.array_equal(p,d.qpos[q:q+7]) and np.array_equal(vel,d.qvel[v:v+6]);assert m.neq==0 and not d.xfrc_applied.any() and not d.qfrc_applied.any()
        mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():raise RuntimeError('Nonfinite state')
        t=float(d.time);n,nf,F,ff,depth,slip=contact.hand_contacts(m,d);W=d.xmat[w].reshape(3,3);rel=W.T@(d.qpos[q:q+3]-d.xpos[w])
        seen=seen or n>0
        if n>0:loss_start=None
        elif seen and loss_start is None:loss_start=t
        if loss_start is not None and t-loss_start>=.05 and loss is None:loss=loss_start
        feet,margin,_=contact.contacts(m,d);fs,_=foot_contact_slip(m,d);tilt=np.degrees(np.arccos(np.clip(d.xmat[m.body('Pelvis').id].reshape(3,3)[2,1],-1,1)))
        e=d.qpos[adrs]-target[adrs];orientation=np.degrees(Rotation.from_matrix(R.T@W).magnitude())
        rows.append(dict(time_s=t,wrist_x_m=d.xpos[w,0],wrist_y_m=d.xpos[w,1],wrist_z_m=d.xpos[w,2],ball_x_m=d.qpos[q],ball_y_m=d.qpos[q+1],ball_z_m=d.qpos[q+2],
                         ball_vx_ms=d.qvel[v],ball_vy_ms=d.qvel[v+1],ball_vz_ms=d.qvel[v+2],relative_x_m=rel[0],relative_y_m=rel[1],relative_z_m=rel[2],drift_m=np.linalg.norm(rel-rel0),
                         contacts=n,finger_contacts=nf,force_z_N=F[2],finger_force_z_N=ff[2],slip_ms=slip,penetration_mm=depth*1000,palm_normal_up=-W[2,1],orientation_error_deg=orientation,
                         arm_tracking_rmse_rad=np.sqrt(np.mean(e*e)),both_feet=int(min(feet.values())>.1),com_margin_m=margin,foot_slip_ms=np.sqrt((fs['L']['rms_speed']**2+fs['R']['rms_speed']**2)/2),fallen=int(d.qpos[2]<.65*root0 or tilt>35)))
        if step%15==0:
            states.append(d.qpos.copy());times.append(t);jr={'time_s':t}
            for name in ARM+list(candidate['angles']):
                j=m.joint(name);a=int(j.qposadr[0]);dv=int(j.dofadr[0]);jr[name+'_error_rad']=d.qpos[a]-target[a];jr[name+'_torque_Nm']=d.qfrc_actuator[dv]
            joint_rows.append(jr)
            bg=m.geom('basketball_geom').id
            for ci in range(d.ncon):
                con=d.contact[ci]
                if bg not in (con.geom1,con.geom2):continue
                g=int(con.geom2 if con.geom1==bg else con.geom1);name=m.geom(g).name
                if name!='R_Wrist' and not contact.is_finger(name):continue
                f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f);normal=con.frame[:3]*(-1 if con.geom1==bg else 1)
                contact_rows.append(dict(time_s=t,geom=name,normal_force_N=f[0],normal_x=normal[0],normal_y=normal[1],normal_z=normal[2]))
        if viewer:
            if not viewer.is_running():break
            viewer.sync();time.sleep(max(0,t-(time.monotonic()-begun)))
    if viewer:viewer.close()
    write_csv(out/'telemetry.csv',rows);write_csv(out/'joint_tracking_torques.csv',joint_rows)
    if contact_rows:write_csv(out/'contacts.csv',contact_rows)
    np.savez_compressed(out/'states.npz',qpos=states,time_s=times)
    prior=[r for r in rows if args.start-.1<=r['time_s']<args.start];final=[r for r in rows if r['time_s']>duration-.5];lifting=[r for r in rows if r['time_s']>=args.start]
    base_w=np.mean([r['wrist_z_m'] for r in prior]);base_b=np.mean([r['ball_z_m'] for r in prior]);rise_w=np.mean([r['wrist_z_m'] for r in final])-base_w;rise_b=np.mean([r['ball_z_m'] for r in final])-base_b
    gap=maxgap=0
    for r in rows:
        if r['time_s']<.2:continue
        gap=gap+1 if r['contacts']==0 else 0;maxgap=max(maxgap,gap)
    summary=dict(condition=condition,height_goal_m=args.height,lift_start_s=args.start,lift_duration_s=args.ramp,duration_s=float(d.time),wrist_rise_m=rise_w,ball_rise_m=rise_b,
                 max_relative_drift_m=max(r['drift_m'] for r in lifting),first_contact_loss_s=loss,contact_ratio=float(np.mean([r['contacts']>0 for r in lifting])),max_contact_gap_s=maxgap*m.opt.timestep,
                 mean_vertical_force_N=float(np.mean([r['force_z_N'] for r in lifting])),mean_finger_vertical_force_N=float(np.mean([r['finger_force_z_N'] for r in lifting])),
                 active_contact_slip_rms_ms=float(np.sqrt(np.mean([r['slip_ms']**2 for r in lifting if r['contacts']>0]))) if any(r['contacts']>0 for r in lifting) else None,
                 slip_rms_ms=float(np.sqrt(np.mean([r['slip_ms']**2 for r in lifting]))),max_orientation_error_deg=max(r['orientation_error_deg'] for r in lifting),min_palm_normal_up=min(r['palm_normal_up'] for r in lifting),
                 arm_tracking_rmse_rad=float(np.sqrt(np.mean([r['arm_tracking_rmse_rad']**2 for r in lifting]))),both_feet_ratio=float(np.mean([r['both_feet'] for r in rows])),
                 min_com_margin_m=min(r['com_margin_m'] for r in rows),foot_slip_rms_ms=float(np.sqrt(np.mean([r['foot_slip_ms']**2 for r in rows]))),fallen=bool(any(r['fallen'] for r in rows)),warnings=[int(x.number) for x in d.warning],
                 ball_state_writes_after_initialization=0,neq=m.neq)
    summary['success']=bool(seen and loss is None and abs(rise_w-args.height)<.04 and rise_b>=args.height*.8 and abs(rise_b-rise_w)<.04 and summary['max_relative_drift_m']<.05 and summary['both_feet_ratio']>=.99 and summary['min_com_margin_m']>=0 and not summary['fallen'] and not any(summary['warnings']) and d.time>=duration-1e-8)
    errors=[dict(joint=n,rmse_rad=float(np.sqrt(np.mean([r[n+'_error_rad']**2 for r in joint_rows if r['time_s']>=args.start]))),max_torque_Nm=max(abs(r[n+'_torque_Nm']) for r in joint_rows)) for n in ARM+list(candidate['angles'])]
    write_csv(out/'per_joint_rmse.csv',errors);(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary));return summary


def input_signature(args):
    paths=[args.models/'body_01.xml',args.source,PREVIOUS/'candidates.json',ROOT/'examples/m3_control.py',ROOT/'examples/m4_hand_kinematics.py']
    return {str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}


def gate(args):
    p=args.output/'trials/fk_fit_2/summary.json'
    if not p.exists():raise RuntimeError('Static A must be reproduced before lifting')
    s=json.loads(p.read_text())
    manifest=args.output/'static_inputs.json'
    if not manifest.exists() or json.loads(manifest.read_text())!=input_signature(args):raise RuntimeError('Static gate input provenance mismatch; reproduce A')
    if not s['held_3s'] or s['max_relative_drift_m']>.02 or s['fallen'] or s['both_feet_ratio']<.99 or s['min_com_margin_m']<0 or any(s['warnings']):raise RuntimeError('Static Holding gate failed')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',type=Path,default=ROOT.parents[2]/'basketball-r1/SMPLSim/outputs/smplh_physics');p.add_argument('--source',type=Path,default=ROOT/'outputs/m3_spl_fitting/spl_P0002_T0049.npz');p.add_argument('--output',type=Path,default=ROOT/'outputs/m4_2d_2_lifting')
    p.add_argument('--condition',choices=['all','A','B','C'],default='all');p.add_argument('--height',type=float,default=.25);p.add_argument('--start',type=float,default=1.);p.add_argument('--ramp',type=float,default=4.);p.add_argument('--end-hold',type=float,default=1.5);p.add_argument('--gui',action='store_true')
    args=p.parse_args()
    if not np.isfinite([args.height,args.start,args.ramp,args.end_hold]).all() or not .2<=args.height<=.3 or args.start<.3 or args.ramp<2 or args.end_hold<.5:p.error('Use height .2.. .3m, start >= .3s, slow ramp >= 2s, end-hold >= .5s')
    args.output.mkdir(parents=True,exist_ok=True);candidate=next(c for c in json.loads((PREVIOUS/'candidates.json').read_text()) if c['name']=='fk_fit_2')
    if args.condition in ('all','A'):
        paths=list(PREVIOUS.rglob('*'))+[ROOT/'examples/m4_hand_kinematics.py',ROOT/'examples/m4_hand_kinematics_report.py',ROOT/'tests/test_hand_kinematics.py',ROOT/'M4_2D_1_HAND_KINEMATICS.md']
        hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths if f.is_file()};(args.output/'d1_preservation_before.json').write_text(json.dumps(hashes,indent=2))
        # Derived scene generated into a new folder; previous files are only read.
        m,d,b,a,q=hand.setup(args);(args.output/'axis_mapping.json').write_text((PREVIOUS/'axis_mapping.json').read_text())
        hand.trial(args,candidate,3.5)
        (args.output/'static_inputs.json').write_text(json.dumps(input_signature(args),indent=2));gate(args)
        target=hand.GraspTargets(b,candidate).target(0);arm_path(b.model,target,args.height,args.output)
    if args.condition in ('all','B','C'):
        gate(args)
        plan=np.load(args.output/'ik_path.npz')
        if 'height_m' not in plan or not np.isclose(plan['height_m'],args.height):
            m,d,b,a,q=hand.setup(args);arm_path(b.model,hand.GraspTargets(b,candidate).target(0),args.height,args.output)
        for condition in ('B','C') if args.condition=='all' else (args.condition,):simulate(args,condition,candidate)

if __name__=='__main__':main()
