"""Independent physical grip hypotheses. No training, ball overwrite, or source edits."""
import sys,json,argparse,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from examples.m5_coordinated_reference import CoordinatedEnv,CoordinatedTargets,unit,scene_rim,BASE,MODELS,SOURCE,lift
from examples.m5_coordinated_eval import readiness
from examples.m5_bimanual_contact import side_contacts,local_ball,hand_geom
from examples.m5_upper_body_ppo_eval import save
from examples.m5_upper_body_ik import minimum_jerk
from examples.m3_control import write_csv
OUT=ROOT/'outputs/m5_1_shooting_grip_hypotheses'


def contact_wrenches(m,d):
    """World wrench ON ball, torque about its center (not contact origin)."""
    ball=m.geom('basketball_geom').id;center=d.geom_xpos[ball];out={s:dict(force=np.zeros(3),torque=np.zeros(3),count=0,normal=0.) for s in ('R','L','other')};points=[]
    for i in range(d.ncon):
        c=d.contact[i]
        if ball not in (c.geom1,c.geom2):continue
        other=c.geom2 if c.geom1==ball else c.geom1;name=m.geom(other).name or '';side=next((s for s in ('R','L') if hand_geom(name,s)),'other')
        w=np.zeros(6);mujoco.mj_contactForce(m,d,i,w)
        if w[0]<=.01:continue
        sign=-1 if c.geom1==ball else 1;F=c.frame.reshape(3,3).T@w[:3]*sign;T=c.frame.reshape(3,3).T@w[3:]*sign;normal=c.frame[:3]*sign
        torque=np.cross(c.pos-center,F)+T;z=out[side];z['force']+=F;z['torque']+=torque;z['count']+=1;z['normal']+=w[0]
        row=dict(time_s=d.time,side=side,geometry=name,normal_force_N=float(w[0]),gap_m=float(c.dist))
        for k,v in [('p',c.pos),('normal',normal),('force',F),('torque',torque)]:
            for a,x in zip('xyz',v):row[k+'_'+a]=float(x)
        points.append(row)
    return out,points


def geometry(e):
    m,d=e.m,e.d;center=d.geom_xpos[m.geom('basketball_geom').id];out={}
    for side in ('R','L'):
        wrist=m.body(side+'_Wrist').id;W=d.xmat[wrist].reshape(3,3);p=d.xpos[wrist];normal=-W[:,1];finger=unit(d.xpos[m.body(side+'_Middle1').id]-p)
        index=d.xpos[m.body(side+'_Index1').id]-p;pinky=d.xpos[m.body(side+'_Pinky1').id]-p;plane=unit(np.cross(index,pinky));plane*=1 if plane@normal>=0 else -1
        out[side]=dict(metacarpal_plane_normal=plane.tolist(),plane_normal_proxy_angle_deg=float(np.degrees(np.arccos(np.clip(plane@normal,-1,1)))),palm_normal=normal.tolist(),finger_direction=finger.tolist(),palm_ball_dot=float(normal@unit(center-p)),palm_up_dot=float(normal[2]),finger_forward_dot=float(finger@e.ref['forward']),wrist_position=p.tolist(),local_ball_position=(W.T@(center-p)).tolist(),wrist_rotation=W.tolist(),forearm_rotation=d.xmat[m.body(side+'_Elbow').id].reshape(3,3).tolist(),forearm_axis=unit(p-d.xpos[m.body(side+'_Elbow').id]).tolist())
    return out


class GripTargets(CoordinatedTargets):
    def target(self,t,actual=None):
        if t<=0 or actual is None:return super().target(t,actual)
        r=self.reference;original=r['right_rotation'].copy();original_left=r['left_rotation'].copy()
        s=minimum_jerk(t/1.5);normal=-original[:,1]
        # Separate yaw about the measured palm normal and pitch toward the actual rim.
        r['right_rotation']=Rotation.from_rotvec(r['side']*np.radians(self.cfg['pitch'])*s).as_matrix()@Rotation.from_rotvec(normal*np.radians(self.cfg['yaw'])*s).as_matrix()@original
        if self.cfg.get('guide_control',False):
            # Palm normal faces ball from its side; fingers point upward rather than clamp.
            inward=-r['side'];finger=np.array([0.,0.,1.]);finger=unit(finger-inward*(finger@inward));x=unit(np.cross(-inward,finger))
            if not hasattr(self,'left_local_basis'):
                probe=mujoco.MjData(self.m);probe.qpos[:]=r['initial'][:self.m.nq];mujoco.mj_kinematics(self.m,probe);bid=self.m.body('L_Wrist').id;P=probe.xpos[bid];W=probe.xmat[bid].reshape(3,3);f=W.T@unit(probe.xpos[self.m.body('L_Middle1').id]-P);f=unit(f-np.array([0.,1.,0.])*f[1]);self.left_local_basis=np.column_stack((unit(np.cross([0.,1.,0.],f)),[0.,1.,0.],f))
            desired=np.column_stack((x,-inward,finger))@self.left_local_basis.T
            r['left_rotation']=original_left@Rotation.from_rotvec(s*Rotation.from_matrix(original_left.T@desired).as_rotvec()).as_matrix()
            measured=side_contacts(actual.model,actual,'L');dt=actual.model.opt.timestep
            # Limited admittance: approach under low load, retreat under lateral overload.
            force=measured['normal_force'];self.guide_gap=np.clip(getattr(self,'guide_gap',.035)+np.clip((force-self.cfg.get('guide_force',.6))*(.04 if self.cfg.get('gentle',False) else .004),-.003,(.08 if self.cfg.get('gentle',False) else .02))*dt,-.002,.08)
            self.clearance=self.guide_gap
        q=super().target(t,actual)
        if self.cfg.get('guide_control',False):self.clearance=self.guide_gap
        self.desired_right_rotation=r['right_rotation'].copy();self.desired_left_rotation=r['left_rotation'].copy();r['right_rotation']=original;r['left_rotation']=original_left
        return q


class GripEnv(CoordinatedEnv):
    def __init__(self,yaw=0,pitch=0,guide=False,guide_force=.6,probe=False,gui=False,gentle=False):
        if not np.isfinite(yaw) or abs(yaw)>45 or not np.isfinite(pitch) or abs(pitch)>15 or not np.isfinite(guide_force) or not .1<=guide_force<=2:raise ValueError('Grip candidate bounds exceeded')
        self.grip_cfg=dict(yaw=float(yaw),pitch=float(pitch),guide_control=guide,guide_force=float(guide_force),gentle=gentle)
        self.wrench_rows=[];self.point_rows=[];self.prev_velocity=None;self.prev_time=None
        super().__init__(guide='fixed' if guide else 'off',tilt_deg=0,probe=probe,ball_feedback=True,prep_seconds=3.5,render_mode='human' if gui else None)
        self.config.update(self.grip_cfg);self.reset(seed=7)
    def _adapter(self):
        if not self.coord_ready:return super()._adapter()
        self.bridge=lift.HumanoidControllerBridge(MODELS/'body_01.xml',self.m,SOURCE);self._upper_cache_time=None
        c=dict(self.config,**self.grip_cfg);self.adapter=GripTargets(self.bridge,self.candidate,self.arm_ids,self.ref,4,self.condition,config=c)
    def metrics(self):
        z=super().metrics()
        if not self.coord_ready:return z
        w,points=contact_wrenches(self.m,self.d);F=sum((v['force'] for v in w.values()),np.zeros(3));T=sum((v['torque'] for v in w.values()),np.zeros(3));mass=float(self.m.body_mass[self.m.body('basketball').id]);net=F+mass*self.m.opt.gravity
        row=dict(time_s=self.d.time,phase=self.adapter.phase,drift_mm=z['drift_m']*1000,ball_height_m=float(self.d.qpos[self.ball_q+2]),right_contacts=w['R']['count'],left_contacts=w['L']['count'],left_normal_N=w['L']['normal'],right_slip_m_s=z['slip_m_s'] if 'slip_m_s' in z else side_contacts(self.m,self.d,'R')['slip'],com_margin_mm=z['com_margin_m']*1000,both_feet=bool(z['both_feet']),fallen=bool(z['fallen']),warning_count=sum(z['warnings']))
        for s,v in w.items():
            for kind in ('force','torque'):
                for a,x in zip('xyz',v[kind]):row[s+'_'+kind+'_'+a]=float(x)
        for a,x in zip('xyz',net):row['net_force_'+a]=float(x)
        for a,x in zip('xyz',T):row['net_torque_'+a]=float(x)
        for a,x in zip('xyz',self.d.qvel[self.ball_v:self.ball_v+3]):row['ball_velocity_'+a]=float(x)
        rel,rv=local_ball(self.m,self.d,'R');row['relative_speed_m_s']=float(np.linalg.norm(rv))
        for a,x in zip('xyz',rel):row['ball_local_'+a]=float(x)
        if self.prev_time is not None and self.d.time>self.prev_time:
            row['force_acceleration_residual_N']=float(np.linalg.norm(net-mass*(self.d.qvel[self.ball_v:self.ball_v+3]-self.prev_velocity)/(self.d.time-self.prev_time)))
        else:row['force_acceleration_residual_N']=float('nan')
        if not self.wrench_rows or self.d.time>self.wrench_rows[-1]['time_s']+1e-9:
            self.wrench_rows.append(row);self.point_rows.extend(points);self.prev_time=self.d.time;self.prev_velocity=self.d.qvel[self.ball_v:self.ball_v+3].copy()
        return z


def maximum_gap(rows):
    gap=longest=0.;last=None
    for r in rows:
        dt=0 if last is None else r['time_s']-last;last=r['time_s']
        gap=gap+dt if r['right_contacts']==0 else 0.;longest=max(longest,gap)
    return longest


def run(name,yaw=0,pitch=0,guide=False,probe=False,static_only=False,gui=False,speed=1.,guide_force=.6,gentle=False):
    folder=OUT/name
    if folder.exists():raise FileExistsError(folder)
    folder.mkdir(parents=True);e=GripEnv(yaw,pitch,guide,guide_force,probe,gui,gentle);e.reset(seed=7);e.wrench_rows=[];e.point_rows=[];e.prev_time=None
    initial=geometry(e);rows=[];terms=[];gates={};started=time.perf_counter();height=e.d.qpos[e.ball_q+2];gate_times=[(3.5,'static'),(7.5,'lifting'),(12.,'setpoint')];reason='unknown'
    try:
        while True:
            wall=time.perf_counter();_,_,terminated,truncated,info=e.step(np.zeros(e.action_space.shape,np.float32));z=readiness(e);rows.append(z);terms.append(dict(time_s=e.d.time,**info['reward_terms']))
            for t,stage in gate_times:
                if e.d.time>=t-1e-8 and stage not in gates:
                    recent=[r for r in e.wrench_rows if r['time_s']>=t-.5];ok=bool(recent and max(r['drift_mm'] for r in recent)<20 and all(r['both_feet'] and not r['fallen'] and r['warning_count']==0 for r in recent) and np.mean([r['right_contacts']>0 for r in recent])>.5 and maximum_gap(recent)<.05)
                    if stage=='lifting':ok=ok and e.d.qpos[e.ball_q+2]-height>.2
                    if stage=='setpoint':ok=ok and z['goal_distance_mm']<30 and z['elbow_extension_reserve_deg']>5
                    gates[stage]=bool(ok)
                    if not ok:reason=stage+'_gate_failed';terminated=True
            if static_only and 'static' in gates:reason='static_gate_completed';break
            if gui and speed>0:time.sleep(max(0,e.control_dt/speed-(time.perf_counter()-wall)))
            if terminated or truncated:reason=reason if reason!='unknown' else info.get('termination_reason');break
            if gui and e.viewer and not e.viewer.is_running():reason='viewer_closed';break
        final=geometry(e);wr=e.wrench_rows;summary=dict(name=name,yaw_deg=yaw,pitch_deg=pitch,guide=guide,guide_force_target_N=guide_force,gentle=gentle,duration_s=e.d.time,reason=reason,gates=gates,wall_seconds=time.perf_counter()-started,max_drift_mm=max(r['drift_mm'] for r in wr),ball_rise_mm=(max(r['ball_height_m'] for r in wr)-height)*1000,left_contact_ratio=float(np.mean([r['left_contacts']>0 for r in wr])),left_normal_peak_N=max(r['left_normal_N'] for r in wr),right_vertical_mean_N=float(np.mean([r['R_force_z'] for r in wr])),left_vertical_mean_N=float(np.mean([r['L_force_z'] for r in wr])),net_contact_torque_rms_Nm=float(np.sqrt(np.mean([sum(r['net_torque_'+a]**2 for a in 'xyz') for r in wr]))),slip_rms_m_s=float(np.sqrt(np.mean([r['right_slip_m_s']**2 for r in wr]))),min_com_margin_mm=min(r['com_margin_mm'] for r in wr),both_feet_ratio=float(np.mean([r['both_feet'] for r in wr])),fallen=any(r['fallen'] for r in wr),warnings=[int(w.number) for w in e.d.warning],probe_executed=e.adapter.pulse_dirs is not None,neq=e.m.neq,ball_writes_after_reset=0,external_force_max=float(max(np.max(np.abs(e.d.xfrc_applied)),np.max(np.abs(e.d.qfrc_applied)))),initial_geometry=initial,final_geometry=final)
        for s in ('R','L'):
            summary[s+'_orientation_rms_deg']=float(np.sqrt(np.mean([r[s+'_orientation_error_deg']**2 for r in e.rows])))
            summary[s+'_position_tracking_rms_mm']=float(np.sqrt(np.mean([r[s+'_target_actual_mm']**2 for r in e.rows])))
            summary[s+'_wrist_translation_mm']=float(np.linalg.norm(np.array(final[s]['wrist_position'])-initial[s]['wrist_position'])*1000)
            summary[s+'_forearm_rotation_deg']=float(np.degrees(Rotation.from_matrix(np.array(initial[s]['forearm_rotation']).T@np.array(final[s]['forearm_rotation'])).magnitude()))
        summary=json.loads(json.dumps(summary,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()))
        save(e,folder,summary,terms);write_csv(folder/'readiness.csv',rows);write_csv(folder/'wrenches.csv',wr);write_csv(folder/'contact_points.csv',e.point_rows);(folder/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if not k.endswith('geometry')}),flush=True);return summary
    finally:e.close()


def audit():
    before=json.loads((OUT/'preservation_before.json').read_text());changed=[p for p,h in before.items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];z=dict(checked_files=len(before),changed_scientific_files=[p for p in changed if Path(p).name!='.DS_Store'],changed_os_metadata=[p for p in changed if Path(p).name=='.DS_Store']);(OUT/'preservation_audit.json').write_text(json.dumps(z,indent=2));assert not z['changed_scientific_files'];return z


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='manual');p.add_argument('--yaw',type=float,default=0);p.add_argument('--pitch',type=float,default=0);p.add_argument('--guide',action='store_true');p.add_argument('--gentle-guide',action='store_true');p.add_argument('--guide-force',type=float,default=.6);p.add_argument('--probe',action='store_true');p.add_argument('--static-only',action='store_true');p.add_argument('--gui',action='store_true');p.add_argument('--speed',type=float,default=1);p.add_argument('--audit',action='store_true');a=p.parse_args()
    if a.audit:print(audit());return
    if not np.isfinite(a.yaw) or abs(a.yaw)>45 or not np.isfinite(a.pitch) or abs(a.pitch)>15 or not np.isfinite(a.speed) or a.speed<0 or not .1<=a.guide_force<=2:raise ValueError('Candidate bounds exceeded')
    run(a.name,a.yaw,a.pitch,a.guide,a.probe,a.static_only,a.gui,a.speed,a.guide_force,a.gentle_guide)
if __name__=='__main__':main()
