"""A-C contact/readiness comparisons. No PPO training or successful Release claim."""
import sys,argparse,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from examples.m5_coordinated_reference import CoordinatedEnv,OUT,reserve,scene_rim,unit,generate
from examples.m5_bimanual_contact import side_contacts,local_ball,gap_to_ball
from examples.m5_upper_body_ppo_eval import save
from examples.m3_control import write_csv
from smpl_sim.motion.physics_metrics import foot_contact_slip
from scipy.spatial.transform import Rotation
import mujoco,numpy as np


def readiness(e):
    m,d=e.m,e.d;center=d.qpos[e.ball_q:e.ball_q+3];W=d.xmat[e.wrist].reshape(3,3);normal=-W[:,1];rim=scene_rim(m,d);direction=unit(rim-center);rel,rv=local_ball(m,d,'R');R=side_contacts(m,d,'R');L=side_contacts(m,d,'L');r=reserve(m,d,e.ref['rel0'])
    chest=m.body('Chest').id;head=m.body('Head').id;C=d.xmat[chest].reshape(3,3);H=d.xmat[head].reshape(3,3);pel=d.xmat[m.body('Pelvis').id].reshape(3,3)
    hips=np.mean([d.xpos[m.body(n).id] for n in ('L_Hip','R_Hip')],axis=0);shoulders=np.mean([d.xpos[m.body(n).id] for n in ('L_Shoulder','R_Shoulder')],axis=0);trunk=unit(shoulders-hips);trunk_tilt=np.degrees(np.arccos(np.clip(trunk[2],-1,1)))
    tilt=np.degrees(np.arccos(np.clip(C[2,1],-1,1)));twist=np.degrees(Rotation.from_matrix(pel.T@C).as_rotvec()[1]);neck=np.degrees(Rotation.from_matrix(C.T@H).magnitude())
    cpos=d.geom_xpos[m.geom('R_Wrist').id]-center;f=e.ref['forward'];gap=min(gap_to_ball(m,d,n,center) for n in ('Head','Neck','Chest'))
    safety=[]
    ball=m.geom('basketball_geom').id
    for i in range(d.ncon):
        c=d.contact[i]
        if ball not in (c.geom1,c.geom2):continue
        other=c.geom2 if c.geom1==ball else c.geom1
        if m.geom(other).name in ('Head','Neck','Chest'):
            force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force);safety.append(float(force[0]))
    feet,_=foot_contact_slip(m,d)
    out=dict(time_s=d.time,phase=e.adapter.phase,goal_distance_mm=e.last_metrics['goal_distance_m']*1000,drift_mm=e.last_metrics['drift_m']*1000,palm_up_dot=float(normal[2]),palm_rim_dot=float(normal@direction),wrist_behind_ball_m=float(cpos@f),wrist_below_ball_m=float(cpos[2]),elbow_bend_deg=r['elbow_bend_deg'],elbow_extension_reserve_deg=r['elbow_extension_reserve_deg'],wrist_rim_rotation_reserve_rad=r['wrist_rim_rotation_reserve_rad'],wrist_rotation_gain_m_rad=r['ball_center_jacobian_rim_gain_m_rad'],left_clearance_m=e.adapter.clearance,right_feedback_offset_mm=float(np.linalg.norm(getattr(e.adapter,'feedback_offset',np.zeros(3)))*1000),nominal_wrist_actual_error_mm=float(np.linalg.norm(getattr(e.adapter,'nominal_cartesian_right',d.xpos[e.wrist])-d.xpos[e.wrist])*1000),left_drift_push_N=float(L['force']@unit(center-(d.xpos[e.wrist]+W@e.rel0))),anatomical_trunk_tilt_deg=trunk_tilt,left_normal_force_N=L['normal_force'],left_forward_force_N=float(L['force']@f),left_vertical_force_N=L['force'][2],right_forward_force_N=float(R['force']@f),right_vertical_force_N=R['force'][2],left_contact_count=L['count'],right_contact_count=R['count'],right_slip_m_s=R['slip'],left_slip_m_s=L['slip'],head_rim_error_deg=e.last_metrics['head_error_deg'],chest_tilt_deg=tilt,chest_relative_rotation_component_deg=twist,head_chest_relative_rotation_deg=neck,body_ball_gap_mm=gap*1000,body_ball_collision_force_N=max(safety,default=0),com_margin_mm=e.last_metrics['com_margin_m']*1000,both_feet=e.last_metrics['both_feet'],foot_slip_rms_m_s=float(np.sqrt((feet['L']['rms_speed']**2+feet['R']['rms_speed']**2)/2)),ball_forward_velocity_m_s=float(d.qvel[e.ball_v:e.ball_v+3]@f),ball_height_m=float(center[2]))
    for side,z in [('R',R),('L',L)]:
        for a,k in enumerate('xyz'):out[side+'_force_'+k+'_N']=float(z['force'][a])
    return out


def run(name,guide,tilt,scale,probe=False,gui=False,speed=1.,condition='B',ball_feedback=False,prep_seconds=1.5):
    if not np.isfinite(speed) or speed<0:raise ValueError('speed must be finite and >=0')
    folder=OUT/name
    if folder.exists():raise FileExistsError(f'Preserve previous trial: {folder}; select a new --name')
    folder.mkdir(parents=True);e=CoordinatedEnv(guide,tilt,scale,probe,condition,'human' if gui else None,ball_feedback=ball_feedback,prep_seconds=prep_seconds);o,info=e.reset(seed=7);rows=[];terms=[];ret=0;began=time.perf_counter();qinitial=e.d.qpos.copy();ready_since=None;stable_time=0.;probe_baseline=None
    try:
        while True:
            start=time.perf_counter();o,r,terminated,truncated,info=e.step(np.zeros(e.action_space.shape,np.float32));ret+=r;z=readiness(e);rows.append(z);terms.append(dict(time_s=e.d.time,**info['reward_terms']))
            ok=e.stage_success(e.last_metrics) and z['elbow_extension_reserve_deg']>=5 and z['wrist_rim_rotation_reserve_rad']>=.1 and z['body_ball_collision_force_N']<.1 and z['anatomical_trunk_tilt_deg']<30
            if ok:
                if ready_since is None:ready_since=e.d.time
                stable_time=max(stable_time,e.d.time-ready_since)
            else:ready_since=None
            if probe and e.d.time>=(prep_seconds+10.)*scale and probe_baseline is None:probe_baseline=z['ball_forward_velocity_m_s']
            if gui and speed>0:time.sleep(max(0,e.control_dt/speed-(time.perf_counter()-start)))
            if terminated or truncated or (gui and e.viewer is not None and not e.viewer.is_running()):break
        final=rows[-1];warnings=[int(w.number) for w in e.d.warning];success=stable_time>=.5 and info.get('termination_reason')=='time_limit' and not any(warnings)
        summary=dict(name=name,config=e.config,condition=condition,action_dim=e.action_space.shape[0],observation_dim=e.observation_space.shape[0],duration_s=e.d.time,wall_seconds=time.perf_counter()-began,termination_reason=info.get('termination_reason'),physical_readiness_success=bool(success),release_success=False,ppo_trained=False,stable_ready_duration_s=stable_time,minimum_goal_error_mm=min(r['goal_distance_mm'] for r in rows),max_drift_mm=max(r['drift_mm'] for r in rows),ball_rise_m=max(r['ball_height_m'] for r in rows)-qinitial[e.ball_q+2],maximum_left_normal_N=max(r['left_normal_force_N'] for r in rows),left_contact_fraction=float(np.mean([r['left_contact_count']>0 for r in rows])),left_clearance_final_m=final['left_clearance_m'],min_com_margin_mm=min(r['com_margin_mm'] for r in rows),both_feet_ratio=float(np.mean([r['both_feet'] for r in rows])),max_body_collision_force_N=max(r['body_ball_collision_force_N'] for r in rows),minimum_body_gap_mm=min(r['body_ball_gap_mm'] for r in rows),fallen=any(r['fallen'] for r in e.rows),warnings=warnings,ball_state_writes_after_reset=0,neq=e.m.neq,head_error_final_deg=final['head_rim_error_deg'],chest_tilt_final_deg=final['chest_tilt_deg'],anatomical_trunk_tilt_final_deg=final['anatomical_trunk_tilt_deg'],extension_reserve_final_deg=final['elbow_extension_reserve_deg'],wrist_reserve_final_rad=final['wrist_rim_rotation_reserve_rad'],probe_executed=bool(e.adapter.pulse_dirs is not None),probe_forward_velocity_change_m_s=None if probe_baseline is None else final['ball_forward_velocity_m_s']-probe_baseline,episode_return=ret)
        for side in ('R','L'):
            for key in ('ref_target_mm','target_actual_mm','ref_actual_mm','orientation_error_deg'):summary[side+'_'+key+'_rms']=float(np.sqrt(np.mean([r[side+'_'+key]**2 for r in e.rows])))
        save(e,folder,summary,terms);write_csv(folder/'readiness.csv',rows);print(json.dumps(summary),flush=True);return summary
    finally:e.close()


def audit():
    before=json.loads((OUT/'preservation_before.json').read_text());changed=[p for p,h in before.items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];scientific=[p for p in changed if Path(p).name!='.DS_Store'];data=dict(checked_files=len(before),scientific_files=sum(Path(p).name!='.DS_Store' for p in before),changed_scientific_files=scientific,changed_os_metadata=[p for p in changed if Path(p).name=='.DS_Store']);(OUT/'preservation_audit.json').write_text(json.dumps(data,indent=2))
    if scientific:raise RuntimeError(f'Protected files changed: {scientific}')
    return data


def report():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paths=sorted(OUT.glob('*/readiness.csv'));summaries=[];fig,axes=plt.subplots(4,1,figsize=(10,11),sharex=True,constrained_layout=True)
    for p in paths:
        s=json.loads((p.parent/'summary.json').read_text());summaries.append(s);x=np.genfromtxt(p,delimiter=',',names=True,dtype=None,encoding='utf8')
        for ax,key in zip(axes,['ball_height_m','drift_mm','left_normal_force_N','head_rim_error_deg']):ax.plot(x['time_s'],x[key],label=p.parent.name)
    for ax,label in zip(axes,['Ball height (m)','Ball drift (mm)','Left contact normal (N)','Head rim error (deg)']):ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(fontsize=7)
    axes[-1].set_xlabel('Simulation time (s)');fig.savefig(OUT/'coordination_comparison.png',dpi=130);plt.close(fig);write_csv(OUT/'comparison.csv',summaries)
    (OUT/'gate.json').write_text(json.dumps(dict(physical_shooting_ready_reference_passed=any(s['physical_readiness_success'] for s in summaries),large_ppo_allowed=False,reason='No large learning in this reference experiment; inspect contact/readiness gates.'),indent=2));print(json.dumps(audit()))


def gif(name):
    import imageio.v2 as imageio
    m=mujoco.MjModel.from_xml_path(str(ROOT/'outputs/m5_1_upper_body/bimanual_contact.xml'));d=mujoco.MjData(m);s=np.load(OUT/name/'states.npz');renderer=mujoco.Renderer(m,420,640);cam=mujoco.MjvCamera();cam.lookat[:]=[.2,0,1.25];cam.distance=2.3;cam.azimuth=110;cam.elevation=-12;frames=[]
    try:
        for q in s['qpos'][::3]:
            d.qpos[:]=q;mujoco.mj_forward(m,d);renderer.update_scene(d,camera=cam);frames.append(renderer.render().copy())
    finally:renderer.close()
    imageio.mimsave(OUT/(name+'.gif'),frames,duration=100,loop=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='manual');p.add_argument('--guide',choices=['off','fixed','phase','feather','light'],default='feather');p.add_argument('--tilt',type=float,default=8.);p.add_argument('--scale',type=float,default=1.);p.add_argument('--probe',action='store_true');p.add_argument('--prep',type=float,default=1.5);p.add_argument('--ball-feedback',action='store_true');p.add_argument('--gui',action='store_true');p.add_argument('--speed',type=float,default=1.);p.add_argument('--condition',choices=['B','C','D'],default='B');p.add_argument('--all',action='store_true');p.add_argument('--report',action='store_true');p.add_argument('--gif',action='store_true');a=p.parse_args()
    if a.report:return report()
    if a.gif:return gif(a.name)
    if not (OUT/'reference/reference.npz').exists():generate()
    if a.all:
        for name,guide,tilt,scale in [('R_control','off',0,1),('G_fixed','fixed',0,1),('G_phase','phase',0,1),('G_shooting8','phase',8,1),('G_shooting8_slow','phase',8,1.5)]:run(name,guide,tilt,scale)
        return report()
    run(a.name,a.guide,a.tilt,a.scale,a.probe,a.gui,a.speed,a.condition,a.ball_feedback,a.prep)
if __name__=='__main__':main()
