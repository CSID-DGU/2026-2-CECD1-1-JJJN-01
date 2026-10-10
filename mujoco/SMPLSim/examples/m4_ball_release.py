"""M4-2 actual-motion hand release; no test launcher or fitted file mutation."""
import argparse
import hashlib
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mujoco
import mujoco.viewer
import numpy as np
from examples.m3_control import contacts, write_csv, FINGERS
from examples.m4_basketball import trail
from smpl_sim.physics.ball_release import ReleaseControllerBridge, HandBallRelease, hand_ball_state
from smpl_sim.physics.basketball import HoopPassage
from smpl_sim.motion.physics_metrics import quaternion_error_deg


def simulate(args):
    args.output.mkdir(parents=True,exist_ok=True)
    model=mujoco.MjModel.from_xml_path(str(args.scene));data=mujoco.MjData(model)
    bridge=ReleaseControllerBridge(args.humanoid,model,args.source,args.hand,args.test_wrist_amplitude)
    bridge.initialize(model,data)
    grip=HandBallRelease(model,args.hand,args.offset,args.release_time,args.hand_collision)
    grip.update(data)
    qa,da=grip.qa,grip.da
    names=[bridge.model.joint(int(j)).name for j in bridge.bound.mapping.actuator_joint_ids]
    body_mask=np.array([not any(f in n for f in FINGERS) for n in names])
    rows=[];contact_rows=[];points=[];frames=[];states=[];times=[];errors=[]
    detector=HoopPassage();ball_contact_seen=False;gravity_errors=[];gravity_velocity_errors=[];discrete_errors=[]
    held_error=[];human_contact_frames=0;max_force=0.;max_depth=0.;fall_frames=0
    start_z=float(data.qpos[2]);postrelease_writes=0
    baseline_path=ROOT/'outputs/m3_spl_physics/aligned/physical_states.npz'
    baseline={}
    if baseline_path.exists():
        with np.load(baseline_path) as old:
            baseline={round(float(t)/model.opt.timestep):q.copy() for t,q in zip(old['time_s'],old['qpos'])}
    parity=[];held_parity=[]
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
    camera.lookat[:]=[1.4,0,1.65];camera.distance=6.;camera.azimuth=100;camera.elevation=-12
    renderer=mujoco.Renderer(model,height=540,width=960) if args.video else None
    context=mujoco.viewer.launch_passive(model,data) if args.gui else nullcontext(None)
    started=time.monotonic()
    with context as viewer:
        if viewer:
            with viewer.lock():
                viewer.cam.lookat[:]=camera.lookat;viewer.cam.distance=camera.distance
                viewer.cam.azimuth=camera.azimuth;viewer.cam.elevation=camera.elevation
        for step in range(round(args.duration/model.opt.timestep)+1):
            if viewer and not viewer.is_running():break
            with viewer.lock() if viewer else nullcontext():
                if step:
                    previous=data.qpos[qa:qa+3].copy()
                    bridge.control(data);mujoco.mj_step(model,data);mujoco.mj_forward(model,data)
                    saved_q=data.qpos[qa:qa+7].copy();saved_v=data.qvel[da:da+6].copy()
                    was_released=grip.released
                    grip.update(data)
                    if was_released and (not np.array_equal(saved_q,data.qpos[qa:qa+7]) or not np.array_equal(saved_v,data.qvel[da:da+6])):
                        postrelease_writes+=1
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or abs(data.time-step*model.opt.timestep)>1e-8:
                    raise RuntimeError('Invalid state or simulation time reset')
                p=data.qpos[qa:qa+3].copy();v=data.qvel[da:da+3].copy()
                if not grip.released:
                    hp=hand_ball_state(model,data,args.hand,grip.offset)[0]
                    held_error.append(float(np.linalg.norm(p-hp)))
                active=[]
                for i in range(data.ncon):
                    c=data.contact[i]
                    if grip.geom not in (c.geom1,c.geom2):continue
                    other=int(c.geom2 if c.geom1==grip.geom else c.geom1)
                    wrench=np.zeros(6);mujoco.mj_contactForce(model,data,i,wrench)
                    if wrench[0]<=.1:continue
                    human=model.geom_bodyid[other]>0
                    active.append(human);max_force=max(max_force,float(wrench[0]));max_depth=max(max_depth,-float(c.dist))
                    contact_rows.append({'time_s':float(data.time),'other':model.geom(other).name,'human':int(human),
                                         'force_N':float(wrench[0]),'penetration_m':max(0.,-float(c.dist))})
                human_contact_frames+=int(any(active))
                if grip.released:
                    elapsed=float(data.time-grip.event['actual_time_s'])
                    expected=np.asarray(grip.event['position_m'])+np.asarray(grip.event['velocity_world_m_s'])*elapsed+np.array([0,0,-.5*9.81*elapsed**2])
                    expected_v=np.asarray(grip.event['velocity_world_m_s'])+[0,0,-9.81*elapsed]
                    if not ball_contact_seen and not active:
                        gravity_errors.append(float(np.linalg.norm(p-expected)))
                        discrete_expected=expected+np.array([0,0,-.5*9.81*model.opt.timestep*elapsed])
                        discrete_errors.append(float(np.linalg.norm(p-discrete_expected)))
                        gravity_velocity_errors.append(float(np.linalg.norm(v-expected_v)))
                    ball_contact_seen=ball_contact_seen or bool(active)
                    if step:detector.update(previous,p,float(data.time),model.opt.timestep)
                # Human-only data keeps basketball-floor contacts out of fall/support metrics.
                bridge.data.qpos[:]=data.qpos[:bridge.model.nq];bridge.data.qvel[:]=data.qvel[:bridge.model.nv]
                mujoco.mj_forward(bridge.model,bridge.data)
                force,margin,nonfoot=contacts(bridge.model,bridge.data)
                up=bridge.data.xmat[bridge.model.body('Pelvis').id].reshape(3,3)[:,1]
                tilt=float(np.degrees(np.arccos(np.clip(up[2],-1,1))))
                fallen=bool(data.qpos[2]<.65*start_z or tilt>35 or nonfoot>0)
                fall_frames+=int(fallen)
                target=bridge.target(data.time);err=data.qpos[7:bridge.model.nq]-target[7:]
                if data.time<=bridge.motion.duration+1e-10:errors.append(err.copy())
                if step in baseline:
                    diff=float(np.max(np.abs(data.qpos[:bridge.model.nq]-baseline[step])))
                    parity.append(diff)
                    if not grip.released:held_parity.append(diff)
                row={'time_s':float(data.time),'released':int(grip.released),
                     **dict(zip(('ball_x_m','ball_y_m','ball_z_m'),map(float,p))),
                     **dict(zip(('vx_m_s','vy_m_s','vz_m_s'),map(float,v))),
                     'body_tracking_rmse_rad':float(np.sqrt(np.mean(err[body_mask]**2))),
                     'root_aligned_error_m':float(np.linalg.norm(data.qpos[:3]-target[:3])),
                     'root_orientation_error_deg':quaternion_error_deg(data.qpos[3:7],target[3:7]),
                     'com_x_m':float(bridge.data.subtree_com[1,0]),'com_y_m':float(bridge.data.subtree_com[1,1]),
                     'support_margin_m':float(margin),'left_foot_force_N':float(force['L']), 'right_foot_force_N':float(force['R']),
                     'fallen':int(fallen),'ball_hand_contact':int(any(active)),'ball_environment_contact':int(any(not h for h in active))}
                rows.append(row)
                if step%15==0:
                    points.append(p.copy());states.append(data.qpos.copy());times.append(float(data.time))
                    if viewer:viewer.user_scn.ngeom=0;trail(viewer.user_scn,points)
                    if renderer:
                        renderer.update_scene(data,camera=camera);trail(renderer.scene,points)
                        from PIL import Image,ImageDraw
                        picture=Image.fromarray(renderer.render());draw=ImageDraw.Draw(picture)
                        draw.text((15,15),f'{args.hand} hand | t={data.time:.3f}s | '+('RELEASED: physics flight' if grip.released else 'HELD: kinematic grip'),fill='yellow' if grip.released else 'white')
                        frames.append(picture.quantize(colors=256))
            if viewer and step%15==0:
                viewer.sync();time.sleep(max(0,data.time-(time.monotonic()-started)))
    if renderer:
        renderer.close();frames[0].save(args.output/'release.gif',save_all=True,append_images=frames[1:],loop=0,
                                      duration=[30 if i%3<2 else 40 for i in range(len(frames))],optimize=False,disposal=2)
    E=np.asarray(errors);rmse=np.sqrt(np.mean(E**2,axis=0))
    summary={'hand':args.hand,'offset_local_m':grip.offset.tolist(),'hand_collision':args.hand_collision,
             'test_wrist_amplitude_rad':args.test_wrist_amplitude,'release':grip.event,
             'completed':bool(abs(data.time-round(args.duration/model.opt.timestep)*model.opt.timestep)<1e-8),
             'duration_s':float(data.time),'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
             'scene_sha256':hashlib.sha256(args.scene.read_bytes()).hexdigest(),
             'max_held_error_m':max(held_error,default=0.),'ball_state_writes_after_release':postrelease_writes,
             'max_precontact_ballistic_position_error_m':max(gravity_errors,default=0.),
             'max_precontact_ballistic_velocity_error_m_s':max(gravity_velocity_errors,default=0.),
             'max_precontact_discrete_position_residual_m':max(discrete_errors,default=0.),
             'ballistic_samples':len(gravity_errors),'body_tracking_rmse_rad':float(np.sqrt(np.mean(E[:,body_mask]**2))),
             'root_aligned_rmse_m':float(np.sqrt(np.mean([r['root_aligned_error_m']**2 for r in rows if r['time_s']<=bridge.motion.duration]))),
             'root_orientation_rmse_deg':float(np.sqrt(np.mean([r['root_orientation_error_deg']**2 for r in rows if r['time_s']<=bridge.motion.duration]))),
             'min_support_margin_m':min((r['support_margin_m'] for r in rows if r['left_foot_force_N']>.1 or r['right_foot_force_N']>.1),default=-1.),
             'fall_count':sum(r['fallen'] and (i==0 or not rows[i-1]['fallen']) for i,r in enumerate(rows)),
             'both_feet_contact_rate':float(np.mean([r['left_foot_force_N']>.1 and r['right_foot_force_N']>.1 for r in rows])),
             'fall_frames':fall_frames,'hand_contact_frames':human_contact_frames,'max_ball_contact_force_N':max_force,
             'max_ball_penetration_m':max_depth,'passed_hoop':bool(detector.passed),
             'max_humanoid_qpos_difference_from_M3':max(parity,default=0.),'max_held_humanoid_qpos_difference_from_M3':max(held_parity,default=0.),
             'warnings':{str(i):int(w.number) for i,w in enumerate(data.warning) if w.number}}
    write_csv(args.output/'trajectory.csv',rows)
    if contact_rows:write_csv(args.output/'ball_contacts.csv',contact_rows)
    else:(args.output/'ball_contacts.csv').write_text('time_s,other,human,force_N,penetration_m\n')
    write_csv(args.output/'joint_tracking.csv',[{'joint':n,'rmse_rad':float(e)} for n,e in zip(names,rmse)])
    np.savez_compressed(args.output/'physical_states.npz',time_s=times,qpos=states)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    if args.check:
        if not summary['completed'] or not grip.event:raise AssertionError('Incomplete run or missing release')
        if summary['warnings'] or postrelease_writes or summary['max_held_error_m']>1e-10:raise AssertionError('Invalid release simulation')
        if grip.event['position_jump_m']>1e-10 or grip.event['velocity_jump_m_s']>1e-10:raise AssertionError('Discontinuous release')
        if summary['ballistic_samples']<2:raise AssertionError('No collision-free flight interval')
        if summary['max_precontact_discrete_position_residual_m']>1e-8:raise AssertionError('Unexpected free-flight position integration')
        if summary['max_precontact_ballistic_velocity_error_m_s']>1e-6:raise AssertionError('Unexpected free-flight acceleration')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--humanoid',type=Path,default=ROOT.parents[2]/'basketball-r1/SMPLSim/outputs/smplh_physics/body_01.xml')
    p.add_argument('--scene',type=Path,default=ROOT/'outputs/m4_basketball/body_01_basketball.xml')
    p.add_argument('--source',type=Path,default=ROOT/'outputs/m3_spl_fitting/spl_P0002_T0049.npz')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/m4_ball_release/default')
    p.add_argument('--hand',choices=('right','left'),default='right')
    p.add_argument('--offset',nargs=3,type=float,help='Wrist-local metres; overrides mirrored default')
    p.add_argument('--release-time',type=float,default=2.5)
    p.add_argument('--duration',type=float,default=6.)
    p.add_argument('--hand-collision',choices=('none','shooting','both'),default='shooting')
    p.add_argument('--test-wrist-amplitude',type=float,default=0.,help='Validation only; temporary wrist goal, does not modify NPZ')
    p.add_argument('--gui',action='store_true');p.add_argument('--video',action='store_true');p.add_argument('--check',action='store_true')
    args=p.parse_args()
    if not np.isfinite([args.duration,args.release_time,args.test_wrist_amplitude]).all() or not 0<=args.release_time<args.duration:
        p.error('Require finite parameters and 0 <= release-time < duration')
    simulate(args)

if __name__=='__main__':main()
