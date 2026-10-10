"""A no ball / B kinematic / C physical weld: identical M3 goals and controller."""
import argparse
import json
import sys
import time
from pathlib import Path
from contextlib import nullcontext
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import mujoco.viewer
import numpy as np
from examples.m3_control import contacts,write_csv,FINGERS
from examples.m4_basketball import trail
from smpl_sim.physics.basketball import HumanoidControllerBridge
from smpl_sim.physics.ball_release import HandBallRelease,hand_ball_state
from smpl_sim.physics.physical_holding import WeldBallRelease,build_holding_xml
from smpl_sim.motion.physics_metrics import quaternion_error_deg,foot_contact_slip


def support_margin_at(model,data,xy):
    from scipy.spatial import ConvexHull
    floor=model.geom('floor').id;geoms=set();points=[]
    for i in range(data.ncon):
        c=data.contact[i]
        if floor not in (c.geom1,c.geom2):continue
        gid=int(c.geom2 if c.geom1==floor else c.geom1)
        if model.body(int(model.geom_bodyid[gid])).name not in ('L_Ankle','L_Toe','R_Ankle','R_Toe'):continue
        w=np.zeros(6);mujoco.mj_contactForce(model,data,i,w)
        if w[0]>.1:geoms.add(gid)
    for gid in geoms:
        if model.geom_type[gid]!=mujoco.mjtGeom.mjGEOM_BOX:continue
        for x in (-1,1):
            for y in (-1,1):
                for z in (-1,1):points.append((data.geom_xpos[gid]+data.geom_xmat[gid].reshape(3,3)@(model.geom_size[gid]*[x,y,z]))[:2])
    if len(points)<3:return -1.
    e=ConvexHull(np.unique(np.round(points,8),axis=0)).equations
    return float(np.min(-(e[:,:2]@xy+e[:,2])/np.linalg.norm(e[:,:2],axis=1)))


def simulate(args,body,mode):
    output=args.output/body/mode;output.mkdir(parents=True,exist_ok=True)
    original=args.models/(body+'.xml')
    path=original if mode=='A' else build_holding_xml(original,output/'holding.xml',args.hand,args.offset,mode=='C',args.test_ball_mass)
    model=mujoco.MjModel.from_xml_path(str(path));model.opt.timestep=1/450
    model.vis.global_.offwidth=960;model.vis.global_.offheight=540
    data=mujoco.MjData(model);bridge=HumanoidControllerBridge(original,model,args.source);bridge.initialize(model,data)
    grip=None
    if mode!='A':
        grip=(WeldBallRelease if mode=='C' else HandBallRelease)(model,args.hand,args.offset,args.release_time,args.hand_collision)
        if mode=='C':grip.initialize(data)
        else:grip.update(data)
    names=[bridge.model.joint(int(j)).name for j in bridge.bound.mapping.actuator_joint_ids]
    body_mask=np.array([not any(f in n for f in FINGERS) for n in names]);dofs=bridge.bound.mapping.actuator_dof
    rows=[];contactrows=[];errors=[];torques=[];torque_times=[];states=[];velocities=[];times=[];points=[];frames=[]
    hold_errors=[];orientation_errors=[];weld_forces=[];state_writes=0
    start_z=data.qpos[2];weld_load=np.zeros(model.nv);ball_contact_seen=False;gravity=[];gravityv=[]
    renderer=mujoco.Renderer(model,height=540,width=960) if args.video else None
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
    camera.lookat[:]=[1.4,0,1.65];camera.distance=6;camera.azimuth=100;camera.elevation=-12
    context=mujoco.viewer.launch_passive(model,data) if args.gui else nullcontext(None);started=time.monotonic()
    with context as viewer:
        if viewer:
            with viewer.lock():
                viewer.cam.lookat[:]=camera.lookat;viewer.cam.distance=camera.distance;viewer.cam.azimuth=camera.azimuth;viewer.cam.elevation=camera.elevation
        for step in range(round(args.duration/model.opt.timestep)+1):
            if viewer and not viewer.is_running():break
            with viewer.lock() if viewer else nullcontext():
                if step:
                    bridge.control(data);mujoco.mj_step(model,data);mujoco.mj_forward(model,data)
                    if grip:
                        p0=data.qpos[grip.qa:grip.qa+7].copy();v0=data.qvel[grip.da:grip.da+6].copy();released_before=grip.released
                        grip.update(data)
                        if (mode=='C' or released_before) and (not np.array_equal(p0,data.qpos[grip.qa:grip.qa+7]) or not np.array_equal(v0,data.qvel[grip.da:grip.da+6])):state_writes+=1
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or abs(data.time-step*model.opt.timestep)>1e-8:raise RuntimeError('Invalid physical state')
                target=bridge.target(data.time);error=data.qpos[7:bridge.model.nq]-target[7:]
                tau=data.qfrc_actuator[dofs].copy()
                if data.time<=bridge.motion.duration+1e-10:
                    errors.append(error);torques.append(tau);torque_times.append(float(data.time))
                # Human geometry support with actual integrated-model contact forces.
                force,margin,nonfoot=contacts(model,data)
                if grip:
                    # Existing helper counts ball-floor contact as nonfoot; remove those.
                    nonfoot=0
                    for i in range(data.ncon):
                        c=data.contact[i];floor=model.geom('floor').id
                        if floor not in (c.geom1,c.geom2):continue
                        other=int(c.geom2 if c.geom1==floor else c.geom1);bid=int(model.geom_bodyid[other])
                        if bid==model.body('basketball').id:continue
                        name=model.body(bid).name
                        if name not in ('L_Ankle','L_Toe','R_Ankle','R_Toe'):
                            wrench=np.zeros(6);mujoco.mj_contactForce(model,data,i,wrench);nonfoot+=int(wrench[0]>.1)
                slip,_=foot_contact_slip(model,data)
                up=data.xmat[model.body('Pelvis').id].reshape(3,3)[:,1];tilt=np.degrees(np.arccos(np.clip(up[2],-1,1)))
                fallen=int(data.qpos[2]<.65*start_z or tilt>35 or nonfoot>0)
                mass=model.body_mass[1:bridge.model.nbody];human_com=mass@data.xipos[1:bridge.model.nbody]/mass.sum()
                ball_force=np.zeros(3);ball_contact=False;handcontact=False;depth=0.;handforce=0.
                weld_load[:]=0
                if grip:
                    ball_position=data.qpos[grip.qa:grip.qa+3].copy();ball_velocity=data.qvel[grip.da:grip.da+3].copy()
                    if not grip.released:
                        expected_p,expected_q,_,_=hand_ball_state(model,data,args.hand,grip.offset)
                        hold_errors.append(np.linalg.norm(ball_position-expected_p));orientation_errors.append(quaternion_error_deg(data.qpos[grip.qa+3:grip.qa+7],expected_q))
                    for i in range(data.ncon):
                        c=data.contact[i]
                        if grip.geom not in (c.geom1,c.geom2):continue
                        other=int(c.geom2 if c.geom1==grip.geom else c.geom1);wrench=np.zeros(6);mujoco.mj_contactForce(model,data,i,wrench)
                        if wrench[0]>.1:
                            hand=model.geom_bodyid[other]>0;ball_contact=True;handcontact=handcontact or hand
                            if hand:depth=max(depth,-float(c.dist));handforce=max(handforce,float(wrench[0]))
                            contactrows.append({'time_s':float(data.time),'other':model.geom(other).name,'human':int(hand),'normal_force_N':float(wrench[0]),'penetration_m':max(0.,-float(c.dist))})
                    if mode=='C' and not grip.released:
                        mask=(data.efc_type==mujoco.mjtConstraint.mjCNSTR_EQUALITY)&(data.efc_id==grip.equality)
                        load=np.where(mask,data.efc_force,0.)
                        mujoco.mj_mulJacTVec(model,data,weld_load,load)
                        ball_force=weld_load[grip.da:grip.da+3].copy();weld_forces.append(ball_force)
                    if grip.released:
                        elapsed=data.time-grip.event['actual_time_s'];p=np.asarray(grip.event['position_m']);v=np.asarray(grip.event['velocity_world_m_s'])
                        if not ball_contact_seen and not ball_contact:
                            expected=p+v*elapsed+[0,0,-.5*9.81*elapsed*(elapsed+model.opt.timestep)]
                            gravity.append(float(np.linalg.norm(ball_position-expected)))
                            gravityv.append(float(np.linalg.norm(ball_velocity-v-[0,0,-9.81*elapsed])))
                        ball_contact_seen=ball_contact_seen or ball_contact
                    total_com=(mass@data.xipos[1:bridge.model.nbody]+model.body_mass[model.body('basketball').id]*ball_position)/(mass.sum()+model.body_mass[model.body('basketball').id])
                else:total_com=human_com
                coupled_margin=support_margin_at(model,data,total_com[:2]) if mode=='C' and not grip.released else margin
                row={'time_s':float(data.time),'holding':int(grip is not None and not grip.released),'body_rmse_rad':float(np.sqrt(np.mean(error[body_mask]**2))),
                     'root_rmse_m':float(np.linalg.norm(data.qpos[:3]-target[:3])),'root_orientation_error_deg':quaternion_error_deg(data.qpos[3:7],target[3:7]),
                     'support_margin_m':float(margin),'coupled_support_margin_m':float(coupled_margin),'pelvis_tilt_deg':float(tilt),'human_com_x_m':float(human_com[0]),'human_com_y_m':float(human_com[1]),
                     'human_ball_com_x_m':float(total_com[0]),'human_ball_com_y_m':float(total_com[1]),
                     'L_force_N':float(force['L']),'R_force_N':float(force['R']),
                     'L_slip_rms_m_s':float(slip['L']['rms_speed']),'R_slip_rms_m_s':float(slip['R']['rms_speed']),
                     'fallen':fallen,'hand_contact':int(handcontact),'hand_force_N':handforce,'hand_penetration_m':depth,
                     'weld_ball_force_x_N':float(ball_force[0]),'weld_ball_force_y_N':float(ball_force[1]),'weld_ball_force_z_N':float(ball_force[2]),
                     'joint_torque_rms_Nm':float(np.sqrt(np.mean(tau**2)))}
                if grip:row.update(dict(zip(('ball_x_m','ball_y_m','ball_z_m','vx_m_s','vy_m_s','vz_m_s'),map(float,np.r_[ball_position,ball_velocity]))))
                rows.append(row)
                if step%15==0:
                    states.append(data.qpos.copy());velocities.append(data.qvel.copy());times.append(float(data.time))
                    if grip:points.append(ball_position.copy())
                    if viewer:viewer.user_scn.ngeom=0;trail(viewer.user_scn,points)
                    if renderer:
                        renderer.update_scene(data,camera=camera);trail(renderer.scene,points)
                        from PIL import Image,ImageDraw
                        im=Image.fromarray(renderer.render());ImageDraw.Draw(im).text((15,15),f'{body} | {mode} | t={data.time:.3f}s | '+('NO BALL' if not grip else 'RELEASED' if grip.released else 'WELD HOLD' if mode=='C' else 'KINEMATIC HOLD'),fill='yellow')
                        frames.append(im.quantize(colors=256))
            if viewer and step%15==0:viewer.sync();time.sleep(max(0,data.time-(time.monotonic()-started)))
    if renderer:
        renderer.close();frames[0].save(output/'holding.gif',save_all=True,append_images=frames[1:],loop=0,duration=[30 if i%3<2 else 40 for i in range(len(frames))],optimize=False,disposal=2)
    E=np.asarray(errors);T=np.asarray(torques);tt=np.asarray(torque_times);held=tt<args.release_time
    motionrows=[r for r in rows if r['time_s']<=bridge.motion.duration+1e-10]
    summary={'body':body,'mode':mode,'ball_mass_kg':float(model.body('basketball').mass[0]) if grip else 0.,
        'gain_scale':bridge.controller.gain_scale,
        'duration_s':float(data.time),'completed':bool(abs(data.time-round(args.duration/model.opt.timestep)*model.opt.timestep)<1e-8),
        'body_joint_rmse_rad':float(np.sqrt(np.mean(E[:,body_mask]**2))),'root_position_rmse_m':float(np.sqrt(np.mean([r['root_rmse_m']**2 for r in motionrows]))),
        'both_feet_contact_rate':float(np.mean([r['L_force_N']>.1 and r['R_force_N']>.1 for r in motionrows])),
        'min_support_margin_m':min((r['support_margin_m'] for r in motionrows if r['L_force_N']>.1 or r['R_force_N']>.1),default=-1.),
        'min_coupled_support_margin_m':min((r['coupled_support_margin_m'] for r in motionrows if r['L_force_N']>.1 or r['R_force_N']>.1),default=-1.),
        'balance_fraction':float(np.mean([r['pelvis_tilt_deg']<25 and r['coupled_support_margin_m']>=-.02 and not r['fallen'] and r['L_force_N']+r['R_force_N']>1 for r in motionrows])),
        'foot_slip_rms_m_s':float(np.sqrt(np.mean([.5*(r['L_slip_rms_m_s']**2+r['R_slip_rms_m_s']**2) for r in motionrows]))),
        'joint_torque_rms_Nm':float(np.sqrt(np.mean(T**2))),'hold_joint_torque_rms_Nm':float(np.sqrt(np.mean(T[held]**2))) if held.any() else 0.,
        'fall_count':sum(r['fallen'] and (i==0 or not rows[i-1]['fallen']) for i,r in enumerate(rows)),
        'warnings':{str(i):int(w.number) for i,w in enumerate(data.warning) if w.number},'release':grip.event if grip else None,
        'max_holding_position_error_m':float(max(hold_errors,default=0.)),'max_holding_orientation_error_deg':float(max(orientation_errors,default=0.)),
        'state_writes_during_weld_or_after_release':state_writes,'hand_contact_frames':sum(r['hand_contact'] for r in rows),
        'max_hand_contact_force_N':max(r['hand_force_N'] for r in rows),'max_hand_penetration_m':max(r['hand_penetration_m'] for r in rows),
        'ballistic_samples':len(gravity),'max_ballistic_discrete_residual_m':max(gravity,default=0.),'max_ballistic_velocity_residual_m_s':max(gravityv,default=0.),
        'initial_weld_error':grip.initial_error if mode=='C' else None,'initial_weld_max_constraint_force':grip.initial_force if mode=='C' else None,
        'mean_weld_ball_force_world_N':np.mean(weld_forces,axis=0).tolist() if weld_forces else None,
        'ball_state_initialization_writes':grip.state_write_count if mode=='C' else None}
    write_csv(output/'trajectory.csv',rows)
    if contactrows:write_csv(output/'contacts.csv',contactrows)
    write_csv(output/'joint_torques.csv',[{'joint':n,'tracking_rmse_rad':float(np.sqrt(np.mean(E[:,i]**2))),
        'torque_rms_Nm':float(np.sqrt(np.mean(T[:,i]**2))),'hold_torque_rms_Nm':float(np.sqrt(np.mean(T[held,i]**2))) if held.any() else 0.,
        'max_abs_torque_Nm':float(np.max(np.abs(T[:,i])))} for i,n in enumerate(names)])
    np.savez_compressed(output/'states.npz',time_s=times,qpos=states,qvel=velocities)
    np.savez_compressed(output/'torques.npz',time_s=torque_times,torque_Nm=T,joint_names=names)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
    if args.check:
        if summary['warnings'] or not summary['completed'] or summary['fall_count'] or state_writes:raise AssertionError(body+'/'+mode+' unstable/incomplete')
        if grip:
            if not grip.event or grip.event['position_jump_m']>1e-12 or grip.event['velocity_jump_m_s']>1e-12:raise AssertionError('Invalid release')
            if len(gravity)<2 or max(gravity)>1e-8 or max(gravityv)>1e-8:raise AssertionError('Invalid postrelease free flight')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,default=ROOT.parents[2]/'basketball-r1/SMPLSim/outputs/smplh_physics')
    p.add_argument('--source',type=Path,default=ROOT/'outputs/m3_spl_fitting/spl_P0002_T0049.npz')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/m4_physical_holding')
    p.add_argument('--body',choices=['body_01','body_02','body_03','body_04'],default='body_01');p.add_argument('--all',action='store_true')
    p.add_argument('--mode',choices=('A','B','C','all'),default='C');p.add_argument('--hand',choices=('right','left'),default='right')
    p.add_argument('--offset',nargs=3,type=float);p.add_argument('--release-time',type=float,default=2.5)
    p.add_argument('--hand-collision',choices=('none','shooting','both'),default='shooting')
    p.add_argument('--test-ball-mass',type=float,default=.6,help='Separate diagnostic only; default M4-1 mass is 0.6 kg')
    p.add_argument('--duration',type=float,default=6.);p.add_argument('--gui',action='store_true');p.add_argument('--video',action='store_true');p.add_argument('--check',action='store_true')
    args=p.parse_args()
    if not np.isfinite([args.duration,args.release_time]).all() or not 0<=args.release_time<args.duration:p.error('Invalid time interval')
    results=[]
    for body in ('body_01','body_02','body_03','body_04') if args.all else (args.body,):
        for mode in ('A','B','C') if args.mode=='all' else (args.mode,):results.append(simulate(args,body,mode))
    write_csv(args.output/'comparison.csv',[{k:s[k] for k in ('body','mode','ball_mass_kg','body_joint_rmse_rad','root_position_rmse_m','both_feet_contact_rate','balance_fraction','min_support_margin_m','min_coupled_support_margin_m','foot_slip_rms_m_s','joint_torque_rms_Nm','hold_joint_torque_rms_Nm','fall_count','hand_contact_frames')} for s in results])
    print('PHYSICAL HOLDING VALIDATION COMPLETED')

if __name__=='__main__':main()
