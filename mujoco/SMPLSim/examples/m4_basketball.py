"""M4-1 physical basketball validation and GUI preview. Initial velocity is test-only."""
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
from examples.m3_control import write_csv
from smpl_sim.physics.basketball import (build_xml,HumanoidControllerBridge,test_ball_initial_state,
    HoopPassage,cases,BALL_RADIUS,BALL_MASS,RIM_CENTER,BOARD_FACE_X)


def trail(scene,points):
    for p in points[::max(1,len(points)//100)]:
        if scene.ngeom>=scene.maxgeom:break
        g=scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.018,0,0]),np.asarray(p),
                           np.eye(3).ravel(),np.array([1,.9,.1,1],dtype=np.float32))
        scene.ngeom+=1


def simulate(name,position,velocity,duration,args):
    model=mujoco.MjModel.from_xml_path(str(args.output/'body_01_basketball.xml'))
    data=mujoco.MjData(model)
    bridge=HumanoidControllerBridge(args.humanoid,model,args.source)
    bridge.initialize(model,data)
    test_ball_initial_state(model,data,position,velocity)
    ball_id=model.body('basketball').id
    joint=model.joint('basketball_free');qa,da=int(joint.qposadr[0]),int(joint.dofadr[0])
    detector=HoopPassage();rows=[];contacts=[];points=[];states=[]
    contact_counts={'floor':0,'backboard':0,'rim':0,'human':0}
    first_impact=None;floor_exit=None;first_rebound_peak=None;impact_velocity=None;exit_velocity=None
    had_contact=False;precontact_errors=[]
    was_floor=False;penetration=0.;camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
    camera.lookat[:]=[2.5,0,1.8];camera.distance=8.;camera.azimuth=100;camera.elevation=-12
    renderer=mujoco.Renderer(model,height=540,width=960) if args.video else None
    frames=[];context=mujoco.viewer.launch_passive(model,data) if args.gui else nullcontext(None)
    wall_start=time.monotonic();previous=data.qpos[qa:qa+3].copy()
    with context as viewer:
        if viewer:
            with viewer.lock():
                viewer.cam.lookat[:]=camera.lookat;viewer.cam.distance=camera.distance
                viewer.cam.azimuth=camera.azimuth;viewer.cam.elevation=camera.elevation
        for step in range(round(duration/model.opt.timestep)+1):
            if viewer and not viewer.is_running():break
            with viewer.lock() if viewer else nullcontext():
                old_velocity=data.qvel[da:da+3].copy()
                if step:
                    previous=data.qpos[qa:qa+3].copy()
                    before=float(data.time);bridge.control(data);mujoco.mj_step(model,data)
                    if data.time<=before or not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                        raise RuntimeError('Invalid physical state/time reset')
                mujoco.mj_forward(model,data)
                current=data.qpos[qa:qa+3].copy();vel=data.qvel[da:da+3].copy()
                if step:detector.update(previous,current,float(data.time),model.opt.timestep)
                active=set();wrench=np.zeros(6)
                for i in range(data.ncon):
                    c=data.contact[i]
                    if model.geom('basketball_geom').id not in (c.geom1,c.geom2):continue
                    other=c.geom2 if c.geom1==model.geom('basketball_geom').id else c.geom1
                    other_name=model.geom(int(other)).name
                    kind='rim' if other_name.startswith('rim_') else 'backboard' if other_name=='backboard' else 'floor' if other_name=='floor' else 'human'
                    mujoco.mj_contactForce(model,data,i,wrench)
                    if wrench[0]>.1:
                        active.add(kind);contact_counts[kind]+=1
                        penetration=min(penetration,float(c.dist))
                        contacts.append({'time_s':float(data.time),'other':other_name,'kind':kind,
                                         'normal_force_N':float(wrench[0]),'distance_m':float(c.dist)})
                floor='floor' in active
                if floor and first_impact is None:first_impact=float(data.time);impact_velocity=float(old_velocity[2])
                if was_floor and not floor and floor_exit is None:
                    floor_exit=float(data.time);exit_velocity=float(vel[2])
                if floor_exit is not None and first_rebound_peak is None and vel[2]<=0:
                    first_rebound_peak=float(current[2]-BALL_RADIUS)
                was_floor=floor
                p_analytic=np.asarray(position)+np.asarray(velocity)*data.time+np.array([0,0,-.5*9.81*data.time**2])
                row={'time_s':float(data.time),'ball_x_m':float(current[0]),'ball_y_m':float(current[1]),'ball_z_m':float(current[2]),
                     'vx_m_s':float(vel[0]),'vy_m_s':float(vel[1]),'vz_m_s':float(vel[2]),
                     'analytic_position_error_m':float(np.linalg.norm(current-p_analytic)),
                     **{k+'_contact':int(k in active) for k in contact_counts},'passed':int(detector.passed)}
                if not had_contact and not active:precontact_errors.append(row['analytic_position_error_m'])
                had_contact=had_contact or bool(active)
                rows.append(row)
                if step%15==0:
                    points.append(current.copy());states.append(data.qpos.copy())
                    if viewer:viewer.user_scn.ngeom=0;trail(viewer.user_scn,points)
                    if renderer:
                        renderer.update_scene(data,camera=camera);trail(renderer.scene,points)
                        from PIL import Image,ImageDraw
                        picture=Image.fromarray(renderer.render())
                        ImageDraw.Draw(picture).text((15,15),f'{name} | t={data.time:.3f}s | TEST initial velocity, no hand release',fill='white')
                        frames.append(picture.quantize(colors=256))
            if viewer and step%15==0:
                viewer.sync();time.sleep(max(0,data.time-(time.monotonic()-wall_start)))
    if renderer:
        renderer.close()
        frames[0].save(args.output/f'{name}.gif',save_all=True,append_images=frames[1:],loop=0,
                       duration=[30 if i%3<2 else 40 for i in range(len(frames))],optimize=False,disposal=2)
    summary={'case':name,'initial_position_m':np.asarray(position).tolist(),'initial_velocity_m_s':np.asarray(velocity).tolist(),
             'duration_s':float(data.time),'expected_duration_s':round(duration/model.opt.timestep)*model.opt.timestep,
             'completed':bool(abs(data.time-round(duration/model.opt.timestep)*model.opt.timestep)<1e-8),'ball_mass_kg':float(model.body_mass[ball_id]),
             'max_precontact_analytic_position_error_m':max(precontact_errors,default=0.),
             'max_ball_z_m':max(r['ball_z_m'] for r in rows),'max_analytic_position_error_m':max(r['analytic_position_error_m'] for r in rows),
             'floor_first_contact_s':first_impact,'first_rebound_bottom_height_m':first_rebound_peak,
             'impact_vz_m_s':impact_velocity,'exit_vz_m_s':exit_velocity,
             'effective_restitution':abs(exit_velocity/impact_velocity) if impact_velocity and exit_velocity is not None else None,
             'max_ball_penetration_m':-penetration,'passed':detector.passed,'pass_entry_time_s':detector.entry_time,
             **{k+'_contact_samples':v for k,v in contact_counts.items()},
             'mujoco_warnings':int(sum(w.number for w in data.warning)),'final_vx_m_s':float(data.qvel[da]),
             'final_ball_z_m':float(data.qpos[qa+2]),'finite':bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
             'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest()}
    write_csv(args.output/f'{name}_timeseries.csv',rows)
    if contacts:write_csv(args.output/f'{name}_contacts.csv',contacts)
    (args.output/f'{name}_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez_compressed(args.output/f'{name}_states.npz',qpos=states,fps=30)
    print(json.dumps(summary),flush=True)
    return summary


def check(rows):
    by={r['case']:r for r in rows}
    for r in rows:
        assert r['completed'] and r['finite'] and r['mujoco_warnings']==0 and r['human_contact_samples']==0,r
    if 'freefall' in by:
        r=by['freefall'];assert r['max_analytic_position_error_m']<.004 and not any(r[k+'_contact_samples'] for k in ('floor','backboard','rim'))
    if 'floor' in by:
        r=by['floor'];assert r['floor_contact_samples'] and r['exit_vz_m_s']>0 and .5<r['effective_restitution']<1
        assert 1.035<=r['first_rebound_bottom_height_m']<=1.085
    if 'backboard' in by:assert by['backboard']['backboard_contact_samples'] and not by['backboard']['rim_contact_samples'] and by['backboard']['final_vx_m_s']<0
    if 'rim' in by:assert by['rim']['rim_contact_samples']
    if 'pass' in by:assert by['pass']['passed'] and not by['pass']['rim_contact_samples']
    if 'launch' in by:assert by['launch']['passed']


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--humanoid',type=Path,default=ROOT.parents[2]/'basketball-r1/SMPLSim/outputs/smplh_physics/body_01.xml')
    p.add_argument('--source',type=Path,default=ROOT/'outputs/m3_spl_fitting/spl_P0002_T0049.npz')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/m4_basketball')
    p.add_argument('--case',choices=['all',*cases()],default='all')
    p.add_argument('--position',type=float,nargs=3,help='TEST initial xyz (m), requires single case')
    p.add_argument('--velocity',type=float,nargs=3,help='TEST initial xyz velocity (m/s), requires single case')
    p.add_argument('--duration',type=float)
    p.add_argument('--gui',action='store_true');p.add_argument('--video',action='store_true');p.add_argument('--check',action='store_true')
    args=p.parse_args()
    if args.case=='all' and (args.position or args.velocity or args.duration):p.error('Initial state overrides require one case')
    if args.duration is not None and (args.duration<=0 or not np.isfinite(args.duration)):p.error('duration must be positive and finite')
    args.output.mkdir(parents=True,exist_ok=True)
    before=hashlib.sha256(args.source.read_bytes()).hexdigest()
    build_xml(args.humanoid,args.output/'body_01_basketball.xml')
    results=[]
    for name,(position,velocity,duration) in cases().items():
        if args.case not in ('all',name):continue
        results.append(simulate(name,args.position if args.position is not None else position,
                                args.velocity if args.velocity is not None else velocity,
                                args.duration if args.duration is not None else duration,args))
    write_csv(args.output/'summary.csv',results)
    if args.check:check(results)
    assert hashlib.sha256(args.source.read_bytes()).hexdigest()==before
    print('M4-1 finished; fitted motion unchanged.',flush=True)
