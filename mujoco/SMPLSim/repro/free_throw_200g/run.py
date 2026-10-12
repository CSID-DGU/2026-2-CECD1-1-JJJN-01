"""Standalone replay / fresh physics. Run with mjpython on macOS for GUI."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import mujoco
from engine import HERE,load_models,Simulation

def camera(viewer,view):
    viewer.cam.lookat[:]=[2.1,0,1.8] if view=='court' else [0,0,1.2]
    viewer.cam.distance=7 if view=='court' else 3.6
    viewer.cam.azimuth={'court':95,'side':90,'front':180}[view]
    viewer.cam.elevation=-12

def replay(m,d,cfg,args):
    p=HERE/'data/replay.csv'
    if hashlib.sha256(p.read_bytes()).hexdigest()!=cfg['replay_sha256']:raise ValueError('Replay checksum mismatch')
    rows=np.loadtxt(p,delimiter=',',skiprows=1)
    if rows.shape[1]!=m.nq+1 or not np.isfinite(rows).all():raise ValueError('Invalid replay data')
    if args.headless:
        for row in rows:
            d.qpos[:]=row[1:];mujoco.mj_forward(m,d)
        print(f'Replay validated: {len(rows)} frames, {rows[-1,0]:.3f}s. No physics stepping.');return
    from mujoco import viewer as viewer_module
    with viewer_module.launch_passive(m,d) as viewer:
        camera(viewer,args.view);n=0
        while viewer.is_running() and (args.repeats==0 or n<args.repeats):
            d.eq_active[:]=0;start=time.monotonic()
            for row in rows:
                if not viewer.is_running():return
                with viewer.lock():
                    d.qpos[:]=row[1:];d.qvel[:]=0;d.time=row[0];mujoco.mj_forward(m,d)
                viewer.sync();time.sleep(max(0,start+row[0]/args.speed-time.monotonic()))
            n+=1

def simulate(h,m,d,cfg,args):
    s=Simulation(h,m,d,cfg)
    if args.gui:
        from mujoco import viewer as viewer_module
        with viewer_module.launch_passive(m,d) as viewer:
            camera(viewer,args.view);start=time.monotonic()
            while viewer.is_running():
                with viewer.lock():done=s.step()
                viewer.sync();time.sleep(max(0,start+d.time/args.speed-time.monotonic()))
                if done:break
    else:
        while not s.step():pass
    if not s.release:raise RuntimeError('Simulation ended before release')
    result={'completed':bool(d.time>=s.motion.duration-1e-9),'score_first_flight':bool(s.scored),'fallen':bool(s.fallen),'warnings':sum(int(w.number) for w in d.warning),'release':s.release,'release_speed_m_s':float(np.linalg.norm(s.release['velocity_m_s'])),'ball_mass_kg':cfg['ball_mass_kg'],'ball_state_writes_after_reset':0}
    result['release_velocity_reference_error_m_s']=float(np.max(abs(np.array(s.release['velocity_m_s'])-cfg['expected_release_velocity_m_s'])))
    result['release_position_reference_error_m']=float(np.max(abs(np.array(s.release['position_m'])-cfg['expected_release_position_m'])))
    old=np.loadtxt(HERE/'data/replay.csv',delimiter=',',skiprows=1)
    if len(s.states)==len(old):result['maximum_replay_qpos_difference']=float(np.max(abs(np.array(s.states)-old[:,1:])))
    out=Path(args.output).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    np.savetxt(out/'states.csv',np.column_stack([s.times,s.states]),delimiter=',',header='time_s,'+','.join(f'qpos_{i:03d}' for i in range(m.nq)),comments='',fmt='%.17g')
    np.savetxt(out/'ball_flight.csv',np.array(s.flight),delimiter=',',header='time_s,x_m,y_m,z_m,floor_contact',comments='',fmt='%.17g')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print(f'Outputs: {out}')
    if args.verify:
        if mujoco.__version__!=cfg['mujoco_version']:raise RuntimeError('Exact verification requires MuJoCo '+cfg['mujoco_version'])
        if not (result['completed'] and s.scored and not s.fallen and result['warnings']==0 and result['release_velocity_reference_error_m_s']<1e-8 and result['release_position_reference_error_m']<1e-8 and result.get('maximum_replay_qpos_difference',float('inf'))<1e-8):raise RuntimeError('Reproduction differs: inspect summary.json; do not call this an identical result.')
        print('PASS: fresh physics reproduces the saved successful trial.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['replay','simulate']);p.add_argument('--humanoid',type=Path,required=True,help='External matching body_01.xml; never upload licensed model assets');p.add_argument('--view',choices=['court','side','front'],default='court');p.add_argument('--repeats',type=int,default=0,help='Replay repetitions; 0=infinite');p.add_argument('--speed',type=float,default=1,help='GUI playback/pacing multiplier; does not change physics');p.add_argument('--headless',action='store_true',help='Validate replay without opening GUI');p.add_argument('--gui',action='store_true',help='Show fresh physics simulation');p.add_argument('--verify',action='store_true',help='Assert exact fresh-physics reproduction');p.add_argument('--output',default='outputs/free_throw_repro');args=p.parse_args()
    if args.speed<=0 or args.repeats<0:p.error('speed must be positive; repeats must be nonnegative')
    h,m,d,cfg=load_models(args.humanoid.expanduser().resolve())
    if args.mode=='replay':replay(m,d,cfg,args)
    else:simulate(h,m,d,cfg,args)
if __name__=='__main__':main()
