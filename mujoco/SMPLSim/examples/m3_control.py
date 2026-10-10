"""M3 PD motion tracking and balance comparison, with an unactuated root."""
import argparse
import csv
import json
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from smpl_sim.motion.smplh_motion import SMPLHMotion, SMPLHMapping, SMPL_TO_WORLD
from smpl_sim.smpllib.smpl_joint_names import SMPLH_BONE_ORDER_NAMES
from smpl_sim.envs.controllers import StablePDController
from examples.smplh_bodies import initial_state, SHAPES

FINGERS = ('Index','Middle','Pinky','Ring','Thumb')
DURATIONS = {'stand': 6., 'arms': 8., 'knees': 8.}


def smooth(x):
    x = np.clip(x, 0, 1)
    return x*x*(3-2*x)


def reference(kind, model):
    """Same anatomical angle trajectory for every shape; adjust root clearance."""
    fps = 30
    times = np.arange(round(DURATIONS[kind]*fps)+1)/fps
    poses = np.zeros((len(times), 52, 3))
    for side, sign in [('L',-1),('R',1)]:
        poses[:, SMPLH_BONE_ORDER_NAMES.index(side+'_Shoulder'), 2] = sign*1.35
        for part, angle in [('Hip',-.05),('Knee',.1),('Ankle',-.05)]:
            poses[:, SMPLH_BONE_ORDER_NAMES.index(side+'_'+part), 0] = angle
    # Build desired rotations as XYZ products, then convert to SMPL axis-angle.
    envelope = smooth((times-2)/2) * (1-smooth((times-5)/2))
    if kind == 'arms':
        for side in ('L','R'):
            poses[:, SMPLH_BONE_ORDER_NAMES.index(side+'_Shoulder'),0] = -1.3*envelope
    if kind == 'knees':
        for side in ('L','R'):
            for part, angle in [('Hip',-.2),('Knee',.4),('Ankle',-.2)]:
                poses[:, SMPLH_BONE_ORDER_NAMES.index(side+'_'+part),0] += angle*envelope
    poses = Rotation.from_euler('XYZ', poses.reshape(-1,3)).as_rotvec().reshape(-1,156)
    d = mujoco.MjData(model)
    initial_state(model, d, clearance=.005)
    mapping = SMPLHMapping(model)
    tmp = mapping.convert(poses[:1], np.zeros((1,3)))
    d.qpos[mapping.actuator_qpos] = tmp[0,mapping.actuator_qpos]
    ground(model,d)
    root = np.tile(SMPL_TO_WORLD.inv().apply(d.qpos[:3]), (len(times),1))
    return SMPLHMotion(poses,root,fps)


def ground(m,d):
    mujoco.mj_forward(m,d)
    bottoms=[]
    for i in np.flatnonzero(m.geom_bodyid>0):
        s,r=m.geom_size[i],d.geom_xmat[i].reshape(3,3)
        typ=m.geom_type[i]
        if typ == mujoco.mjtGeom.mjGEOM_BOX: extent=np.abs(r[2])@s
        elif typ == mujoco.mjtGeom.mjGEOM_CAPSULE: extent=s[0]+abs(r[2,2])*s[1]
        elif typ == mujoco.mjtGeom.mjGEOM_SPHERE: extent=s[0]
        else: raise ValueError('Unsupported geom')
        bottoms.append(d.geom_xpos[i,2]-extent)
    d.qpos[2]+=.005-min(bottoms)
    mujoco.mj_forward(m,d)


class BalancedPD:
    def __init__(self,m,mapping):
        self.m,self.mapping=m,mapping
        if not np.array_equal(mapping.actuator_qpos, np.arange(7,m.nq)) or not np.array_equal(mapping.actuator_dof, np.arange(6,m.nv)):
            raise ValueError('StablePD expects actuator order to match hinge qpos/dof order')
        names=[m.joint(int(j)).name for j in mapping.actuator_joint_ids]
        self.index={n:i for i,n in enumerate(names)}
        kp,kd,limit=np.full(m.nu,300.),np.full(m.nu,30.),np.full(m.nu,150.)
        for i,n in enumerate(names):
            if any(k in n for k in ('Hip','Knee','Ankle','Toe','Torso','Spine','Chest')):
                kp[i],kd[i],limit[i]=1000.,100.,500.
            if any(k in n for k in FINGERS):
                kp[i],kd[i],limit[i]=5.,.5,2.
        # Same gain rule across shapes: scale rotational dynamics by m*l^2.
        leg_length = np.mean([sum(np.linalg.norm(m.body(n).pos) for n in
                              (side+'_Knee', side+'_Ankle')) for side in ('L','R')])
        self.gain_scale = float(m.body_mass.sum()/80 * (leg_length/.82)**2)
        kp *= self.gain_scale
        kd *= self.gain_scale
        limit *= self.gain_scale
        self.limit=limit
        self.pd=StablePDController(np.ones(m.nu),np.zeros(m.nu),m.nv,limit,kp,kd)
        self.jac=np.zeros((3,m.nv))
        self.last_correction=np.zeros(2)

    def control(self,d,q,balance=True):
        target=q[self.mapping.actuator_qpos].copy()
        correction=np.zeros(2)
        if balance:
            feet=np.mean([d.xpos[self.m.body(n).id] for n in ('L_Ankle','R_Ankle')],axis=0)
            error=d.subtree_com[1,:2]-feet[:2]-[.02,0]
            mujoco.mj_jacSubtreeCom(self.m,d,self.jac,1)
            velocity=(self.jac@d.qvel)[:2]
            correction=np.clip(8*error+2*velocity,-.6,.6)
            for side in ('L','R'):
                target[self.index[side+'_Ankle_x']]+=correction[0]
                target[self.index[side+'_Ankle_z']]-=correction[1]
                target[self.index[side+'_Hip_z']]+=0.5*correction[1]
        self.last_correction=correction
        # The shared reference root is diagnostic only: no root actuator/force.
        return self.pd.control(target,self.m,d),target


def contacts(m,d):
    force={'L':0.,'R':0.}
    contact_geoms=set()
    nonfoot=0
    floor=m.geom('floor').id
    wrench=np.zeros(6)
    for i in range(d.ncon):
        c=d.contact[i]
        if floor not in (c.geom1,c.geom2):continue
        gid=c.geom2 if c.geom1==floor else c.geom1
        name=m.body(int(m.geom_bodyid[gid])).name
        side=next((s for s in ('L','R') if name in (s+'_Ankle',s+'_Toe')),None)
        mujoco.mj_contactForce(m,d,i,wrench)
        if side is not None and wrench[0]>.1:
            force[side]+=float(wrench[0]);contact_geoms.add(gid)
        elif side is None and wrench[0]>.1:nonfoot+=1
    points=[]
    for gid in contact_geoms:
        if m.geom_type[gid]!=mujoco.mjtGeom.mjGEOM_BOX:continue
        # Foot geometry corners project to a conservative support footprint.
        for x in (-1,1):
            for y in (-1,1):
                for z in (-1,1):
                    points.append((d.geom_xpos[gid]+d.geom_xmat[gid].reshape(3,3)@(m.geom_size[gid]*[x,y,z]))[:2])
    margin=-1.
    if len(points)>=3:
        hull=ConvexHull(np.unique(np.round(points,8),axis=0))
        e=hull.equations
        margin=float(np.min(-(e[:,:2]@d.subtree_com[1,:2]+e[:,2])/np.linalg.norm(e[:,:2],axis=1)))
    return force,margin,nonfoot


def joint_table(m,body,out):
    index={n:i for i,n in enumerate(SMPLH_BONE_ORDER_NAMES)}
    rows=[]
    for j in range(m.njnt):
        b=m.body(int(m.jnt_bodyid[j])).name
        aids=np.flatnonzero(m.actuator_trnid[:,0]==j).tolist()
        rows.append({'shape':body,'joint':m.joint(j).name,'body':b,'smpl_joint_index':index[b],
                     'type':'free' if m.jnt_type[j]==0 else 'hinge','qpos_address':int(m.jnt_qposadr[j]),
                     'dof_address':int(m.jnt_dofadr[j]),'axis':json.dumps(m.jnt_axis[j].tolist()),
                     'actuator_ids':json.dumps(aids),'actuator_names':json.dumps([m.actuator(i).name for i in aids])})
    write_csv(out/f'{body}_joints.csv',rows)


def write_csv(path,rows):
    if not rows:raise ValueError('No samples collected')
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run(body,kind,args):
    m=mujoco.MjModel.from_xml_path(str(args.models/f'{body}.xml'));m.opt.timestep=1/450
    mapping=SMPLHMapping(m)
    motion=SMPLHMotion.load(args.motion_file) if args.motion_file else reference(kind,m)
    bound=motion.bind(m)
    if args.align_floor:
        probe=mujoco.MjData(m)
        probe.qpos[:]=bound.sample(0)
        z_before=float(probe.qpos[2])
        ground(m,probe)
        offset=np.array([0.,0.,probe.qpos[2]-z_before])
        if motion.coordinate_system=='smpl':offset=SMPL_TO_WORLD.inv().apply(offset)
        motion=SMPLHMotion(motion.poses,motion.root_position+offset,motion.fps,
                           motion.coordinate_system,motion.translation_kind)
        bound=motion.bind(m)
    label=getattr(args, 'motion_label', None) or ('imported' if args.motion_file else kind)
    motion.save(args.output/f'{body}_{label}_reference.npz')
    joint_table(m,body,args.output)
    d=mujoco.MjData(m);d.qpos[:]=bound.sample(0);mujoco.mj_forward(m,d)
    controller=BalancedPD(m,mapping)
    rows=[];joint_rows=[]
    init_z=float(d.qpos[2])
    ankles=[m.body(n).id for n in ('L_Ankle','R_Ankle')]
    feet0=d.xpos[ankles,:2].copy()
    # root local Y is the SMPL vertical axis.
    body_mask=np.array([not any(k in m.joint(int(j)).name for k in FINGERS) for j in mapping.actuator_joint_ids])
    task_names = ([side+'_Shoulder_x' for side in ('L','R')] if kind == 'arms'
                  else [side+'_Knee_x' for side in ('L','R')] if kind == 'knees'
                  else [m.joint(int(j)).name for j in mapping.actuator_joint_ids if
                        not any(k in m.joint(int(j)).name for k in FINGERS)])
    task_names = getattr(args, "tracking_joint_names", None) or task_names
    task_indices = [controller.index[n] for n in task_names]
    context=mujoco.viewer.launch_passive(m,d) if args.gui else nullcontext(None)
    with context as viewer:
        if viewer:
            with viewer.lock():
                viewer.cam.lookat[:]=[0,0,1];viewer.cam.distance=3;viewer.cam.azimuth=135;viewer.cam.elevation=-15
        frames=round(motion.duration*30)
        if args.frames is not None:frames=min(frames,args.frames)
        for frame in range(frames):
            if viewer and not viewer.is_running():break
            start=time.monotonic()
            lock=viewer.lock() if viewer else nullcontext()
            with lock:
                for _ in range(15):
                    desired=bound.sample(d.time+m.opt.timestep)
                    torque,effective=controller.control(d,desired,not args.no_balance)
                    d.ctrl[:]=torque
                    mujoco.mj_step(m,d)
                    if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                        raise RuntimeError('Nonfinite simulation state')
                mujoco.mj_forward(m,d)
                ref=bound.sample(d.time)
                err=d.qpos[mapping.actuator_qpos]-ref[mapping.actuator_qpos]
                force,margin,nonfoot=contacts(m,d)
                tilt=float(np.degrees(np.arccos(np.clip(d.xmat[1].reshape(3,3)[2,1],-1,1))))
                fall=d.qpos[2]<.65*init_z or tilt>35 or nonfoot>0
                balanced=not fall and tilt<25 and (force['L']>1 or force['R']>1) and margin>=-.02
                row={'shape':body,'motion':label,'time_s':float(d.time),
                     'joint_rmse_rad':float(np.sqrt(np.mean(err**2))),
                     'body_joint_rmse_rad':float(np.sqrt(np.mean(err[body_mask]**2))),
                     'motion_joint_rmse_rad':float(np.sqrt(np.mean(err[task_indices]**2))),
                     'effective_target_rmse_rad':float(np.sqrt(np.mean((d.qpos[mapping.actuator_qpos]-effective)**2))),
                     'root_z_m':float(d.qpos[2]),'root_target_z_error_m':float(d.qpos[2]-ref[2]),
                     'pelvis_tilt_deg':tilt,'com_x_m':float(d.subtree_com[1,0]),'com_y_m':float(d.subtree_com[1,1]),
                     'support_margin_m':margin,'left_normal_force_N':force['L'],'right_normal_force_N':force['R'],
                     'left_contact':int(force['L']>1),'right_contact':int(force['R']>1),
                     'foot_drift_m':float(np.max(np.linalg.norm(d.xpos[ankles,:2]-feet0,axis=1))),
                     'left_wrist_z_m':float(d.xpos[m.body('L_Wrist').id,2]),
                     'right_wrist_z_m':float(d.xpos[m.body('R_Wrist').id,2]),
                     'left_knee_x_rad':float(d.qpos[m.joint('L_Knee_x').qposadr[0]]),
                     'ankle_pitch_correction_rad':float(controller.last_correction[0]),
                     'ankle_roll_correction_rad':float(controller.last_correction[1]),
                     'torque_saturation_fraction':float(np.mean(np.abs(torque)>=controller.limit-1e-7)),
                     'nonfoot_floor_contacts':nonfoot,'fallen':int(fall),'balanced':int(balanced)}
                rows.append(row)
                if frame % getattr(args, "joint_log_stride", 3) == 0:
                    for i,j in enumerate(mapping.actuator_joint_ids):
                        joint_rows.append({'shape':body,'motion':label,'time_s':float(d.time),'joint':m.joint(int(j)).name,
                                           'target_rad':float(ref[mapping.actuator_qpos[i]]),
                                           'effective_target_rad':float(effective[i]),'actual_rad':float(d.qpos[mapping.actuator_qpos[i]]),
                                           'error_rad':float(err[i]),'torque_Nm':float(torque[i])})
            if viewer:
                viewer.sync();time.sleep(max(0,1/30-(time.monotonic()-start)))
    if d.warning.number.any():raise RuntimeError(f'MuJoCo warnings: {d.warning.number}')
    write_csv(args.output/f'{body}_{label}_timeseries.csv',rows)
    write_csv(args.output/f'{body}_{label}_joint_tracking.csv',joint_rows)
    summary_start = getattr(args, 'summary_start_time', 1.)
    measured=[r for r in rows if r['time_s']>=summary_start] or rows
    summary={'shape':body,'motion':label,'duration_s':float(d.time),
             'body_joint_rmse_rad':float(np.sqrt(np.mean([r['body_joint_rmse_rad']**2 for r in measured]))),
             'joint_rmse_rad':float(np.sqrt(np.mean([r['joint_rmse_rad']**2 for r in measured]))),
             'motion_joint_rmse_rad':float(np.sqrt(np.mean([r['motion_joint_rmse_rad']**2 for r in measured]))),
             'balance_fraction':float(np.mean([r['balanced'] for r in measured])),
             'both_feet_contact_fraction':float(np.mean([r['left_contact'] and r['right_contact'] for r in measured])),
             'max_pelvis_tilt_deg':max(r['pelvis_tilt_deg'] for r in measured),
             'min_root_height_m':min(r['root_z_m'] for r in measured),
             'max_foot_drift_m':max(r['foot_drift_m'] for r in measured),
             'wrist_height_gain_m':max((r['left_wrist_z_m']+r['right_wrist_z_m'])/2 for r in measured)-(rows[0]['left_wrist_z_m']+rows[0]['right_wrist_z_m'])/2,
             'max_knee_angle_rad':max(r['left_knee_x_rad'] for r in measured),
             'fallen_any':int(any(r['fallen'] for r in measured)),
             'gain_scale':controller.gain_scale,'mujoco_warnings':int(d.warning.number.sum()),'balance_feedback':int(not args.no_balance)}
    write_csv(args.output/f'{body}_{label}_summary.csv',[summary])
    print(json.dumps(summary),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--body',choices=SHAPES,default='body_01')
    p.add_argument('--all',action='store_true')
    p.add_argument('--motion',choices=[*DURATIONS,'all'],default='all')
    p.add_argument('--gui',action='store_true')
    p.add_argument('--frames',type=int)
    p.add_argument('--no-balance',action='store_true')
    p.add_argument('--check',action='store_true',help='Require complete builtin motions to meet balance/tracking criteria')
    p.add_argument('--motion-file',type=Path)
    p.add_argument('--align-floor',action='store_true',help='Translate the complete root trajectory so the initial collision pose clears the floor')
    p.add_argument('--models',type=Path,default=ROOT/'outputs/smplh_physics')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/m3_control')
    args=p.parse_args()
    if args.frames is not None and args.frames<1:p.error('--frames must be positive')
    if args.check and (args.frames or args.motion_file or args.gui):
        p.error('--check requires complete headless built-in motion tests')
    args.output.mkdir(parents=True,exist_ok=True)
    bodies=list(SHAPES) if args.all else [args.body]
    motions=list(DURATIONS) if args.motion=='all' else [args.motion]
    if args.motion_file:motions=['stand']
    if args.gui and (len(bodies)*len(motions)>1):
        launcher=Path(sys.prefix)/'bin'/('mjpython' if sys.platform=='darwin' else 'python')
        gui_summaries=[]
        for body in bodies:
            for motion in motions:
                cmd=[str(launcher),str(Path(__file__).resolve()),'--body',body,'--motion',motion,'--gui',
                     '--models',str(args.models.resolve()),'--output',str(args.output.resolve())]
                if args.frames:cmd+=['--frames',str(args.frames)]
                if args.no_balance:cmd+=['--no-balance']
                if args.align_floor:cmd+=['--align-floor']
                if args.motion_file:cmd+=['--motion-file',str(args.motion_file.resolve())]
                subprocess.run(cmd,check=True)
                label='imported' if args.motion_file else motion
                with (args.output/f'{body}_{label}_summary.csv').open() as f:
                    gui_summaries.extend(csv.DictReader(f))
        write_csv(args.output/'summary.csv',gui_summaries)
        return
    summaries=[run(b,m,args) for b in bodies for m in motions]
    write_csv(args.output/'summary.csv',summaries)
    if args.check:
        for s in summaries:
            if s['fallen_any'] or s['balance_fraction'] < .95 or s['both_feet_contact_fraction'] < .95 or s['body_joint_rmse_rad'] > .08:
                raise RuntimeError(f"Balance or tracking check failed: {s}")
            if s['motion']=='arms' and (s['wrist_height_gain_m'] < .15 or s['motion_joint_rmse_rad'] > .1):
                raise RuntimeError(f"Arm raise check failed: {s}")
            if s['motion']=='knees' and (s['max_knee_angle_rad'] < .4 or s['motion_joint_rmse_rad'] > .15):
                raise RuntimeError(f"Knee bend check failed: {s}")
        print(f"All {len(summaries)} motion checks passed",flush=True)


if __name__=='__main__':main()
