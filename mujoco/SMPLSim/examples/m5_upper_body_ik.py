"""M5-1 bounded wrist-pose IK and continuous upper-body reference candidates.
FK probe writes never affect the live physical ball. IK is not a physics gate.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.m5_residual_ppo_env import BasketballResidualEnv, BASE, OUT as M5_OUT
from examples.m5_bimanual_contact import OUT, derived_scene, gap_to_ball
from examples.m4_contact_lifting import ARM
from examples.m3_control import write_csv, FINGERS
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
import mujoco
import numpy as np


def minimum_jerk(x):
    s = np.clip(x, 0, 1)
    return s**3*(10-15*s+6*s*s)


def joint_ids(m, names):
    return np.array([int(m.joint(n).qposadr[0]) for n in names])


def rotation_error(a, b):
    return Rotation.from_matrix(a.T @ b).as_rotvec()


def fit_wrist(m, q, side, position, orientation, anchor=None, regularization=.02):
    names = [n.replace('R_', side+'_', 1) for n in ARM]
    ids = joint_ids(m, names)
    bounds = np.array([m.jnt_range[m.joint(n).id] for n in names])
    probe = mujoco.MjData(m)
    start = q[ids].copy() if anchor is None else np.asarray(anchor).copy()
    lo, hi = bounds[:, 0]+1e-5, bounds[:, 1]-1e-5
    def residual(x):
        probe.qpos[:] = q
        probe.qpos[ids] = x
        mujoco.mj_forward(m, probe)
        bid = m.body(side+'_Wrist').id
        return np.r_[30*(probe.xpos[bid]-position),
                     2*rotation_error(orientation, probe.xmat[bid].reshape(3, 3)),
                     regularization*(x-start)]
    fit = least_squares(residual, np.clip(start, lo, hi), bounds=(lo, hi), max_nfev=100)
    result = q.copy()
    result[ids] = fit.x
    probe.qpos[:] = result
    mujoco.mj_forward(m, probe)
    b = m.body(side+'_Wrist').id
    report = dict(position_error_mm=float(np.linalg.norm(probe.xpos[b]-position)*1000),
                  orientation_error_deg=float(np.degrees(np.linalg.norm(rotation_error(orientation, probe.xmat[b].reshape(3, 3))))))
    return result, report


def anatomical_mapping(m, d, side):
    # Match measured world axes to geometric flexion/abduction/twist directions.
    normal = -d.xmat[m.body(side+'_Wrist').id].reshape(3, 3)[:, 1]
    rows = []
    for f in FINGERS:
        for part in (1, 2, 3):
            name = f'{side}_{f}{part}'
            b = m.body(name).id
            from examples.m4_hand_kinematics import capsule
            a, z, _ = capsule(m, d, m.geom(name).id)
            bone = z-a
            if np.dot(bone, (a+z)/2-d.xpos[b]) < 0:
                bone = -bone
            bone /= np.linalg.norm(bone)
            flex = np.cross(bone, normal)
            flex /= np.linalg.norm(flex)
            ideals = np.array([flex, np.cross(flex, bone), bone])
            axes = np.array([d.xaxis[m.joint(name+'_'+ax).id] for ax in 'xyz'])
            scores = abs(axes @ ideals.T)
            permutation = max(itertools.permutations(range(3)), key=lambda p:sum(scores[p[k], k] for k in range(3)))
            for k, role in enumerate(('flexion', 'abduction', 'twist')):
                axis = permutation[k]
                rows.append(dict(joint=name+'_'+'xyz'[axis], role=role, sign=1 if np.dot(axes[axis], ideals[k])>=0 else -1, alignment=float(scores[axis,k])))
    return rows


def generate(gap=.001, flexion=0., name=None):
    name = name or f'guide_gap_{gap:+.3f}_flex_{flexion:.2f}'
    out = OUT/'references'/name
    out.mkdir(parents=True, exist_ok=True)
    env = BasketballResidualEnv(use_snapshot=True)
    env.load_snapshot(M5_OUT/'reset_snapshot.npz')
    env.reset(seed=7)
    q = env.d.qpos.copy()
    base = env.adapter.target(0., env.d)
    q[7:env.bridge.model.nq] = base[7:] # free root is actual settled state, not a commanded pose
    m = mujoco.MjModel.from_xml_path(str(derived_scene().resolve()))
    d = mujoco.MjData(m)
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    ball = env.d.qpos[env.ball_q:env.ball_q+3].copy()
    RW = np.load(BASE/'ik_path.npz')['rotation']
    left_R = np.column_stack(([0,0,1.], [0,1.,0], [-1.,0,0]))
    lw = m.body('L_Wrist').id
    # Desired capsule center on ball side; convert center to wrist origin.
    offset = d.xmat[lw].reshape(3,3).T @ (d.geom_xpos[m.geom('L_Wrist').id]-d.xpos[lw])
    left_origin = ball+np.array([0,.121+float(m.geom('L_Wrist').size[0])+gap,0])-left_R@offset
    approached, guide_fit = fit_wrist(m, q, 'L', left_origin, left_R)
    d.qpos[:] = approached
    mujoco.mj_forward(m, d)
    mapping = anatomical_mapping(m, d, 'L')
    for r in mapping:
        if r['role']=='flexion' and 'Thumb' not in r['joint']:
            approached[int(m.joint(r['joint']).qposadr[0])] = flexion*r['sign']
    # Geometric axis effects for torso and head: label by measured world axis.
    axis_rows = []
    for body in ('Torso','Spine','Chest','Neck','Head'):
        for axis in 'xyz':
            jid = m.joint(body+'_'+axis).id
            world = d.xaxis[jid]
            axis_rows.append(dict(joint=body+'_'+axis,world_x=world[0],world_y=world[1],world_z=world[2]))
    write_csv(out/'upper_body_axes.csv', axis_rows)
    (out/'left_axis_mapping.json').write_text(json.dumps(mapping,indent=2))
    # Fit head local +Z forward to the actual rim direction; retain small trunk posture.
    from smpl_sim.physics.basketball import RIM_CENTER
    trunk_ids = joint_ids(m,[f'{b}_{a}' for b in ('Torso','Spine','Chest') for a in 'xyz'])
    head_ids = joint_ids(m,[f'{b}_{a}' for b in ('Neck','Head') for a in 'xyz'])
    trunk_target = approached[trunk_ids]*.8  # moderate neutralization, not forced vertical
    aligned = approached.copy()
    aligned[trunk_ids] = trunk_target
    probe = mujoco.MjData(m)
    prior = aligned[head_ids].copy()
    def head_residual(x):
        probe.qpos[:] = aligned
        probe.qpos[head_ids] = x
        mujoco.mj_forward(m,probe)
        head = m.body('Head').id
        direction = RIM_CENTER-probe.xpos[head]
        direction /= np.linalg.norm(direction)
        return np.r_[3*(probe.xmat[head].reshape(3,3)[:,2]-direction),.05*(x-prior)]
    hfit=least_squares(head_residual,prior,bounds=(prior-.65,prior+.65),max_nfev=80)
    aligned[head_ids]=hfit.x
    # Compensate both wrist poses for the changed trunk in FK.
    rpos = d.xpos[m.body('R_Wrist').id].copy()
    aligned,rfit=fit_wrist(m,aligned,'R',rpos,RW)
    aligned,lfit=fit_wrist(m,aligned,'L',left_origin,left_R)
    # Use settled physical pelvis at the existing lifting endpoint for later IK.
    endpoint=np.load(BASE/'C/states.npz')['qpos'][-1].copy()
    endpoint[7:env.bridge.model.nq] = env.adapter.target(6.5,env.d)[7:]
    endpoint[trunk_ids] = aligned[trunk_ids]
    endpoint[head_ids] = aligned[head_ids]
    endpoint[joint_ids(m,[r['joint'] for r in mapping])] = aligned[joint_ids(m,[r['joint'] for r in mapping])]
    pd = mujoco.MjData(m)
    pd.qpos[:] = endpoint
    mujoco.mj_forward(m,pd)
    root_R=pd.xmat[m.body('Pelvis').id].reshape(3,3)
    forward=root_R[:,2].copy();forward[2]=0;forward/=np.linalg.norm(forward)
    head=pd.xpos[m.body('Head').id]
    final=head+.20*forward+[0,0,.20]
    initial_goal=env.goal_position()
    intermediate=np.array([final[0]+.04,ball[1],head[2]+.02])
    rel0=env.rel0.copy()
    path=[];reports=[];previous=endpoint.copy()
    for fraction in np.linspace(0,1,61):
        # Smooth spatial path via P1/P2/P3; time parameter handled by target adapter.
        center=(1-fraction)*initial_goal+fraction*final
        if fraction<.5:center=(1-2*fraction)*initial_goal+2*fraction*intermediate
        else:center=(2-2*fraction)*intermediate+(2*fraction-1)*final
        candidate,rreport=fit_wrist(m,previous,'R',center-RW@rel0,RW)
        candidate,lreport=fit_wrist(m,candidate,'L',center+[0,.121+float(m.geom('L_Wrist').size[0])+gap,0]-left_R@offset,left_R)
        pd.qpos[:]=candidate;mujoco.mj_forward(m,pd)
        reports.append(dict(fraction=float(fraction),**{'right_'+k:v for k,v in rreport.items()},**{'left_'+k:v for k,v in lreport.items()},head_ball_gap_m=gap_to_ball(m,pd,'Head',center),goal_x_m=center[0],goal_y_m=center[1],goal_z_m=center[2]))
        path.append(candidate.copy());previous=candidate
    np.savez_compressed(out/'reference.npz',initial=q,approached=approached,aligned=aligned,
                        shooting_path=path,right_rotation=RW,left_rotation=left_R,left_offset=offset,
                        ball_initial=ball,ball_goal_lift=initial_goal,ball_goal_shoot=final,ball_goal_intermediate=intermediate,
                        rel0=rel0,trunk_ids=trunk_ids,head_ids=head_ids,left_origin=left_origin)
    write_csv(out/'shooting_ik.csv',reports)
    summary=dict(name=name,gap_m=gap,flexion_rad=flexion,guide_ik=guide_fit,alignment_right_ik=rfit,alignment_left_ik=lfit,
                 max_shooting_right_ik_mm=max(r['right_position_error_mm'] for r in reports),max_shooting_left_ik_mm=max(r['left_position_error_mm'] for r in reports),
                 final_ball_goal_m=final.tolist(),ik_is_physics_success=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));env.close();print(json.dumps(summary),flush=True)
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--gap',type=float,default=.001);p.add_argument('--flexion',type=float,default=0.);p.add_argument('--name');a=p.parse_args();generate(a.gap,a.flexion,a.name)
if __name__=='__main__':main()
