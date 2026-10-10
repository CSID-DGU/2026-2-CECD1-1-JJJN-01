"""M3-1B-4: body_01 physical diagnostics, unchanged M3 PD + COM controller."""
import argparse
import hashlib
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import mujoco.viewer
import numpy as np
from examples.m3_control import BalancedPD, FINGERS, contacts, ground, write_csv
from smpl_sim.motion.smplh_motion import SMPLHMotion
from smpl_sim.motion.physics_metrics import foot_contact_slip, quaternion_error_deg


def skeleton(scene, model, reference):
    """Green kinematic reference; never added to collision/dynamic model."""
    for bid in range(2, model.nbody):
        if any(k in model.body(bid).name for k in FINGERS):
            continue
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                           np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                           np.array([.05, 1., .2, .8], dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .009,
                             reference.xpos[int(model.body_parentid[bid])], reference.xpos[bid])
        scene.ngeom += 1


def plots(output, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = [r['time_s'] for r in rows]
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    for key in ('body_joint_rmse_rad', 'effective_target_rmse_rad'):
        axes[0].plot(t, [r[key] for r in rows], label=key)
    axes[0].set_ylabel('angle error (rad)')
    axes[1].plot(t, [r['root_position_error_m'] for r in rows], label='root position (m)')
    axes[1].plot(t, [r['root_orientation_error_deg']/100 for r in rows], label='orientation (deg / 100)')
    axes[2].plot(t, [r['support_margin_m'] if r['any_contact'] else np.nan for r in rows], label='COM support margin (m)')
    axes[2].axhline(0, color='k', linewidth=.5)
    for side in ('L', 'R'):
        axes[3].plot(t, [r[f'{side}_normal_force_N'] for r in rows], label=side+' force (N)')
        axes[4].plot(t, [r[f'{side}_contact_rms_speed_m_s'] for r in rows], label=side+' slip (m/s)')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    axes[-1].set_xlabel('simulation time (s)')
    fig.tight_layout()
    fig.savefig(output/'physics_diagnostics.png', dpi=150)
    plt.close(fig)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    model = mujoco.MjModel.from_xml_path(str(args.model))
    model.opt.timestep = 1/450
    motion = SMPLHMotion.load(args.source)
    bound = motion.bind(model)
    mapping = bound.mapping
    names = [model.joint(int(j)).name for j in mapping.actuator_joint_ids]
    body_mask = np.array([not any(k in n for k in FINGERS) for n in names])
    probe = mujoco.MjData(model)
    probe.qpos[:] = bound.sample(0)
    original_z = float(probe.qpos[2])
    ground(model, probe)
    floor_offset = float(probe.qpos[2] - original_z)
    offset = floor_offset if args.root_mode == 'aligned' else 0.

    def target(t):
        q = bound.sample(t)  # returns a copy; fitting arrays are untouched
        q[2] += offset
        return q

    data, ref = mujoco.MjData(model), mujoco.MjData(model)
    data.qpos[:] = target(0)
    mujoco.mj_forward(model, data)
    init_z = float(data.qpos[2])
    foot_names = ['L_Ankle', 'L_Toe', 'R_Ankle', 'R_Toe']
    foot_ids = [model.body(n).id for n in foot_names]
    foot_start = data.xpos[foot_ids].copy()
    controller = BalancedPD(model, mapping)
    rows, joint_rows, contact_rows, reference_rows = [], [], [], []
    joint_errors, effective_errors, qpos_states, qvel_states = [], [], [], []
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0, 0, 1]
    camera.distance, camera.azimuth, camera.elevation = 3., 135., -15.
    renderer = None
    video_frames = []
    if args.video:
        renderer = mujoco.Renderer(model, height=480, width=640)
    context = mujoco.viewer.launch_passive(model, data) if args.gui else nullcontext(None)
    first_fall = None
    warning_first = {}
    incomplete_reason = None
    slip_path = {'L': 0., 'R': 0.}
    sample_steps = set(np.ceil(np.arange(len(motion.poses))/motion.fps/model.opt.timestep-1e-9).astype(int))
    total_steps = max(sample_steps)
    wall_start = time.monotonic()
    with context as viewer:
        if viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = camera.lookat
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3., 135., -15.
        for step in range(total_steps+1):
            if viewer and not viewer.is_running():
                incomplete_reason = 'GUI closed before motion end'
                break
            with viewer.lock() if viewer else nullcontext():
                desired = target(data.time + (model.opt.timestep if step else 0))
                torque, effective = controller.control(data, desired)
                if step:
                    before = float(data.time)
                    data.ctrl[:] = torque
                    mujoco.mj_step(model, data)
                    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or data.time <= before:
                        incomplete_reason = 'Nonfinite state or MuJoCo time reset'
                        break
                mujoco.mj_forward(model, data)
                desired = target(data.time)
                ref.qpos[:] = desired
                mujoco.mj_forward(model, ref)
                err = data.qpos[mapping.actuator_qpos] - desired[mapping.actuator_qpos]
                eff_err = data.qpos[mapping.actuator_qpos] - effective
                forces, margin, nonfoot = contacts(model, data)
                slip, point_rows = foot_contact_slip(model, data)
                tilt = float(np.degrees(np.arccos(np.clip(data.xmat[1].reshape(3,3)[2,1], -1, 1))))
                height_fall, tilt_fall = data.qpos[2] < .65*init_z, tilt > 35
                fallen = bool(height_fall or tilt_fall or nonfoot)
                if fallen and first_fall is None:
                    first_fall = float(data.time)
                any_contact = forces['L'] > 1 or forces['R'] > 1
                balanced = not fallen and tilt < 25 and any_contact and margin >= -.02
                root_error = data.qpos[:3] - desired[:3]
                row = {'time_s': float(data.time), 'body_joint_rmse_rad': float(np.sqrt(np.mean(err[body_mask]**2))),
                       'effective_target_rmse_rad': float(np.sqrt(np.mean(eff_err[body_mask]**2))),
                       'root_position_error_m': float(np.linalg.norm(root_error)),
                       'original_source_root_error_m': float(np.linalg.norm(data.qpos[:3]-bound.sample(data.time)[:3])),
                       'root_x_error_m': float(root_error[0]), 'root_y_error_m': float(root_error[1]),
                       'root_z_error_m': float(root_error[2]),
                       'root_orientation_error_deg': quaternion_error_deg(data.qpos[3:7], desired[3:7]),
                       'root_x_m': float(data.qpos[0]), 'root_y_m': float(data.qpos[1]), 'root_z_m': float(data.qpos[2]),
                       'com_x_m': float(data.subtree_com[1,0]), 'com_y_m': float(data.subtree_com[1,1]),
                       'com_z_m': float(data.subtree_com[1,2]), 'support_margin_m': margin,
                       'any_contact': int(any_contact), 'both_contact': int(forces['L']>1 and forces['R']>1),
                       'pelvis_tilt_deg': tilt, 'nonfoot_contacts': nonfoot, 'fallen': int(fallen),
                       'height_fall': int(height_fall), 'tilt_fall': int(tilt_fall), 'balanced': int(balanced),
                       'com_correction_x_rad': float(controller.last_correction[0]),
                       'com_correction_y_rad': float(controller.last_correction[1])}
                for side in ('L', 'R'):
                    row.update({side+'_normal_force_N': forces[side], side+'_contact': int(forces[side]>1),
                                side+'_contact_rms_speed_m_s': float(slip[side]['rms_speed']),
                                side+'_contact_max_speed_m_s': slip[side]['max_speed']})
                    if step:
                        slip_path[side] += slip[side]['mean_speed']*model.opt.timestep
                for i, name in enumerate(foot_names):
                    delta = data.xpos[foot_ids[i]] - foot_start[i]
                    row[name+'_movement_xy_m'] = float(np.linalg.norm(delta[:2]))
                    row[name+'_movement_3d_m'] = float(np.linalg.norm(delta))
                rows.append(row)
                joint_errors.append(err.copy())
                effective_errors.append(eff_err.copy())
                for i, w in enumerate(data.warning):
                    if w.number and str(i) not in warning_first:
                        warning_first[str(i)] = float(data.time)
                if step in sample_steps:
                    qpos_states.append(data.qpos.copy())
                    qvel_states.append(data.qvel.copy())
                    contact_rows.extend(point_rows)
                    reference_rows.append({'time_s': float(data.time),
                                           'reference_root_z_m': float(desired[2]),
                                           'reference_com_x_m': float(ref.subtree_com[1,0]),
                                           'reference_com_y_m': float(ref.subtree_com[1,1]),
                                           'reference_min_foot_collision_z_m': min(
                                               float(ref.geom_xpos[g,2]-np.abs(ref.geom_xmat[g].reshape(3,3)[2])@model.geom_size[g])
                                               for g in range(model.ngeom)
                                               if model.geom_bodyid[g] in foot_ids and model.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX),
                                           **{name+'_reference_z_m': float(ref.xpos[bid,2]) for name,bid in zip(foot_names,foot_ids)}})
                    for i, name in enumerate(names):
                        joint_rows.append({'time_s': float(data.time), 'joint': name,
                                           'target_rad': float(desired[mapping.actuator_qpos[i]]),
                                           'actual_rad': float(data.qpos[mapping.actuator_qpos[i]]),
                                           'error_rad': float(err[i]), 'effective_target_rad': float(effective[i]),
                                           'torque_Nm': float(data.ctrl[i])})
                    if viewer:
                        viewer.user_scn.ngeom = 0
                        skeleton(viewer.user_scn, model, ref)
                    if renderer and (len(qpos_states)-1)%2 == 0:
                        renderer.update_scene(data, camera=camera)
                        skeleton(renderer.scene, model, ref)
                        from PIL import Image
                        video_frames.append(Image.fromarray(renderer.render()).quantize(colors=256))
            if viewer and step in sample_steps:
                viewer.sync()
                time.sleep(max(0, float(data.time)-(time.monotonic()-wall_start)))
    if renderer:
        renderer.close()
        if video_frames:
            # GIF has 10 ms timing precision: alternating 30/30/40 ms gives 30 FPS.
            video_frames[0].save(args.output/'reference_vs_physics.gif', save_all=True,
                                 append_images=video_frames[1:], loop=0,
                                 duration=[30 if i%3<2 else 40 for i in range(len(video_frames))],
                                 optimize=False, disposal=2)
    errors = np.array(joint_errors)
    prefix = args.output
    write_csv(prefix/'timeseries_450hz.csv', rows)
    write_csv(prefix/'timeseries_60hz.csv', [rows[s] for s in sorted(sample_steps) if s<len(rows)])
    write_csv(prefix/'joint_tracking_60hz.csv', joint_rows)
    write_csv(prefix/'reference_kinematics_60hz.csv', reference_rows)
    if contact_rows:
        write_csv(prefix/'foot_contact_points_60hz.csv', contact_rows)
    write_csv(prefix/'joint_summary.csv', [{'joint': n, 'is_body': int(body_mask[i]),
              'rmse_rad': float(np.sqrt(np.mean(errors[:,i]**2))),
              'max_abs_error_rad': float(np.max(np.abs(errors[:,i])))} for i,n in enumerate(names)])
    supported = [r['support_margin_m'] for r in rows if r['any_contact']]
    summary = {'body': getattr(args, 'body', 'body_01'), 'root_mode': args.root_mode, 'source_sha256': source_hash,
               'source_fps': motion.fps, 'source_frames': len(motion.poses), 'source_duration_s': motion.duration,
               'simulation_duration_s': float(data.time), 'physics_steps': len(rows)-1,
               'metric_samples_450hz': len(rows), 'metric_samples_60hz': len(qpos_states),
               'floor_offset_applied_m': offset, 'floor_offset_needed_m': floor_offset,
               'initial_original_root_z_m': original_z, 'initial_simulation_root_z_m': init_z,
               'controller_gain_scale': controller.gain_scale, 'mujoco_version': mujoco.__version__,
               'body_joint_rmse_rad': float(np.sqrt(np.mean(errors[:,body_mask]**2))),
               'body_effective_target_rmse_rad': float(np.sqrt(np.mean(np.array(effective_errors)[:,body_mask]**2))),
               'root_position_rmse_m': float(np.sqrt(np.mean([r['root_position_error_m']**2 for r in rows]))),
               'original_source_root_position_rmse_m': float(np.sqrt(np.mean([r['original_source_root_error_m']**2 for r in rows]))),
               'root_orientation_rmse_deg': float(np.sqrt(np.mean([r['root_orientation_error_deg']**2 for r in rows]))),
               'balance_fraction': float(np.mean([r['balanced'] for r in rows])),
               'both_feet_contact_fraction': float(np.mean([r['both_contact'] for r in rows])),
               'no_foot_contact_fraction': float(np.mean([not r['any_contact'] for r in rows])),
               'min_supported_com_margin_m': min(supported) if supported else None,
               'supported_com_inside_fraction': float(np.mean([r['support_margin_m']>=0 for r in rows if r['any_contact']])) if supported else None,
               'min_supported_com_margin_after_100ms_m': min((r['support_margin_m'] for r in rows if r['any_contact'] and r['time_s']>=.1), default=None),
               'root_position_max_error_m': max(r['root_position_error_m'] for r in rows),
               'max_pelvis_tilt_deg': max(r['pelvis_tilt_deg'] for r in rows),
               'fall_count': int(first_fall is not None), 'first_fall_time_s': first_fall,
               'incomplete_reason': incomplete_reason,
               'warnings': {mujoco.mjtWarning(i).name: {'count': int(w.number), 'first_time_s': warning_first.get(str(i))}
                            for i,w in enumerate(data.warning)},
               'finite_final_state': bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())}
    for axis in ('x','y','z'):
        summary['root_'+axis+'_rmse_m'] = float(np.sqrt(np.mean([r['root_'+axis+'_error_m']**2 for r in rows])))
    for side in ('L','R'):
        arm_indices = [i for i,n in enumerate(names) if n.startswith(side+'_') and any(k in n for k in ('Shoulder','Elbow','Wrist'))]
        summary[side+'_arm_joint_rmse_rad'] = float(np.sqrt(np.mean(errors[:,arm_indices]**2)))
        loaded = [r for r in rows if r[side+'_contact']]
        summary[side+'_contact_fraction'] = float(np.mean([r[side+'_contact'] for r in rows]))
        summary[side+'_loaded_contact_slip_rms_m_s'] = float(np.sqrt(np.mean([r[side+'_contact_rms_speed_m_s']**2 for r in loaded]))) if loaded else None
        summary[side+'_max_contact_slip_m_s'] = max(r[side+'_contact_max_speed_m_s'] for r in rows)
        summary[side+'_loaded_slip_over_1cm_s_fraction'] = float(np.mean([r[side+'_contact_rms_speed_m_s']>.01 for r in loaded])) if loaded else None
        summary[side+'_integrated_contact_slip_path_m'] = slip_path[side]
    for name in foot_names:
        for kind in ('xy','3d'):
            summary[name+'_max_movement_'+kind+'_m'] = max(r[name+'_movement_'+kind+'_m'] for r in rows)
    prefall = [r for r in rows if first_fall is None or r['time_s']<first_fall]
    summary['prefall_body_joint_rmse_rad'] = float(np.sqrt(np.mean([r['body_joint_rmse_rad']**2 for r in prefall]))) if prefall else None
    summary['prefall_balance_fraction'] = float(np.mean([r['balanced'] for r in prefall])) if prefall else None
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError('Source NPZ changed during verification')
    (prefix/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    write_csv(prefix/'summary.csv', [{k: v for k,v in summary.items() if k!='warnings'}])
    np.savez_compressed(prefix/'physical_states.npz', qpos=qpos_states, qvel=qvel_states,
                        time_s=[r['time_s'] for r in reference_rows], source_sha256=source_hash,
                        floor_offset_m=offset, model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest())
    plots(prefix, rows)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT/'outputs/m3_spl_fitting/spl_P0002_T0049.npz')
    p.add_argument('--model', type=Path, default=ROOT.parents[2]/'basketball-r1/SMPLSim/outputs/smplh_physics/body_01.xml')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/m3_spl_physics/aligned')
    p.add_argument('--root-mode', choices=('aligned','raw'), default='aligned')
    p.add_argument('--gui', action='store_true')
    p.add_argument('--video', action='store_true')
    run(p.parse_args())
