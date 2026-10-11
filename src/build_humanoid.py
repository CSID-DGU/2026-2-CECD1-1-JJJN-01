"""Build an articulated MuJoCo humanoid from local SMPL+H rest geometry.

Axes: MuJoCo X forward, Y left, Z up = SMPL Z, X, Y.
Primitive collision geometry and approximate inertial/actuator parameters;
no learned balance controller and no SMPL pose-corrective visual skin.
"""
from pathlib import Path
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import smplx
import torch

ROOT = Path(__file__).resolve().parents[1]
BODY_NAMES = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee',
              'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3',
              'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar',
              'head', 'left_shoulder', 'right_shoulder', 'left_elbow',
              'right_elbow', 'left_wrist', 'right_wrist']
NAMES = BODY_NAMES + [f'{side}_{finger}{segment}' for side in ['left', 'right']
                      for finger in ['index', 'middle', 'pinky', 'ring', 'thumb']
                      for segment in [1, 2, 3]]


def fmt(values):
    return ' '.join(f'{float(x):.9g}' for x in np.atleast_1d(values))


def joint_spec(index):
    """Axes are local XYZ at rest; limits are engineering initial estimates."""
    if index in (4, 5):
        return [('y', (-0.08, 2.5))], 200., 22., 180.
    if index in (18, 19):
        limits = (-2.5, 0.08) if index == 18 else (-0.08, 2.5)
        return [('z', limits)], 65., 5., 65.
    if index in (10, 11):
        return [('y', (-0.5, 0.8))], 25., 2., 35.
    if index >= 22:
        local = (index - 22) % 15
        left = index < 37
        flex = (-0.15, 1.5) if left else (-1.5, 0.15)
        if local >= 12:  # Thumb axes need calibration against a grasp reference.
            axes = [('x', (-1.2, 1.2)), ('y', (-0.8, 0.8)), ('z', (-0.8, 0.8))] if local == 12 else [('x', (-1.2, 1.2))]
        else:
            axes = [('x', flex), ('z', (-0.35, 0.35))] if local % 3 == 0 else [('x', flex)]
        return axes, 2., 0.08, 1.5
    if index in (1, 2):
        return [('x', (-0.8, 0.8)), ('y', (-2.0, 0.8)), ('z', (-0.8, 0.8))], 200., 22., 180.
    if index in (7, 8):
        return [('x', (-0.45, 0.45)), ('y', (-0.8, 0.6)), ('z', (-0.3, 0.3))], 160., 14., 120.
    if index in (16, 17):
        return [('x', (-3.0, 3.0)), ('y', (-1.55, 1.55)), ('z', (-3.0, 3.0))], 80., 7., 80.
    if index in (20, 21):
        return [('x', (-1.2, 1.2)), ('y', (-1.0, 1.0)), ('z', (-0.7, 0.7))], 15., 1., 15.
    if index in (13, 14):
        return [('x', (-0.35, 0.35)), ('y', (-0.3, 0.3)), ('z', (-0.35, 0.35))], 60., 5., 35.
    if index in (12, 15):
        return [('x', (-0.5, 0.5)), ('y', (-0.6, 0.6)), ('z', (-0.8, 0.8))], 25., 2., 20.
    return [('x', (-0.4, 0.4)), ('y', (-0.5, 0.5)), ('z', (-0.5, 0.5))], 160., 14., 120.


def build(model_path, output, total_mass=75.):
    if not np.isfinite(total_mass) or total_mass <= 0:
        raise ValueError('Total mass must be positive and finite')
    source = smplx.SMPLH(str(model_path), gender='male', use_pca=False,
                        flat_hand_mean=True, num_betas=10)
    with torch.no_grad():
        rest = source()
    vertices = rest.vertices[0].numpy()[:, [2, 0, 1]]
    joints = rest.joints[0, :52].numpy()[:, [2, 0, 1]]
    parents = source.parents.numpy()
    weights = source.lbs_weights.numpy()
    if not np.isfinite(vertices).all() or not np.isfinite(joints).all():
        raise ValueError('Invalid SMPL geometry')
    scale = float(np.ptp(vertices[:, 2]) / 1.79)
    masses = np.array([10., 7., 7., 5., 3.3, 3.3, 6., .8, .8, 10., .45, .45,
                       1., .4, .4, 4.5, 1.9, 1.9, 1.2, 1.2, .3, .3] + [.018] * 30)
    masses *= total_mass / masses.sum()
    xml = ET.Element('mujoco', model='smplh_male_humanoid')
    ET.SubElement(xml, 'compiler', angle='radian', autolimits='true', inertiafromgeom='true')
    ET.SubElement(xml, 'option', timestep='0.001', gravity='0 0 -9.81',
                  integrator='implicitfast', cone='elliptic', iterations='80')
    visual = ET.SubElement(xml, 'visual')
    ET.SubElement(visual, 'global', offwidth='1280', offheight='960')
    ET.SubElement(visual, 'headlight', ambient='0.35 0.35 0.35', diffuse='0.7 0.7 0.7')
    defaults = ET.SubElement(xml, 'default')
    ET.SubElement(defaults, 'joint', type='hinge', damping='0.15', armature='0.01',
                  solreflimit='0.01 1')
    ET.SubElement(defaults, 'geom', friction='0.9 0.02 0.002', condim='4',
                  solref='0.008 1', solimp='0.9 0.95 0.001', rgba='0.45 0.65 0.85 1')
    world = ET.SubElement(xml, 'worldbody')
    ET.SubElement(world, 'light', pos='2 -3 4', dir='-1 1 -2')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='5 5 0.1',
                  rgba='0.22 0.25 0.28 1')
    actuator = ET.SubElement(xml, 'actuator')
    contacts = ET.SubElement(xml, 'contact')
    sensor = ET.SubElement(xml, 'sensor')
    bodies = {}
    mapping = []
    origin_shift = np.array([-joints[0, 0], -joints[0, 1], -vertices[:, 2].min() + .005])
    rest_world = joints + origin_shift
    children = {i: np.flatnonzero(parents == i).tolist() for i in range(52)}

    def geom(body, idx, **attrs):
        attrs.update(name=NAMES[idx] + '_geom', mass=fmt(masses[idx]))
        if idx >= 22:
            attrs['rgba'] = '0.85 0.6 0.35 1'
        elif idx in (20, 21):
            attrs['rgba'] = '0.9 0.7 0.4 1'
        ET.SubElement(body, 'geom', **attrs)

    for idx, name in enumerate(NAMES):
        parent = int(parents[idx])
        pos = rest_world[0] if idx == 0 else joints[idx] - joints[parent]
        body = ET.SubElement(world if idx == 0 else bodies[parent], 'body', name=name, pos=fmt(pos))
        bodies[idx] = body
        record = {'smpl_index': idx, 'body': name, 'parent_index': parent,
                  'rest_joint_world_m': rest_world[idx].tolist(), 'joints': []}
        if idx == 0:
            ET.SubElement(body, 'freejoint', name='root')
        else:
            specs, kp, kv, force = joint_spec(idx)
            for axis, limits in specs:
                joint_name = name + '_' + axis
                axis_vector = np.eye(3)['xyz'.index(axis)]
                ET.SubElement(body, 'joint', name=joint_name, axis=fmt(axis_vector),
                              range=fmt(limits), damping='0.005' if idx >= 22 else '0.15',
                              armature='0.00002' if idx >= 22 else '0.01')
                ET.SubElement(actuator, 'position', name=joint_name + '_servo', joint=joint_name,
                              kp=fmt(kp), kv=fmt(kv), ctrlrange=fmt(limits),
                              forcerange=fmt([-force, force]))
                record['joints'].append({'name': joint_name, 'axis': axis,
                                         'range_rad': list(limits), 'kp': kp, 'kv': kv,
                                         'torque_limit_nm': force})
        mapping.append(record)
        ET.SubElement(body, 'site', name=name + '_joint_site', size='0.006', rgba='1 0.2 0.2 1', group='3')
        if idx in (0, 3, 6, 9):
            dims = {0: [.105, .135, .085], 3: [.10, .115, .085],
                    6: [.11, .14, .055], 9: [.115, .17, .12]}[idx]
            center = {0: [0, 0, -.015], 3: [0, 0, .04],
                      6: [0, 0, .025], 9: [0, 0, .095]}[idx]
            geom(body, idx, type='ellipsoid', size=fmt(np.array(dims) * scale), pos=fmt(np.array(center) * scale))
        elif idx == 15:
            points = vertices[weights[:, idx] > .5]
            low, high = np.quantile(points, [.01, .99], axis=0)
            geom(body, idx, type='ellipsoid', size=fmt((high - low) / 2), pos=fmt((low + high) / 2 - joints[idx]))
        elif idx in (7, 8, 10, 11):
            ankle, toe = (7, 10) if idx in (7, 10) else (8, 11)
            pts = vertices[weights[:, [ankle, toe]].sum(axis=1) > .5]
            low, high = pts.min(axis=0), pts.max(axis=0)
            split = joints[toe, 0]
            xmin, xmax = (low[0], split) if idx == ankle else (split, high[0])
            center = np.array([(xmin + xmax) / 2, (low[1] + high[1]) / 2, low[2] + .028 * scale])
            half = np.array([(xmax - xmin) / 2, (high[1] - low[1]) * .45, .028 * scale])
            geom(body, idx, type='box', size=fmt(half), pos=fmt(center - joints[idx]))
            if idx == ankle:
                ET.SubElement(body, 'site', name=name.replace('ankle', 'sole'), type='box',
                              pos=fmt(center - joints[idx] - [0, 0, half[2]]),
                              size=fmt([half[0], half[1], .003]), rgba='0 1 0 0.3')
        elif idx in (20, 21):
            base = 22 if idx == 20 else 37
            knuckles = joints[[base, base + 3, base + 6, base + 9]]
            endpoint = knuckles.mean(axis=0)
            center = (joints[idx] + endpoint) / 2
            half = [max(np.ptp(knuckles[:, 0]) * .55, .026 * scale),
                    abs(endpoint[1] - joints[idx, 1]) * .5 + .005 * scale, .012 * scale]
            geom(body, idx, type='box', size=fmt(half), pos=fmt(center - joints[idx]))
            ET.SubElement(body, 'site', name=name.replace('wrist', 'palm'), pos=fmt(center - joints[idx]), size='.012')
        else:
            if children[idx]:
                endpoint = joints[children[idx][0]] - joints[idx]
            else:
                # Terminal phalanx length estimated from skin vertices along its bone.
                direction = joints[idx] - joints[parent]
                direction /= np.linalg.norm(direction)
                points = vertices[weights[:, idx] > .35] - joints[idx]
                length = max(float(np.quantile(points @ direction, .95)), .012 * scale)
                endpoint = direction * length
            if idx >= 22:
                radius = (.009 if 'thumb' in name else .007) * scale
            elif idx == 12:
                radius = .047 * scale
            elif idx in (13, 14):
                radius = .033 * scale
            else:
                points = vertices[weights[:, idx] > .5] - joints[idx]
                unit = endpoint / np.linalg.norm(endpoint)
                distances = np.linalg.norm(points - np.outer(points @ unit, unit), axis=1)
                radius = float(np.clip(np.quantile(distances, .65), .025 * scale, .085 * scale))
            geom(body, idx, type='capsule', fromto=fmt(np.r_[np.zeros(3), endpoint]), size=fmt(radius))
            if idx >= 22 and not children[idx]:
                ET.SubElement(body, 'site', name=name + '_tip', pos=fmt(endpoint), size='.004')
    # Exclude near-neighbour overlap, and within each hand. Other self-contacts remain active.
    def ancestors(idx):
        path = [idx]
        while parents[path[-1]] >= 0:
            path.append(int(parents[path[-1]]))
        return path
    paths = [ancestors(i) for i in range(52)]
    hand_sets = [{20, *range(22, 37)}, {21, *range(37, 52)}]
    exclusions = []
    for a in range(52):
        for b in range(a + 1, 52):
            common = next(x for x in paths[a] if x in paths[b])
            distance = paths[a].index(common) + paths[b].index(common)
            if distance <= 2 or (a == 6 and b in (16, 17)) or any(a in hand and b in hand for hand in hand_sets):
                ET.SubElement(contacts, 'exclude', body1=NAMES[a], body2=NAMES[b])
                exclusions.append([NAMES[a], NAMES[b]])
    ET.SubElement(bodies[0], 'site', name='imu', size='.01')
    ET.SubElement(sensor, 'gyro', name='pelvis_gyro', site='imu')
    ET.SubElement(sensor, 'accelerometer', name='pelvis_accelerometer', site='imu')
    for side in ['left', 'right']:
        ET.SubElement(sensor, 'touch', name=side + '_sole_touch', site=side + '_sole')
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
    keyframes = ET.SubElement(xml, 'keyframe')
    ET.SubElement(keyframes, 'key', name='t_pose', qpos=fmt(temp_model.qpos0), ctrl=fmt(np.zeros(temp_model.nu)))
    qpos = temp_model.qpos0.copy()
    targets = {'left_shoulder_x': -1.15, 'right_shoulder_x': 1.15,
               'left_elbow_z': -.15, 'right_elbow_z': .15}
    ctrl = np.zeros(temp_model.nu)
    for name, value in targets.items():
        qpos[temp_model.joint(name).qposadr[0]] = value
        ctrl[temp_model.actuator(name + '_servo').id] = value
    ET.SubElement(keyframes, 'key', name='standing', qpos=fmt(qpos), ctrl=fmt(ctrl))
    ET.indent(xml)
    ET.ElementTree(xml).write(output, encoding='utf-8', xml_declaration=True)
    compiled = mujoco.MjModel.from_xml_string(output.read_text(encoding='utf-8'))
    meta = {
        'source_model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'source_model': str(model_path.resolve()), 'shape_betas': [0.] * 10,
        'coordinate_transform': 'mujoco_xyz = smpl_zxy (proper rotation)',
        'root_world_translation_m': origin_shift.tolist(),
        'mass_kg': float(compiled.body_mass.sum()), 'timestep_s': compiled.opt.timestep,
        'nq': compiled.nq, 'nv': compiled.nv, 'actuators': compiled.nu,
        'joint_mapping': mapping, 'contact_exclusions': exclusions,
        'assumptions': ['Mass distribution, joint limits, torque limits, gains, and friction are initial engineering estimates, not measured human parameters.',
                        'Capsule radii and feet/head dimensions are fitted to rest skin; torso and palm geometry are simplified.',
                        'Hinges use local XYZ composition, not SMPL axis-angle components. Retargeting requires rotation conversion and reduced-DOF fitting.',
                        'Within-hand collisions are excluded; hand-object and opposite-hand contacts remain enabled.',
                        'Standing is an initial pose, not a balance controller. Root is free and has no actuator, weld, or gravity compensation.',
                        'No basketball, task reward, motion clip, learned policy, or deformable visual skin is included.'],
    }
    output.with_suffix('.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'data/models/smplh/SMPLH_MALE.pkl')
    parser.add_argument('--output', type=Path, default=ROOT / 'assets/smplh_humanoid.xml')
    parser.add_argument('--mass', type=float, default=75.)
    args = parser.parse_args()
    meta = build(args.model, args.output, args.mass)
    print(json.dumps({key: meta[key] for key in ['mass_kg', 'nq', 'nv', 'actuators']}, indent=2))
    print('Saved:', args.output)


if __name__ == '__main__':
    main()



