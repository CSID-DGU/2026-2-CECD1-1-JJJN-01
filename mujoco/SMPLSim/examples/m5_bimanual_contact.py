"""M5-1 derived contact masks and side-specific contact telemetry.
No model dimensions, masses, inertias, friction or solver parameters are changed.
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.m5_residual_ppo_env import BASE
from examples.m3_control import FINGERS
from examples.m4_hand_kinematics import capsule, closest
import mujoco
import numpy as np
OUT = ROOT / 'outputs/m5_1_upper_body'


def hand_geom(name, side):
    return name == side+'_Wrist' or name.startswith(side+'_') and any(f in name for f in FINGERS)


def derived_scene(destination=OUT/'bimanual_contact.xml'):
    destination = Path(destination)
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(BASE/'contact.xml')
    changes = []
    for geom in tree.findall('.//geom'):
        name = geom.get('name', '')
        if hand_geom(name, 'L'):
            before = int(geom.get('contype', '7'))
            geom.set('contype', str(before | 32))
            changes.append(dict(geom=name, before=before, after=before | 32))
        # Collision safety is measured, rather than allowing the ball through the face.
        if name in ('Head', 'Neck', 'Chest'):
            before = int(geom.get('contype', '7'))
            geom.set('contype', str(before | 64))
            changes.append(dict(geom=name, before=before, after=before | 64))
        if name == 'basketball_geom':
            geom.set('conaffinity', '112')  # environment16 | hands32 | face/chest64
    tree.getroot().set('model', 'M5_1_bimanual_contact')
    ET.indent(tree, space='  ')
    tree.write(destination, encoding='unicode')
    (destination.parent/'collision_changes.json').write_text(json.dumps(changes, indent=2))
    return destination


def side_contacts(m, d, side):
    ball = m.geom('basketball_geom').id
    result = dict(count=0, force=np.zeros(3), normal_force=0., max_normal=0.,
                  fingers=np.zeros(5), finger_flags=np.zeros(5), slip=0., records=[])
    slips = []
    for i in range(d.ncon):
        c = d.contact[i]
        if ball not in (c.geom1, c.geom2):
            continue
        other = int(c.geom2 if c.geom1 == ball else c.geom1)
        name = m.geom(other).name or ''
        if not hand_geom(name, side):
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, wrench)
        if wrench[0] <= .01:
            continue
        sign = -1 if c.geom1 == ball else 1
        normal = c.frame[:3]*sign
        force = c.frame.reshape(3, 3).T @ wrench[:3]*sign
        result['count'] += 1
        result['force'] += force
        result['normal_force'] += float(wrench[0])
        result['max_normal'] = max(result['max_normal'], float(wrench[0]))
        for k, f in enumerate(FINGERS):
            if name.startswith(side+'_'+f):
                result['fingers'][k] += wrench[0]
                result['finger_flags'][k] = 1
        velocities = []
        for bid in (int(m.geom_bodyid[other]), m.body('basketball').id):
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jp, jr, c.pos, bid)
            velocities.append(jp @ d.qvel)
        relative = velocities[1]-velocities[0]
        slips.append(np.linalg.norm(relative-normal*np.dot(normal, relative)))
        result['records'].append(dict(geom=name, normal_force_N=float(wrench[0]),
            force_x_N=float(force[0]), force_y_N=float(force[1]), force_z_N=float(force[2]),
            normal_x=float(normal[0]), normal_y=float(normal[1]), normal_z=float(normal[2]),
            penetration_mm=float(max(0, -c.dist)*1000)))
    result['slip'] = float(np.sqrt(np.mean(np.square(slips)))) if slips else 0.
    return result


def local_ball(m, d, side):
    wrist = m.body(side+'_Wrist').id
    R = d.xmat[wrist].reshape(3, 3)
    bq = int(m.joint('basketball_free').qposadr[0])
    bv = int(m.joint('basketball_free').dofadr[0])
    r = d.qpos[bq:bq+3]-d.xpos[wrist]
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, wrist, velocity, 0)
    return R.T@r, R.T@(d.qvel[bv:bv+3]-velocity[3:]-np.cross(velocity[:3], r))


def gap_to_ball(m, d, geom_name, center):
    g = m.geom(geom_name).id
    if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        a, b, r = capsule(m, d, g)
        return float(np.linalg.norm(center-closest(center, a, b))-.121-r)
    if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX):
        local=d.geom_xmat[g].reshape(3,3).T @ (center-d.geom_xpos[g])
        excess=abs(local)-m.geom_size[g]
        return float(np.linalg.norm(np.maximum(excess,0))+min(float(np.max(excess)),0)-.121)
    return float(np.linalg.norm(center-d.geom_xpos[g])-.121-m.geom_size[g,0])
