"""Diagnostics only: shortest quaternion angle and loaded floor-contact slip."""
import numpy as np
import mujoco


def quaternion_error_deg(actual, target):
    """MuJoCo wxyz; q and -q describe the same orientation."""
    a, b = np.asarray(actual), np.asarray(target)
    dot = abs(float(a @ b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(2 * np.arccos(np.clip(dot, 0, 1))))


def foot_contact_slip(model, data):
    """Force-weighted tangent velocity of foot material at contact, static floor.

    A contact's world position itself is not a persistent material point. Its
    frame-to-frame displacement is therefore deliberately not used as slip.
    """
    totals = {s: [0., 0., 0., 0.] for s in ('L', 'R')}
    rows = []
    floor = model.geom('floor').id
    jac = np.zeros((3, model.nv))
    wrench = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        if floor not in (c.geom1, c.geom2):
            continue
        gid = c.geom2 if c.geom1 == floor else c.geom1
        bid = int(model.geom_bodyid[gid])
        name = model.body(bid).name
        side = next((s for s in ('L', 'R') if name in (s+'_Ankle', s+'_Toe')), None)
        mujoco.mj_contactForce(model, data, i, wrench)
        force = float(wrench[0])
        if side is None or force <= .1:
            continue
        mujoco.mj_jac(model, data, jac, None, c.pos, bid)
        velocity = jac @ data.qvel
        normal = c.frame[:3]
        tangent = velocity - normal * (velocity @ normal)
        speed = float(np.linalg.norm(tangent))
        v = totals[side]
        v[0] += force
        v[1] += force * speed
        v[2] += force * speed**2
        v[3] = max(v[3], speed)
        rows.append({'time_s': float(data.time), 'side': side, 'body': name,
                     'x_m': float(c.pos[0]), 'y_m': float(c.pos[1]), 'z_m': float(c.pos[2]),
                     'normal_force_N': force, 'tangent_speed_m_s': speed})
    result = {}
    for s, (force, linear, squared, maximum) in totals.items():
        result[s] = {'force': force, 'mean_speed': linear / force if force else 0.,
                     'rms_speed': np.sqrt(squared / force) if force else 0.,
                     'max_speed': maximum}
    return result, rows
