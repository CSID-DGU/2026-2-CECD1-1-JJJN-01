"""Derived shooting model with anatomical forearm pronation/supination axes."""
import xml.etree.ElementTree as ET
import hashlib,json
import numpy as np
import mujoco
from humanoid_model import ROOT,MODEL_PATH
from shoot_spl import numbers
from track_spl import motor_model


def model_xml():
    tree=ET.fromstring(MODEL_PATH.read_text(encoding='utf-8'))
    key=tree.find('keyframe')
    if key is not None:tree.remove(key)
    for a in tree.find('actuator'):
        a.tag='motor';a.attrib.pop('kp',None);a.attrib.pop('kv',None);a.set('ctrlrange',a.get('forcerange'))
    for side in ['left','right']:
        body=tree.find(f".//body[@name='{side}_elbow']")
        wrist=body.find(f"body[@name='{side}_wrist']")
        axis=np.fromstring(wrist.get('pos'),sep=' ');axis/=np.linalg.norm(axis)
        joint=ET.Element('joint',name=side+'_forearm_roll',axis=numbers(axis),range='-1.5 1.5',damping='.05',armature='.004')
        body.insert(1,joint)
        ET.SubElement(tree.find('actuator'),'motor',name=side+'_forearm_roll_motor',joint=side+'_forearm_roll',
                      ctrlrange='-8 8',forcerange='-8 8',forcelimited='true')
    return ET.tostring(tree,encoding='unicode')


def contact_motor_model():
    return mujoco.MjModel.from_xml_string(model_xml())


def expand_reference(original,model):
    """Transfer by joint names; never assume adding joints preserves qpos order."""
    old=motor_model();result={k:v.copy() for k,v in original.items()}
    result['qpos']=np.zeros((len(original['times']),model.nq))
    result['qvel']=np.zeros((len(original['times']),model.nv));result['qacc']=result['qvel'].copy()
    for j in range(old.njnt):
        new=model.joint(old.joint(j).name).id
        qa,va=int(old.jnt_qposadr[j]),int(old.jnt_dofadr[j]);qb,vb=int(model.jnt_qposadr[new]),int(model.jnt_dofadr[new])
        nq,nv=(7,6) if old.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE else (1,1)
        result['qpos'][:,qb:qb+nq]=original['qpos'][:,qa:qa+nq]
        for key in ['qvel','qacc']:result[key][:,vb:vb+nv]=original[key][:,va:va+nv]
    return result


def save_model():
    xml=model_xml();path=ROOT/'assets/smplh_shooting.xml';path.write_text(xml,encoding='utf-8')
    report={'base_model_sha256':hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
            'model_sha256':hashlib.sha256(xml.encode()).hexdigest(),
            'added_axes':['left_forearm_roll','right_forearm_roll'],
            'range_rad':[-1.5,1.5],'torque_limit_nm':8.,
            'note':'Axes follow elbow-to-wrist segment; engineering range/torque, not subject-specific measurements.'}
    path.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return path
