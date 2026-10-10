"""M4-1 static hoop, independently free basketball, unchanged humanoid controller."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from examples.m3_control import BalancedPD, ground
from smpl_sim.motion.smplh_motion import SMPLHMotion

BALL_RADIUS = .121
BALL_MASS = .600
FREE_THROW_X = .350  # far edge; original humanoid stands behind it
BOARD_FACE_X = FREE_THROW_X + 4.600
BOARD_BOTTOM_Z = 2.900
RIM_INNER_RADIUS = .227
RIM_TUBE_RADIUS = .010
RIM_CENTERLINE_RADIUS = RIM_INNER_RADIUS + RIM_TUBE_RADIUS
RIM_TOP_Z = 3.050
RIM_CENTER = np.array([BOARD_FACE_X-.151-RIM_INNER_RADIUS, 0., RIM_TOP_Z-RIM_TUBE_RADIUS])
RIM_SEGMENTS = 64


def build_xml(source, destination):
    """Extend a copy of existing XML; ball appended after all humanoid joints."""
    tree = ET.parse(source)
    root = tree.getroot()
    root.set('model','body_01_basketball_m4_1')
    option=root.find('option')
    if option is None:option=ET.SubElement(root,'option')
    option.set('timestep',str(1/450))
    option.set('gravity','0 0 -9.81')
    visual=root.find('visual')
    if visual is None:visual=ET.SubElement(root,'visual')
    global_visual=visual.find('global')
    if global_visual is None:global_visual=ET.SubElement(visual,'global')
    global_visual.set('offwidth','960');global_visual.set('offheight','540')
    world=root.find('worldbody')
    floor=world.find("geom[@name='floor']")
    floor.set('contype','23')  # original 7 | environment bit 16
    floor.set('conaffinity','9')  # original 1 | ball bit 8
    ET.SubElement(world,'geom',name='free_throw_line',type='box',
                  pos=f'{FREE_THROW_X-.025} 0 .001',size='.025 1.8 .001',rgba='1 1 1 1',
                  contype='0',conaffinity='0',mass='0')
    hoop={'contype':'16','conaffinity':'8','condim':'6','priority':'3',
          'friction':'.6 .005 .0002','solref':'-80000 -60','solimp':'.95 .99 .01','margin':'0'}
    ET.SubElement(world,'geom',name='backboard',type='box',
                  pos=f'{BOARD_FACE_X+.02} 0 {BOARD_BOTTOM_Z+.525}',size='.02 .9 .525',
                  rgba='.8 .9 1 .65',**hoop)
    for i in range(RIM_SEGMENTS):
        a,b=2*np.pi*np.array([i,i+1])/RIM_SEGMENTS
        p=RIM_CENTER+[RIM_CENTERLINE_RADIUS*np.cos(a),RIM_CENTERLINE_RADIUS*np.sin(a),0]
        q=RIM_CENTER+[RIM_CENTERLINE_RADIUS*np.cos(b),RIM_CENTERLINE_RADIUS*np.sin(b),0]
        ET.SubElement(world,'geom',name=f'rim_{i:02d}',type='capsule',
                      fromto=' '.join(map(str,np.r_[p,q])),size=str(RIM_TUBE_RADIUS),
                      rgba='1 .28 .02 1',**{**hoop,'solref':'-80000 -80'})
    # Decorative bracket/stand: not a certified basket support and not a collision obstacle.
    ET.SubElement(world,'geom',name='hoop_stand',type='capsule',
                  fromto=f'{BOARD_FACE_X+.6} 0 0 {BOARD_FACE_X+.6} 0 3.4',size='.06',
                  rgba='.25 .25 .3 1',contype='0',conaffinity='0')
    ET.SubElement(world,'geom',name='hoop_support',type='capsule',
                  fromto=f'{BOARD_FACE_X+.6} 0 3.4 {BOARD_FACE_X+.04} 0 3.4',size='.05',
                  rgba='.25 .25 .3 1',contype='0',conaffinity='0')
    ball=ET.SubElement(world,'body',name='basketball',pos='2 0 2')
    ET.SubElement(ball,'freejoint',name='basketball_free')
    inertia=2/3*BALL_MASS*BALL_RADIUS**2  # uniform thin spherical shell approximation
    ET.SubElement(ball,'inertial',pos='0 0 0',mass=str(BALL_MASS),diaginertia=f'{inertia} {inertia} {inertia}')
    ET.SubElement(ball,'geom',name='basketball_geom',type='sphere',size=str(BALL_RADIUS),
                  rgba='.9 .3 .035 1',contype='8',conaffinity='16',condim='6',priority='2',
                  friction='.7 .005 .0002',solref='-40000 -45',solimp='.95 .99 .01',margin='0')
    ET.indent(tree,space='  ')
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    tree.write(destination,encoding='unicode')
    return destination


def test_ball_initial_state(model,data,position,velocity,angular_velocity=(0,0,0)):
    """TEST LAUNCHER ONLY. No hand release, attachment or per-step applied force."""
    joint=model.joint('basketball_free')
    qa,da=int(joint.qposadr[0]),int(joint.dofadr[0])
    values=np.r_[position,velocity,angular_velocity]
    if values.shape!=(9,) or not np.isfinite(values).all():raise ValueError('Expected finite xyz vectors')
    data.qpos[qa:qa+3]=position;data.qpos[qa+3:qa+7]=[1,0,0,0]
    data.qvel[da:da+3]=velocity;data.qvel[da+3:da+6]=angular_velocity
    mujoco.mj_forward(model,data)


class HumanoidControllerBridge:
    """Synchronize humanoid state into original model; use its existing PD + COM.

    Valid in M4-1 because ball/hoop do not contact the humanoid. Hand-ball coupling
    in M4-2 will require contact-force-aware integration; this is not a release API.
    """
    def __init__(self, humanoid_xml, integrated_model, motion_file):
        self.model=mujoco.MjModel.from_xml_path(str(humanoid_xml))
        self.model.opt.timestep=integrated_model.opt.timestep
        self.data=mujoco.MjData(self.model)
        self.motion=SMPLHMotion.load(motion_file)
        self.bound=self.motion.bind(self.model)
        self.controller=BalancedPD(self.model,self.bound.mapping)
        probe=mujoco.MjData(self.model);probe.qpos[:]=self.bound.sample(0)
        z=float(probe.qpos[2]);ground(self.model,probe)
        self.offset=float(probe.qpos[2]-z)
        if integrated_model.nu!=self.model.nu:raise ValueError('Humanoid actuators changed')
        for j in range(self.model.njnt):
            if integrated_model.joint(j).name!=self.model.joint(j).name:raise ValueError('Humanoid joint order changed')

    def target(self,t):
        q=self.bound.sample(t);q[2]+=self.offset;return q

    def initialize(self,model,data):
        mujoco.mj_resetData(model,data)
        data.qpos[:self.model.nq]=self.target(0)
        mujoco.mj_forward(model,data)

    def control(self,data):
        self.data.qpos[:]=data.qpos[:self.model.nq]
        self.data.qvel[:]=data.qvel[:self.model.nv]
        self.data.ctrl[:]=data.ctrl
        self.data.time=data.time
        mujoco.mj_forward(self.model,self.data)
        torque,_=self.controller.control(self.data,self.target(data.time+self.model.opt.timestep))
        data.ctrl[:]=torque


class HoopPassage:
    """Geometric passage event, no RL reward. Require descending entry and full exit.

    Centre crossing uses radius clearance; exit waits until the whole ball is
    below the ring. Misses, lateral bypass and upward crossings are rejected.
    """
    def __init__(self):self.entered=False;self.passed=False;self.entry_time=None
    def update(self,previous,current,t,dt):
        z=RIM_CENTER[2]
        if previous[2]>z>=current[2]:
            a=(previous[2]-z)/(previous[2]-current[2])
            p=previous+a*(current-previous)
            # Capsule chord lowers the minimum inner clearance slightly.
            aperture=RIM_CENTERLINE_RADIUS*np.cos(np.pi/RIM_SEGMENTS)-RIM_TUBE_RADIUS
            if np.linalg.norm(p[:2]-RIM_CENTER[:2])+BALL_RADIUS<aperture:
                self.entered=True;self.entry_time=t-dt+a*dt
        if self.entered and abs(current[2]-z)<BALL_RADIUS:
            section=np.sqrt(max(0.,BALL_RADIUS**2-(current[2]-z)**2))
            aperture=RIM_CENTERLINE_RADIUS*np.cos(np.pi/RIM_SEGMENTS)-RIM_TUBE_RADIUS
            if np.linalg.norm(current[:2]-RIM_CENTER[:2])+section>=aperture:
                self.entered=False
        if self.entered and current[2]+BALL_RADIUS<z-RIM_TUBE_RADIUS:
            self.passed=True
        return self.passed


def cases():
    c=RIM_CENTER
    return {
      'freefall':(np.array([2.,0.,2.]),np.zeros(3),.3),
      'floor':(np.array([2.,-1.,1.8+BALL_RADIUS]),np.zeros(3),1.5),
      'backboard':(np.array([BOARD_FACE_X-.6,.55,3.65]),np.array([3.,0.,0.]),.65),
      'rim':(c+np.array([-RIM_CENTERLINE_RADIUS,0.,.6]),np.array([0.,0.,-1.]),.8),
      'pass':(c+np.array([0.,0.,.7]),np.array([0.,0.,-1.]),.8),
      'launch':(np.array([0.,0.,2.]),np.array([c[0],0.,c[2]-2.+4.905]),1.5),
    }
