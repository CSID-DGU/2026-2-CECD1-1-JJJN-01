"""M4-2 physical weld holding; ball state written once during initialization."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from smpl_sim.physics.basketball import build_xml, BALL_MASS
from smpl_sim.physics.ball_release import HandBallRelease, hand_ball_state


def build_holding_xml(humanoid, destination, hand='right', offset=None, weld=True, test_mass=BALL_MASS):
    """Derived scene only. Preserve M4-1 geometry, inertia law and contact settings."""
    destination=build_xml(humanoid,destination)
    tree=ET.parse(destination);root=tree.getroot()
    root.set('model',Path(humanoid).stem+'_physical_holding')
    offset=np.asarray(offset if offset is not None else [-.07 if hand=='right' else .07,-.19,0.])
    if offset.shape!=(3,) or not np.isfinite(offset).all():raise ValueError('Invalid offset')
    if not np.isfinite(test_mass) or test_mass<=0:raise ValueError('Invalid test mass')
    # Explicit optional diagnostic fixture, not the standard 600 g experiment.
    if test_mass!=BALL_MASS:
        inertia=root.find(".//body[@name='basketball']/inertial")
        inertia.set('mass',str(test_mass));values=np.fromstring(inertia.get('diaginertia'),sep=' ')*test_mass/BALL_MASS
        inertia.set('diaginertia',' '.join(map(str,values)))
    if weld:
        equality=root.find('equality')
        if equality is None:equality=ET.SubElement(root,'equality')
        ET.SubElement(equality,'weld',name='hand_ball_weld',body1=('R' if hand=='right' else 'L')+'_Wrist',
                      body2='basketball',relpose=' '.join(map(str,offset))+' 1 0 0 0',anchor='0 0 0',
                      active='false',solref='.02 1',solimp='.95 .99 .001',torquescale='.121')
    ET.indent(tree,space='  ');tree.write(destination,encoding='unicode');return destination


class WeldBallRelease(HandBallRelease):
    """One initial alignment, then solver-only holding and unchanged-state release."""
    def __init__(self,model,hand='right',offset=None,release_time=2.5,hand_collision='shooting'):
        super().__init__(model,hand,offset,release_time,hand_collision)
        self.equality=model.equality('hand_ball_weld').id
        # Ensure the compiled relpose and grip offset describe the same anchor.
        if not np.allclose(model.eq_data[self.equality,3:6],self.offset):
            raise ValueError('Weld relpose and initial grip offset differ')
        self.initialized=False;self.initial_error=None;self.initial_force=None
        self.state_write_count=0

    def _masks(self,values):
        bid=int(self.model.geom_bodyid[self.geom])
        self.model.geom_contype[self.geom],self.model.geom_conaffinity[self.geom]=values
        self.model.body_contype[bid],self.model.body_conaffinity[bid]=values

    def initialize(self,data):
        if self.initialized:raise RuntimeError('Initialize only once on a fresh state')
        data.eq_active[self.equality]=0
        p,q,v,w=hand_ball_state(self.model,data,self.hand,self.offset)
        R=data.xmat[self.model.body(('R' if self.hand=='right' else 'L')+'_Wrist').id].reshape(3,3)
        data.qpos[self.qa:self.qa+7]=np.r_[p,q]
        data.qvel[self.da:self.da+6]=np.r_[v,R.T@w]
        self.state_write_count+=1
        # Floor/rim/backboard remain physical even while held; hand contact off.
        self._masks((8,16));data.eq_active[self.equality]=1
        mujoco.mj_forward(self.model,data)
        mask=(data.efc_type==mujoco.mjtConstraint.mjCNSTR_EQUALITY)&(data.efc_id==self.equality)
        self.initial_error=float(np.max(np.abs(data.efc_pos[mask]),initial=0.))
        self.initial_force=float(np.max(np.abs(data.efc_force[mask]),initial=0.))
        if self.initial_error>1e-9:raise RuntimeError('Initial weld not aligned')
        self.initialized=True
        self.update(data)

    def update(self,data):
        if not self.initialized:raise RuntimeError('Initialize weld first')
        if self.released or data.time+1e-10<self.release_time:return False
        p=data.qpos[self.qa:self.qa+7].copy();v=data.qvel[self.da:self.da+6].copy()
        R=data.xmat[self.model.body('basketball').id].reshape(3,3)
        data.eq_active[self.equality]=0;self._masks(self.release_masks)
        mujoco.mj_forward(self.model,data)
        self.released=True
        self.event={'requested_time_s':self.release_time,'actual_time_s':float(data.time),
                    'position_m':p[:3].tolist(),'quaternion_wxyz':p[3:].tolist(),
                    'velocity_world_m_s':v[:3].tolist(),'angular_velocity_world_rad_s':(R@v[3:]).tolist(),
                    'position_jump_m':float(np.linalg.norm(data.qpos[self.qa:self.qa+3]-p[:3])),
                    'velocity_jump_m_s':float(np.linalg.norm(data.qvel[self.da:self.da+3]-v[:3])),
                    'angular_velocity_jump_rad_s':float(np.linalg.norm(data.qvel[self.da+3:self.da+6]-v[3:]))}
        return False
