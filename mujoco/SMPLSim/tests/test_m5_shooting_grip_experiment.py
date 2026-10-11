"""Actual MuJoCo wrench/frame checks for the independent grip experiment."""
import numpy as np
import mujoco
from examples.m5_shooting_grip_experiment import contact_wrenches,maximum_gap,ROOT


def sphere_contact():
    m=mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 -9.81"/><worldbody><geom name="R_Wrist" type="plane" size="1 1 .1"/><body name="basketball" pos="0 0 .099"><freejoint/><geom name="basketball_geom" type="sphere" size=".1" mass=".6"/></body></worldbody></mujoco>''');d=mujoco.MjData(m);mujoco.mj_forward(m,d);return m,d


def test_contact_force_is_on_ball_and_torque_about_center():
    m,d=sphere_contact();w,points=contact_wrenches(m,d)
    assert w['R']['force'][2]>0
    assert np.linalg.norm(w['R']['torque'])<1e-9
    assert points and points[0]['normal_z']>0
    assert w['L']['count']==0


def test_wrench_agrees_with_contact_transform():
    m,d=sphere_contact();w,points=contact_wrenches(m,d)
    force=np.zeros(6);mujoco.mj_contactForce(m,d,0,force);c=d.contact[0]
    sign=-1 if c.geom1==m.geom('basketball_geom').id else 1
    expected=c.frame.reshape(3,3).T@force[:3]*sign
    np.testing.assert_allclose(w['R']['force'],expected)
    lever=c.pos-d.geom_xpos[m.geom('basketball_geom').id]
    np.testing.assert_allclose(w['R']['torque'],np.cross(lever,expected)+sign*c.frame.reshape(3,3).T@force[3:])


def test_no_contacts_returns_zero_wrench():
    m,d=sphere_contact();d.qpos[2]=1;mujoco.mj_forward(m,d);w,p=contact_wrenches(m,d)
    assert not p
    for z in w.values():assert z['count']==0 and not np.any(z['force']) and not np.any(z['torque'])


def test_contact_microbounce_gate_is_time_based():
    rows=[dict(time_s=i*.01,right_contacts=0 if 2<=i<=4 else 1) for i in range(8)]
    assert np.isclose(maximum_gap(rows),.03)
    assert maximum_gap(rows)<.05
    rows=[dict(time_s=i*.01,right_contacts=0) for i in range(8)]
    assert maximum_gap(rows)>.05


def test_original_scene_has_no_weld_and_ball_parameters_preserved():
    m=mujoco.MjModel.from_xml_path(str(ROOT/'outputs/m5_1_upper_body/bimanual_contact.xml'))
    assert m.neq==0
    assert np.isclose(m.body_mass[m.body('basketball').id],.6)
    assert np.isclose(m.geom('basketball_geom').size[0],.121)


def load_tests(loader,tests,pattern):
    import unittest
    return unittest.TestSuite(unittest.FunctionTestCase(v) for k,v in sorted(globals().items()) if k.startswith('test_') and callable(v))


def test_new_guide_targets_and_controller_do_not_write_ball_or_external_force():
    from examples.m5_shooting_grip_experiment import GripEnv
    e=GripEnv(yaw=-15,guide=True,gentle=True)
    try:
        e.reset(seed=7);p=e.d.qpos[e.ball_q:e.ball_q+7].copy();v=e.d.qvel[e.ball_v:e.ball_v+6].copy();original=e.adapter.reference['right_rotation'].copy()
        q=e.adapter.target(.1,e.d);e.adapter.control(e.d,q)
        np.testing.assert_array_equal(e.d.qpos[e.ball_q:e.ball_q+7],p);np.testing.assert_array_equal(e.d.qvel[e.ball_v:e.ball_v+6],v)
        np.testing.assert_array_equal(e.adapter.reference['right_rotation'],original)
        assert not e.d.xfrc_applied.any() and not e.d.qfrc_applied.any() and e.m.neq==0
        np.testing.assert_allclose(e.adapter.desired_left_rotation.T@e.adapter.desired_left_rotation,np.eye(3),atol=1e-10)
        assert np.isclose(np.linalg.det(e.adapter.desired_left_rotation),1)
    finally:e.close()
