"""M4-2D staged SMPL+H contact experiments; original M3 controller is imported unchanged."""
import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from smpl_sim.physics.basketball import build_xml,HumanoidControllerBridge,BALL_RADIUS
from smpl_sim.motion.physics_metrics import foot_contact_slip
from examples.m3_control import write_csv,contacts,FINGERS
from examples.m4_physical_holding import support_margin_at


def is_finger(name):
    return name.startswith('R_') and any(f in name for f in FINGERS)


def build_scene(source,destination,palm_pad=False,soft_hand_contact=False):
    build_xml(source,destination)
    tree=ET.parse(destination);root=tree.getroot()
    root.set('model','SMPLH_contact_only')
    for geom in root.findall('.//geom'):
        name=geom.get('name','')
        if name=='R_Wrist' or is_finger(name):geom.set('contype',str(int(geom.get('contype','1'))|32))
    root.find(".//geom[@name='basketball_geom']").set('conaffinity','48')
    if palm_pad:
        # Optional derived collision proxy. Explicit mass=0 avoids changing hand inertia.
        ET.SubElement(root.find(".//body[@name='R_Wrist']"),'geom',name='contact_palm_pad',type='box',
                      pos='-.066 -.037 -.007',size='.040 .004 .030',mass='0',contype='32',conaffinity='8',
                      rgba='.3 .7 .5 .5')
    if soft_hand_contact:
        contact=root.find('contact')
        if contact is None:contact=ET.SubElement(root,'contact')
        for geom in root.findall('.//geom'):
            name=geom.get('name','')
            if name=='R_Wrist' or name=='contact_palm_pad' or is_finger(name):
                ET.SubElement(contact,'pair',geom1=name,geom2='basketball_geom',condim='6',margin='0',
                              friction='.7 .7 .005 .0002 .0002',solref='.005 1',solimp='.95 .99 .001')
    ET.indent(tree,space='  ');tree.write(destination,encoding='unicode');return destination


def hand_contacts(m,d):
    bg=m.geom('basketball_geom').id;count=0;finger_count=0;force=np.zeros(3);f_force=np.zeros(3);depth=0.;slips=[]
    for i in range(d.ncon):
        c=d.contact[i]
        if bg not in (c.geom1,c.geom2):continue
        other=int(c.geom2 if c.geom1==bg else c.geom1);body=int(m.geom_bodyid[other]);name=m.body(body).name
        if name!='R_Wrist' and not is_finger(name):continue
        w=np.zeros(6);mujoco.mj_contactForce(m,d,i,w)
        if w[0]<=.01:continue
        f=c.frame.reshape(3,3).T@w[:3]*(-1 if c.geom1==bg else 1)
        count+=1;force+=f;depth=max(depth,-c.dist)
        if is_finger(name):finger_count+=1;f_force+=f
        velocities=[]
        for bid in (body,m.body('basketball').id):
            jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,jr,c.pos,bid);velocities.append(jp@d.qvel)
        relative=velocities[1]-velocities[0];normal=c.frame[:3]
        slips.append(np.linalg.norm(relative-normal*np.dot(normal,relative)))
    return count,finger_count,force,f_force,depth,float(np.sqrt(np.mean(np.square(slips)))) if slips else 0.


class ContactTargets:
    """Goal adapter, not a controller replacement. No writes to fitted motion."""
    def __init__(self,bridge,stage,cup=.3,center_x=-.10,thumb=0.,mcp=None):
        self.bridge,self.stage,self.cup=bridge,stage,cup
        self.m=bridge.model;self.probe=mujoco.MjData(self.m)
        self.wrist=[int(self.m.joint('R_Wrist_'+a).qposadr[0]) for a in 'xyz']
        self.fingers=[int(self.m.jnt_qposadr[j]) for j in range(self.m.njnt) if is_finger(self.m.joint(j).name)]
        self.base=bridge.target(0)
        self.center_x=center_x
        self.thumb=thumb
        self.mcp=mcp

    def palm_up(self,q,actual=None):
        self.probe.qpos[:]=q;mujoco.mj_forward(self.m,self.probe)
        bid=self.m.body('R_Wrist').id
        current=self.probe.xmat[bid].reshape(3,3);normal=-current[:,1]
        alignment,_=Rotation.align_vectors([[0,0,1]],[normal])
        desired=alignment.as_matrix()@current
        parent=(actual if actual is not None else self.probe).xmat[self.m.body_parentid[bid]].reshape(3,3)
        angles=Rotation.from_matrix(parent.T@desired).as_euler('XYZ')
        alternate=np.array([angles[0]+np.pi,np.pi-angles[1],angles[2]+np.pi])
        alternate=(alternate+np.pi)%(2*np.pi)-np.pi
        reference=(actual.qpos if actual is not None else q)[self.wrist]
        # Equivalent XYZ solutions: select the one closest within the hinge limits.
        q[self.wrist]=min((angles,alternate),key=lambda a:np.linalg.norm(a-reference))
        return q

    def target(self,t,actual=None):
        if self.stage in ('motion','release'):
            q=self.bridge.target(max(0,t-1.))
        elif self.stage=='lift':
            s=np.clip((t-1)/2,0,1);s=s*s*(3-2*s)
            q=(1-s)*self.base+s*self.bridge.target(2.5)
        else:q=self.base.copy()
        if self.stage not in ('motion','release'):q=self.palm_up(q,actual)
        for f in FINGERS:
            for part in (1,2,3):
                # Anatomical curling direction from SMPL hand local bone axes: +Z.
                q[int(self.m.joint(f'R_{f}{part}_z').qposadr[0])]=(self.thumb if part==1 else 0.) if f=='Thumb' else self.cup*(.7 if part==3 else 1.)
                if f!='Thumb' and part==1 and self.mcp is not None:q[int(self.m.joint(f'R_{f}{part}_z').qposadr[0])]=self.mcp
        if self.stage=='release':
            a=np.clip((t-3.5)/.2,0,1);a=a*a*(3-2*a)
            q[self.fingers]*=1-a
        return q

    def control(self,data,target,disable_fingers=False):
        b=self.bridge
        b.data.qpos[:]=data.qpos[:b.model.nq];b.data.qvel[:]=data.qvel[:b.model.nv];b.data.time=data.time
        mujoco.mj_forward(b.model,b.data)
        tau,_=b.controller.control(b.data,target)
        if disable_fingers:
            for i in range(b.model.nu):
                if is_finger(b.model.actuator(i).name):tau[i]=0
        data.ctrl[:]=tau


def initialize_ball(m,d,center_x):
    wrist=m.body('R_Wrist').id;q=int(m.joint('basketball_free').qposadr[0]);v=int(m.joint('basketball_free').dofadr[0])
    # Start vertically above palm; suitable also for a tilted source wrist in D4/D5.
    center=d.xpos[wrist]+d.xmat[wrist].reshape(3,3)@np.array([center_x,0,-.007])
    lo,hi=.121,.35
    for _ in range(40):
        height=(lo+hi)/2;d.qpos[q:q+3]=center+[0,0,height];mujoco.mj_forward(m,d)
        bg=m.geom('basketball_geom').id
        overlap=any(bg in (c.geom1,c.geom2) and c.dist<0 for c in d.contact)
        if overlap:lo=height
        else:hi=height
    d.qpos[q:q+7]=np.r_[center+[0,0,hi+.0002],1,0,0,0];d.qvel[v:v+6]=0
    mujoco.mj_forward(m,d)
    return q,v


def simulate(args,body,stage):
    output=args.output/body/(stage+('_pad' if args.palm_pad else '')+('_no_pd' if args.no_finger_pd else ''));output.mkdir(parents=True,exist_ok=True)
    source=args.models/(body+'.xml');path=build_scene(source,output/'contact.xml',args.palm_pad,getattr(args,'soft_hand_contact',False))
    m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m)
    if getattr(args,'fine_step',False):m.opt.timestep=.0005
    bridge=HumanoidControllerBridge(source,m,args.source);adapter=ContactTargets(bridge,stage,args.cup,args.center_x,getattr(args,'thumb',0.),getattr(args,'mcp',None))
    bridge.initialize(m,d);d.qpos[:bridge.model.nq]=adapter.target(0);mujoco.mj_forward(m,d)
    # Pre-settle the humanoid alone; the measured experiment starts with ball on hand.
    warm=mujoco.MjData(bridge.model);warm.qpos[:]=d.qpos[:bridge.model.nq];mujoco.mj_forward(bridge.model,warm)
    for _ in range(round(getattr(args,'warmup',1.5)/m.opt.timestep)):
        adapter.control(warm,adapter.target(0,warm));mujoco.mj_step(bridge.model,warm);mujoco.mj_forward(bridge.model,warm)
    d.qpos[:bridge.model.nq]=warm.qpos
    # A stationary initial condition: zero humanoid velocity once before ball placement.
    d.qvel[:bridge.model.nv]=0;mujoco.mj_forward(m,d)
    q,v=initialize_ball(m,d,args.center_x)
    finger_dofs=[int(m.jnt_dofadr[j]) for j in range(m.njnt-1) if is_finger(m.joint(j).name)]
    initial=d.qpos[q:q+3].copy();initial_root=float(d.qpos[2]);rows=[];frames=[];states=[];times=[];contacts_seen=False;loss_start=None;loss_time=None;release_event=None;ball_free_time=None;release_candidate=None
    renderer=mujoco.Renderer(m,height=540,width=960) if args.gif else None
    viewer=None
    if args.gui:
        from mujoco import viewer as mv
        viewer=mv.launch_passive(m,d)
    cam=mujoco.MjvCamera();cam.lookat[:]=[.3,0,1.2];cam.distance=3.2;cam.azimuth=120;cam.elevation=-12
    if viewer:viewer.cam.lookat[:]=cam.lookat;viewer.cam.distance=cam.distance;viewer.cam.azimuth=cam.azimuth;viewer.cam.elevation=cam.elevation
    started=time.monotonic()
    sample_every=max(1,round(1/(30*m.opt.timestep)))
    for step in range(round(args.duration/m.opt.timestep)):
        target=adapter.target(d.time+m.opt.timestep,d)
        before=d.qpos[q:q+7].copy();before_v=d.qvel[v:v+6].copy()
        adapter.control(d,target,args.no_finger_pd)
        if not np.array_equal(before,d.qpos[q:q+7]) or not np.array_equal(before_v,d.qvel[v:v+6]):raise RuntimeError('Controller wrote ball state')
        mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():raise RuntimeError('Nonfinite physical state')
        n,nf,force,fforce,depth,slip=hand_contacts(m,d);t=float(d.time)
        wrist=m.body('R_Wrist').id;R=d.xmat[wrist].reshape(3,3)
        relative=R.T@(d.qpos[q:q+3]-d.xpos[wrist])
        contacts_seen=contacts_seen or (n>0 and t>=.2)
        if n>0:loss_start=None
        elif contacts_seen and loss_start is None and t>=.2:loss_start=t
        if loss_start is not None and t-loss_start>=.05 and loss_time is None:loss_time=loss_start
        if stage=='release' and t>=3.5 and n==0:
            if ball_free_time is None:
                ball_free_time=t;release_candidate={'time_s':t,'position_m':d.qpos[q:q+3].tolist(),'velocity_ms':d.qvel[v:v+3].tolist()}
            if t-ball_free_time>=.05 and release_event is None:
                release_event={**release_candidate,'confirmed_time_s':t,'valid_held_until_command':contacts_seen and (loss_time is None or loss_time>=3.5)}
        else:ball_free_time=None
        feet,margin,_=contacts(m,d);foot_slip,_=foot_contact_slip(m,d)
        human_mass=m.body_mass[1:bridge.model.nbody];human_com=human_mass@d.xipos[1:bridge.model.nbody]/human_mass.sum()
        com=(human_mass.sum()*human_com+.6*d.qpos[q:q+3])/(human_mass.sum()+.6)
        combined_margin=support_margin_at(m,d,com[:2])
        up=d.xmat[m.body('Pelvis').id].reshape(3,3)[:,1];tilt=float(np.degrees(np.arccos(np.clip(up[2],-1,1))))
        fallen=int(d.qpos[2]<.65*initial_root or tilt>35)
        row={'time_s':t,'hand_contacts':n,'finger_contacts':nf,'vertical_force_N':float(force[2]),'finger_vertical_force_N':float(fforce[2]),
             'contact_slip_ms':slip,'penetration_mm':float(depth*1000),'max_finger_torque_Nm':float(np.max(np.abs(d.qfrc_actuator[finger_dofs]))),
             'ball_x_m':float(d.qpos[q]),'ball_y_m':float(d.qpos[q+1]),'ball_z_m':float(d.qpos[q+2]),
             'ball_vx_ms':float(d.qvel[v]),'ball_vy_ms':float(d.qvel[v+1]),'ball_vz_ms':float(d.qvel[v+2]),
             'relative_x_m':float(relative[0]),'relative_y_m':float(relative[1]),'relative_z_m':float(relative[2]),
             'palm_z_m':float(d.xpos[wrist,2]),'palm_normal_up':float(-R[2,1]),
             'left_foot_force_N':feet['L'],'right_foot_force_N':feet['R'],'human_support_margin_m':margin,
             'human_ball_support_margin_m':combined_margin,'human_com_x_m':float(human_com[0]),'human_com_y_m':float(human_com[1]),
             'foot_slip_ms':float(np.sqrt((foot_slip['L']['rms_speed']**2+foot_slip['R']['rms_speed']**2)/2)),
             'fallen':fallen,'pelvis_tilt_deg':tilt}
        rows.append(row)
        if step%sample_every==0:
            states.append(d.qpos.copy());times.append(t)
            if renderer:
                renderer.update_scene(d,camera=cam)
                from PIL import Image,ImageDraw
                im=Image.fromarray(renderer.render());ImageDraw.Draw(im).text((10,10),f'{body} {stage} | t={t:.2f} | contacts={n} | Fz={force[2]:.2f}N',fill='yellow');frames.append(im)
            if viewer:
                if not viewer.is_running():break
                viewer.sync();time.sleep(max(0,t-(time.monotonic()-started)))
    if viewer:viewer.close()
    if renderer:
        renderer.close();frames[0].save(output/'motion.gif',save_all=True,append_images=frames[1:],duration=round(1000*sample_every*m.opt.timestep),loop=0)
    write_csv(output/'telemetry.csv',rows);np.savez_compressed(output/'states.npz',qpos=states,time_s=times)
    held=[r for r in rows if .2<=r['time_s']<min(3.,args.duration)];rel0=np.array([rows[0][f'relative_{a}_m'] for a in 'xyz'])
    supported=[r for r in held if r['time_s']<(loss_time if loss_time is not None else d.time)]
    summary={'body':body,'stage':stage,'palm_pad':args.palm_pad,'finger_PD_enabled':not args.no_finger_pd,'cup_rad':args.cup,'thumb_rad':getattr(args,'thumb',0.),'center_x_m':args.center_x,
        'duration_s':float(d.time),'warmup_s':getattr(args,'warmup',1.5),'mcp_override_rad':getattr(args,'mcp',None),
        'soft_hand_contact':getattr(args,'soft_hand_contact',False),'timestep_s':float(m.opt.timestep),
        'holding_success_time_s':max(0.,float(loss_time if loss_time is not None else d.time)-.2) if contacts_seen else 0.,
        'held_3s':bool(contacts_seen and (loss_time is None or loss_time>=3.2) and d.time>=3.2-1e-8),
        'first_sustained_contact_loss_s':loss_time,'max_hand_contacts':max(r['hand_contacts'] for r in rows),
        'contact_ratio':float(np.mean([r['hand_contacts']>0 for r in held])),'finger_contact_ratio':float(np.mean([r['finger_contacts']>0 for r in held])),
        'mean_vertical_force_N':float(np.mean([r['vertical_force_N'] for r in held])),'mean_finger_vertical_force_N':float(np.mean([r['finger_vertical_force_N'] for r in held])),
        'max_penetration_mm':max(r['penetration_mm'] for r in rows),'max_ball_height_gain_m':max(r['ball_z_m'] for r in rows)-initial[2],
        'max_palm_height_gain_m':max(r['palm_z_m'] for r in rows)-rows[0]['palm_z_m'],
        'relative_drift_rms_m':float(np.sqrt(np.mean([np.linalg.norm(np.array([r[f'relative_{a}_m'] for a in 'xyz'])-rel0)**2 for r in held]))),
        'supported_relative_drift_rms_m':float(np.sqrt(np.mean([np.linalg.norm(np.array([r[f'relative_{a}_m'] for a in 'xyz'])-rel0)**2 for r in supported]))) if supported else None,
        'supported_mean_vertical_force_N':float(np.mean([r['vertical_force_N'] for r in supported])) if supported else None,
        'contact_slip_rms_ms':float(np.sqrt(np.mean([r['contact_slip_ms']**2 for r in held if r['hand_contacts']])) ) if any(r['hand_contacts'] for r in held) else None,
        'both_feet_contact_ratio':float(np.mean([r['left_foot_force_N']>.1 and r['right_foot_force_N']>.1 for r in rows])),
        'min_human_support_margin_m':min((r['human_support_margin_m'] for r in rows if r['left_foot_force_N']+r['right_foot_force_N']>.1),default=-1.),
        'min_combined_support_margin_m':min((r['human_ball_support_margin_m'] for r in rows if r['hand_contacts'] and r['left_foot_force_N']+r['right_foot_force_N']>.1),default=-1.),
        'fall_count':sum(r['fallen'] and (i==0 or not rows[i-1]['fallen']) for i,r in enumerate(rows)),
        'warnings':[int(w.number) for w in d.warning],
        'release':release_event if release_event is not None and release_event['valid_held_until_command'] else None,
        'release_attempt':release_event,'neq':m.neq,
        'ball_state_writes_after_initialization':0,'gain_scale':bridge.controller.gain_scale}
    (output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary));return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,default=ROOT.parents[2]/'basketball-r1/SMPLSim/outputs/smplh_physics')
    p.add_argument('--source',type=Path,default=ROOT/'outputs/m3_spl_fitting/spl_P0002_T0049.npz')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/m4_2d_contact')
    p.add_argument('--body',default='body_01',choices=['body_01','body_02','body_03','body_04']);p.add_argument('--all',action='store_true')
    p.add_argument('--stage',default='hold',choices=['hold','lift','motion','release'])
    p.add_argument('--duration',type=float,default=3.5);p.add_argument('--cup',type=float,default=.3);p.add_argument('--center-x',type=float,default=-.10)
    p.add_argument('--warmup',type=float,default=1.5)
    p.add_argument('--thumb',type=float,default=0.)
    p.add_argument('--mcp',type=float,help='Separate proximal-finger goal for geometry diagnostics')
    p.add_argument('--soft-hand-contact',action='store_true',help='Critical damping only for new hand-ball contact pairs; environment unchanged')
    p.add_argument('--fine-step',action='store_true',help='2000 Hz contact timestep; original PD gains unchanged')
    p.add_argument('--palm-pad',action='store_true');p.add_argument('--no-finger-pd',action='store_true');p.add_argument('--gui',action='store_true');p.add_argument('--gif',action='store_true')
    p.add_argument('--diagnostic',action='store_true',help='Allow later-stage failure analysis when Holding gate is not passed')
    args=p.parse_args()
    if not np.isfinite([args.duration,args.cup,args.center_x,args.warmup,args.thumb]).all() or args.duration<.3 or not -.5<=args.cup<=1 or not 0<=args.warmup<=5 or abs(args.thumb)>1 or (args.mcp is not None and (not np.isfinite(args.mcp) or abs(args.mcp)>1)):p.error('Invalid experiment parameters')
    if args.stage!='hold' and not args.diagnostic:
        for body in ('body_01','body_02','body_03','body_04') if args.all else (args.body,):
            gate=args.output/body/('hold'+('_pad' if args.palm_pad else ''))/'summary.json'
            if not gate.exists() or not json.loads(gate.read_text())['held_3s']:
                p.error(body+' D2 Holding gate failed; use --diagnostic only for failure analysis')
    bodies=('body_01','body_02','body_03','body_04') if args.all else (args.body,)
    results=[simulate(args,b,args.stage) for b in bodies]
    write_csv(args.output/(args.stage+'_comparison.csv'),[{k:s[k] for k in ['body','stage','palm_pad','held_3s','holding_success_time_s','mean_vertical_force_N','contact_slip_rms_ms','max_ball_height_gain_m','both_feet_contact_ratio','fall_count']} for s in results])


if __name__=='__main__':main()
