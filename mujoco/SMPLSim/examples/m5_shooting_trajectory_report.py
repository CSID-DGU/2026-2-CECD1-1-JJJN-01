"""Read-only physical-state visualization and quantitative trajectory assessment."""
import sys,json,argparse,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,mujoco
from examples.m5_shooting_trajectory import OUT,audit_new
from examples.m5_bimanual_reorientation_report import arrows,camera,MODEL,table
from examples.m3_control import write_csv
from examples.m5_coordinated_reference import reserve
NAMES=['A_face','B_face','C_face','D_face']

def motion_camera():
    c=camera();c.lookat[:]=[.25,0,1.05];c.distance=3.0;return c

def run_folder(name):
    return ROOT/'outputs/m5_1_bimanual_reorientation/D_coupled' if name=='legacy_D' else OUT/name

def analyze():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summaries=[];jointrows=[];m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);ref=np.load(ROOT/'outputs/m5_1_coordinated_reference/reference/reference.npz')
    for path in sorted(OUT.glob('*/summary.json')):
        s=json.loads(path.read_text());p=table(path.parent/'physics.csv');q=table(path.parent/'joints.csv');dt=np.diff(p['time_s'],prepend=p['time_s'][0]);motion=p['phase']!='A';names=[n[:-7] for n in q.dtype.names if n.endswith('_target')]
        for n in names:
            error=q[n+'_actual']-q[n+'_target'];actual=q[n+'_actual'];vel=np.gradient(actual,q['time_s']);acc=np.gradient(vel,q['time_s']);jerk=np.gradient(acc,q['time_s']);jointrows.append(dict(run=s['name'],joint=n,rmse_rad=float(np.sqrt(np.mean(error**2))),torque_peak_Nm=float(max(abs(q[n+'_torque_Nm']))),velocity_peak_rad_s=float(max(abs(vel))),acceleration_peak_rad_s2=float(max(abs(acc))),jerk_rms_rad_s3=float(np.sqrt(np.mean(jerk**2)))))
        wanted=(np.gradient(p['rise_progress'],p['time_s'])>.01)&(np.gradient(p['rotation_progress'],p['time_s'])>.01)
        s['commanded_overlap_s']=float(np.sum(dt*wanted));s['max_right_orientation_error_deg']=float(max(p['wrist_orientation_goal_error_deg']));s['head_late_error_deg']=float(np.mean(p['head_error_deg'][-225:]));s['other_contact_force_peak_N']=float(max(p['other_contact_force_N']))
        s['initial_holding_gate_posthoc']=bool(np.max(p['ball_path_error_mm'][p['time_s']<1.5])<30 and not np.any(p['fallen']))
        s['final_R_contacts']=int(p['right_contacts'][-1]);s['final_L_contacts']=int(p['left_contacts'][-1]);s['observed_contact_loss_at_stop']=s['final_R_contacts']+s['final_L_contacts']==0
        contacts=(p['right_contacts']+p['left_contacts'])>0;lastcontact=np.maximum.accumulate(np.where(contacts,p['time_s'],0));s['max_both_contact_gap_s']=float(max(p['time_s']-lastcontact));s['first_sustained_contact_loss_s']=next((float(t) for t,g in zip(p['time_s'],p['time_s']-lastcontact) if g>.1),None)
        forcesR=np.column_stack([p['R_force_'+a] for a in 'xyz']);forcesL=np.column_stack([p['L_force_'+a] for a in 'xyz']);side=ref['side'];left=forcesL@side;right=forcesR@side;mask=p['left_contacts']>0
        s['guide_lateral_counterforce_ratio']=float(np.mean(left[mask]*right[mask]<0)) if mask.any() else None;s['contact_torque_rms_Nm']=float(np.sqrt(np.mean(sum(p['net_torque_'+a]**2 for a in 'xyz'))))
        states=np.load(path.parent/'states.npz');d.qpos[:]=states['qpos'][-1];mujoco.mj_forward(m,d);room=reserve(m,d,ref['rel0']);s['elbow_extension_reserve_deg']=float(room['elbow_extension_reserve_deg']);s['wrist_rim_reserve_rad']=float(room['wrist_rim_rotation_reserve_rad']);s['final_ball_head_height_mm']=float((d.qpos[int(m.joint('basketball_free').qposadr[0])+2]-d.xpos[m.body('Head').id,2])*1000)
        # No automatic naturalness claim from thresholds. Human validation remains required.
        s['shooting_ready_success']=False;s['natural_shooting_motion_verified']=False
        goal_height=float(ref['ball_goal_shoot'][2]-ref['ball_initial'][2]) if s['config'].get('face_path',False) else s['config']['height_m']
        s['requested_rise_mm']=goal_height*1000;s['raising_gate_pass']=bool(s['reason']=='sequence_completed' and s['ball_rise_mm']>goal_height*800 and max(p['ball_path_error_mm'][-225:])<50)
        s['full_path_within_50mm']=bool(max(p['ball_path_error_mm'])<50)
        summaries.append(s)
    (OUT/'evaluation.json').write_text(json.dumps(summaries,indent=2));write_csv(OUT/'joint_evaluation.csv',jointrows)
    keys=sorted(set().union(*({k for k,v in s.items() if not isinstance(v,(dict,list))} for s in summaries)));write_csv(OUT/'comparison.csv',[{k:s.get(k,'') for k in keys} for s in summaries])
    fig,axs=plt.subplots(4,2,figsize=(13,13),constrained_layout=True)
    keys=['ball_z','finger_change_deg','rise_progress','rotation_progress','R_force_z','L_force_z','wrist_extension_delta_deg','trunk_extension_delta_deg']
    labels=['Actual ball world height (m)','Actual finger direction change (deg)','Desired rise progress','Desired SO(3) turn progress','Right vertical contact force (N)','Left vertical contact force (N)','Geometric wrist extension component (deg)','Trunk straightening (deg)']
    for n in NAMES:
        p=table(OUT/n/'physics.csv')
        for ax,k,label in zip(axs.flat,keys,labels):ax.plot(p['time_s'],p[k],label=n);ax.set_ylabel(label);ax.set_xlabel('Physical time (s)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(OUT/'trajectory_forces.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for n in ['B_2s','B_3s','B_5s']:
        p=table(OUT/n/'physics.csv');axs[0].plot(p['time_s'],p['ball_z'],label=n);axs[1].plot(p['time_s'],p['ball_path_error_mm'],label=n);axs[2].plot(p['time_s'],p['right_slip_m_s']*1000,label=n)
    for ax,label in zip(axs,['Ball height (m)','World path error (mm)','Tangential contact slip (mm/s)']):ax.set_ylabel(label);ax.set_xlabel('Physical time (s)');ax.legend();ax.grid(alpha=.2)
    fig.savefig(OUT/'duration_comparison.png',dpi=140);plt.close(fig)
    print(json.dumps([dict(name=s['name'],rise_mm=s['ball_rise_mm'],stop=s['reason']) for s in summaries]));return summaries

def trajectory_lines(scene,m,states,index):
    adr=int(m.joint('basketball_free').qposadr[0]);points=states['qpos'][:index+1:3,adr:adr+3]
    for a,b in zip(points,points[1:]):
        if scene.ngeom>=scene.maxgeom:return
        g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([1,.6,0,1],np.float32));mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_LINE,2,a,b);scene.ngeom+=1

def gif(names,speed=1):
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);renderer=mujoco.Renderer(m,420,360);cam=motion_camera();series=[np.load(run_folder(n)/'states.npz') for n in names];frames=[]
    try:
        for t in np.arange(0,max(s['time_s'][-1] for s in series)+.05,.1*speed):
            images=[]
            for n,s in zip(names,series):
                i=int(np.argmin(abs(s['time_s']-t)));d.qpos[:]=s['qpos'][i];mujoco.mj_forward(m,d);renderer.update_scene(d,camera=cam);arrows(renderer.scene,m,d);trajectory_lines(renderer.scene,m,s,i);im=Image.fromarray(renderer.render().copy());draw=ImageDraw.Draw(im);draw.text((8,8),f'{n} physical t={s["time_s"][i]:.2f}s / {speed}x',fill='white');draw.text((8,25),'green: finger / blue: palm / orange: ball path',fill='white')
                if t>s['time_s'][-1]:draw.text((8,43),'FAILED / ENDED - frozen recorded state',fill='orange')
                images.append(np.asarray(im))
            frames.append(np.concatenate(images,axis=1))
        name='comparison_'+'_'.join(names)+f'_{speed:g}x';imageio.mimsave(OUT/(name+'.gif'),frames,duration=100,loop=0);Image.fromarray(frames[min(30,len(frames)-1)]).save(OUT/(name+'.png'))
    finally:renderer.close()

def replay(name,speed=1,seconds=None):
    import mujoco.viewer
    s=np.load(OUT/name/'states.npz');m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m)
    with mujoco.viewer.launch_passive(m,d) as v:
        c=motion_camera();v.cam.lookat[:]=c.lookat;v.cam.distance=c.distance;v.cam.azimuth=c.azimuth;v.cam.elevation=c.elevation;start=time.perf_counter()
        while v.is_running() and (seconds is None or time.perf_counter()-start<seconds):
            t=((time.perf_counter()-start)*speed)%s['time_s'][-1];i=int(np.argmin(abs(s['time_s']-t)))
            with v.lock():
                d.qpos[:]=s['qpos'][i];mujoco.mj_forward(m,d);v.user_scn.ngeom=0;arrows(v.user_scn,m,d);trajectory_lines(v.user_scn,m,s,i)
            v.sync();time.sleep(.01)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gif',nargs='+');p.add_argument('--replay');p.add_argument('--speed',type=float,default=1);p.add_argument('--seconds',type=float);a=p.parse_args()
    if not np.isfinite(a.speed) or a.speed<=0:raise ValueError('speed >0 required')
    if a.gif:gif(a.gif,a.speed)
    elif a.replay:replay(a.replay,a.speed,a.seconds)
    else:analyze();print(audit_new())
