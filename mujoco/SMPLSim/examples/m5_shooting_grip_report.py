"""Derived analysis/visuals for independent grip trials; input results are read only."""
import sys,json,argparse,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from examples.m5_shooting_grip_experiment import OUT,unit,contact_wrenches,maximum_gap,audit
from examples.m3_control import write_csv
MODEL=ROOT/'outputs/m5_1_upper_body/bimanual_contact.xml'


def table(path):return np.genfromtxt(path,names=True,delimiter=',',dtype=None,encoding='utf8')
def report():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);ref=dict(np.load(ROOT/'outputs/m5_1_coordinated_reference/reference/reference.npz'));f=ref['forward'];side=ref['side'];summaries=[];phase_rows=[];kinematic_rows=[];region_rows=[]
    for p in sorted(OUT.glob('*/summary.json')):
        if not (p.parent/'wrenches.csv').exists():continue
        s=json.loads(p.read_text());w=table(p.parent/'wrenches.csv');states=np.load(p.parent/'states.npz');q=table(p.parent/'joint_targets_torques.csv');s['max_contact_gap_ms']=maximum_gap([dict(time_s=float(x['time_s']),right_contacts=int(x['right_contacts'])) for x in w])*1000
        s['right_torque_peak_Nm']=max(float(np.max(np.abs(q[k]))) for k in q.dtype.names if k.startswith('R_') and k.endswith('_torque_Nm'));s['left_torque_peak_Nm']=max(float(np.max(np.abs(q[k]))) for k in q.dtype.names if k.startswith('L_') and k.endswith('_torque_Nm'))
        kin=[]
        for t,pose in zip(states['time_s'],states['qpos']):
            d.qpos[:]=pose;mujoco.mj_kinematics(m,d);bid=m.body('R_Wrist').id;W=d.xmat[bid].reshape(3,3);wp=d.xpos[bid];n=-W[:,1];fd=unit(d.xpos[m.body('R_Middle1').id]-wp);index=d.xpos[m.body('R_Index1').id]-wp;pinky=d.xpos[m.body('R_Pinky1').id]-wp;normal=unit(np.cross(index,pinky));normal*=1 if normal@n>=0 else -1
            eR=d.xmat[m.body('R_Elbow').id].reshape(3,3);arm=unit(wp-d.xpos[m.body('R_Elbow').id]);r=dict(name=s['name'],time_s=float(t),finger_forward_dot=float(fd@f),finger_up_dot=float(fd[2]),palm_up_dot=float(n[2]),metacarpal_up_dot=float(normal[2]),normal_proxy_error_deg=float(np.degrees(np.arccos(np.clip(normal@n,-1,1)))),forearm_axis_x=arm[0],forearm_axis_y=arm[1],forearm_axis_z=arm[2],wrist_x=wp[0],wrist_y=wp[1],wrist_z=wp[2])
            ball=pose[int(m.joint('basketball_free').qposadr[0]):][:3];r['wrist_behind_ball_mm']=float((wp-ball)@f*1000);r['wrist_below_ball_mm']=float((wp[2]-ball[2])*1000)
            kin.append(r);kinematic_rows.append(r)
        points=table(p.parent/'contact_points.csv')
        for side_name in ['R','L']:
            for geom in np.unique(points['geometry'][points['side']==side_name]):
                z=points[(points['side']==side_name)&(points['geometry']==geom)];normal=np.column_stack([z['normal_'+a] for a in 'xyz']);weight=z['normal_force_N'];region_rows.append(dict(name=s['name'],side=side_name,geometry=geom,contact_samples=len(z),mean_contact_normal_N=float(np.mean(weight)),lower_contact_weight_fraction=float(np.average(normal[:,2]>0,weights=weight)),behind_contact_weight_fraction=float(np.average(normal@f>0,weights=weight)),mean_normal_z=float(np.average(normal[:,2],weights=weight)),mean_normal_forward=float(np.average(normal@f,weights=weight))))
        s['finger_forward_final_dot']=kin[-1]['finger_forward_dot'];s['palm_up_final_dot']=kin[-1]['palm_up_dot'];s['wrist_behind_final_mm']=kin[-1]['wrist_behind_ball_mm']
        for phase in ['A','B','C']:
            z=w[w['phase']==phase]
            if not len(z):continue
            R=np.column_stack([z['R_force_'+a] for a in 'xyz']);L=np.column_stack([z['L_force_'+a] for a in 'xyz']);lf=L@side;rf=R@side;mask=z['left_contacts']>0
            row=dict(name=s['name'],phase=phase,samples=len(z),mean_R_vertical_N=float(np.mean(R[:,2])),mean_L_vertical_N=float(np.mean(L[:,2])),mean_total_vertical_N=float(np.mean(R[:,2]+L[:,2]+z['other_force_z'])),left_contact_ratio=float(np.mean(mask)),peak_left_normal_N=float(np.max(z['left_normal_N'])),right_contact_ratio=float(np.mean(z['right_contacts']>0)),drift_peak_mm=float(np.max(z['drift_mm'])),net_torque_rms_Nm=float(np.sqrt(np.mean(sum(z['net_torque_'+a]**2 for a in 'xyz')))),relative_speed_rms_m_s=float(np.sqrt(np.mean(z['relative_speed_m_s']**2))),mean_left_side_contact_N=float(np.mean(lf[mask])) if mask.any() else 0.,mean_right_side_at_left_contact_N=float(np.mean(rf[mask])) if mask.any() else 0.,mean_lateral_residual_at_left_contact_N=float(np.mean((lf+rf)[mask])) if mask.any() else 0.,counterforce_fraction=float(np.mean(lf[mask]*rf[mask]<0)) if mask.any() else 0.)
            phase_rows.append(row)
        summaries.append(s)
    flat=[{k:v for k,v in s.items() if not isinstance(v,(dict,list))} for s in summaries];keys=sorted(set().union(*(r.keys() for r in flat)));flat=[{k:r.get(k,'') for k in keys} for r in flat];write_csv(OUT/'comparison.csv',flat);write_csv(OUT/'phase_force_balance.csv',phase_rows);write_csv(OUT/'kinematics.csv',kinematic_rows);write_csv(OUT/'contact_regions.csv',region_rows);(OUT/'derived_summaries.json').write_text(json.dumps(summaries,indent=2))
    trials=[s for s in summaries if 'full' in s['name'] or 'guide' in s['name']];fig,axs=plt.subplots(5,1,figsize=(10,13),sharex=True,constrained_layout=True)
    for s in trials:
        w=table(OUT/s['name']/'wrenches.csv');name=s['name']
        for ax,key in zip(axs,['ball_height_m','drift_mm','R_force_z','left_normal_N']):ax.plot(w['time_s'],w[key],label=name,linewidth=.7,alpha=.8)
        axs[4].plot(w['time_s'],np.sqrt(sum(w['net_torque_'+a]**2 for a in 'xyz')),label=name,linewidth=.7)
    for ax,label in zip(axs,['Ball height (m)','Relative drift (mm)','Right vertical force (N)','Left normal load (N)','Ball contact torque (Nm)']):ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(fontsize=7)
    axs[-1].set_xlabel('Simulation time (s)');fig.savefig(OUT/'force_balance_comparison.png',dpi=130);plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(9,6),constrained_layout=True)
    for s in trials:
        rows=[r for r in kinematic_rows if r['name']==s['name']];t=[r['time_s'] for r in rows]
        for ax,key in zip(axs,['finger_forward_dot','palm_up_dot']):ax.plot(t,[r[key] for r in rows],label=s['name'])
    for ax,label in zip(axs,['Measured finger / rim dot','Palm normal / up dot']):ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(fontsize=7)
    axs[-1].set_xlabel('Simulation time (s)');fig.savefig(OUT/'hand_direction_comparison.png',dpi=130);plt.close(fig)
    layout(m,['A_full_v2','B_minus15_pitch_minus5','C_minus15_minus5_gentle'])
    (OUT/'gate.json').write_text(json.dumps(dict(large_ppo_allowed=False,physical_release_validated=False,reason='Only limited grip/reference feasibility; inspect all phase gates, contact force and actual orientation errors.'),indent=2));print(json.dumps(audit()))



def layout(m,names):
    import matplotlib.pyplot as plt
    from examples.m5_bimanual_contact import hand_geom
    fig=plt.figure(figsize=(15,5));d=mujoco.MjData(m)
    for i,name in enumerate(names):
        ss=np.load(OUT/name/'states.npz');k=int(np.argmin(abs(ss['time_s']-3.5)));d.qpos[:]=ss['qpos'][k];mujoco.mj_kinematics(m,d);ball=d.qpos[int(m.joint('basketball_free').qposadr[0]):][:3];ax=fig.add_subplot(1,3,i+1,projection='3d')
        u=np.linspace(0,2*np.pi,22);v=np.linspace(0,np.pi,12);ax.plot_wireframe(ball[0]+.121*np.outer(np.cos(u),np.sin(v)),ball[1]+.121*np.outer(np.sin(u),np.sin(v)),ball[2]+.121*np.outer(np.ones_like(u),np.cos(v)),color='orange',alpha=.4,linewidth=.5)
        for g in range(m.ngeom):
            n=m.geom(g).name or ''
            if not any(hand_geom(n,side) for side in ('R','L')):continue
            p=d.geom_xpos[g];R=d.geom_xmat[g].reshape(3,3);a=p-R[:,2]*m.geom_size[g,1];b=p+R[:,2]*m.geom_size[g,1];ax.plot(*np.array([a,b]).T,color='brown' if n.startswith('R_') else 'gray',linewidth=max(1,m.geom_size[g,0]*220))
        pts=table(OUT/name/'contact_points.csv');near=pts[abs(pts['time_s']-ss['time_s'][k])<.003]
        for row in near:
            pos=np.array([row['p_'+a] for a in 'xyz']);force=np.array([row['force_'+a] for a in 'xyz']);ax.quiver(*pos,*(force*.015),color='red',linewidth=.8)
        bid=m.body('R_Wrist').id;W=d.xmat[bid].reshape(3,3);p=d.xpos[bid];finger=unit(d.xpos[m.body('R_Middle1').id]-p);ax.quiver(*p,*(-W[:,1]*.1),color='blue');ax.quiver(*p,*(finger*.1),color='green')
        ax.set_xlim(ball[0]-.23,ball[0]+.23);ax.set_ylim(ball[1]-.23,ball[1]+.23);ax.set_zlim(ball[2]-.23,ball[2]+.23);ax.set_box_aspect([1,1,1]);ax.view_init(elev=20,azim=125);ax.set_title(name+' at 3.5s');ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)');ax.set_zlabel('World Z (m)')
    fig.suptitle('Actual FK capsules: blue palm proxy normal, green finger direction, red contact force');fig.tight_layout();fig.savefig(OUT/'grip_layout.png',dpi=135);plt.close(fig)

def gif(names):
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);renderer=mujoco.Renderer(m,400,420);cam=mujoco.MjvCamera();cam.lookat[:]=[.25,0,1.25];cam.distance=2.2;cam.azimuth=110;cam.elevation=-10;ss=[np.load(OUT/n/'states.npz') for n in names];frames=[]
    try:
        for i in range(0,max(len(s['qpos']) for s in ss),3):
            imgs=[]
            for name,s in zip(names,ss):
                j=min(i,len(s['qpos'])-1);d.qpos[:]=s['qpos'][j];mujoco.mj_forward(m,d);renderer.update_scene(d,camera=cam);im=Image.fromarray(renderer.render().copy());ImageDraw.Draw(im).text((8,8),f'{name}  {s["time_s"][j]:.2f}s',fill='white');imgs.append(np.asarray(im))
            frames.append(np.concatenate(imgs,axis=1))
    finally:renderer.close()
    destination=OUT/'grip_ABC.gif';imageio.mimsave(destination,frames,duration=100,loop=0);Image.fromarray(frames[min(35,len(frames)-1)]).save(OUT/'grip_ABC.png');print(destination)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gif',nargs='+');a=p.parse_args()
    if a.gif:gif(a.gif)
    else:report()
