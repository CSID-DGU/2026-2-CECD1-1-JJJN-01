"""M4-2D-3B physical diagnostics, geometry/contact plots, GIF and preservation audit."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from examples.m4_dynamic_grasp import OUT,lift,contact_records
from examples.m3_control import write_csv


def report(out):
    summaries=[];focus=[];forces=[];loss_rows=[];joint_summary=[]
    for path in sorted(out.glob('*_*/summary.json')):
        s=json.loads(path.read_text());folder=path.parent;x=np.genfromtxt(folder/'telemetry.csv',delimiter=',',names=True)
        if s['first_loss_sim_s'] is not None:
            i=int(np.argmin(abs(x['time_s']-s['first_loss_sim_s'])))
            for k in [max(0,i-23),i,min(len(x)-1,i+23)]:loss_rows.append(dict(run=folder.name,**{n:float(x[n][k]) for n in ['time_s','source_time_s','drift_m','contacts','force_z_N','orientation_error_deg','arm_rmse_rad','arm_torque_saturation_fraction','slip_ms']}))
        jr=list(csv.DictReader((folder/'joint_tracking_torques.csv').open()));jr=[r for r in jr if float(r['time_s'])>=1 and float(r['source_time_s'])<=2.4166666667]
        for name in [k[:-10] for k in jr[0] if k.endswith('_error_rad') and not k.endswith('_source_error_rad')]:
            joint_summary.append(dict(run=folder.name,joint=name,tracking_rmse_rad=float(np.sqrt(np.mean([float(r[name+'_error_rad'])**2 for r in jr]))),source_rmse_rad=float(np.sqrt(np.mean([float(r[name+'_source_error_rad'])**2 for r in jr]))),max_torque_Nm=max(abs(float(r[name+'_torque_Nm'])) for r in jr)))
        summaries.append(dict(run=folder.name,condition=s['condition'],speed=s['speed'],candidate=s['candidate'],first_loss_source_s=s['first_loss_source_s'],max_goal_step_rad=s['max_goal_step_rad'],max_goal_acceleration_rad_s2=s['max_goal_acceleration_rad_s2'],warning_count=sum(s['warnings']),max_left_force_N=s['max_left_contact_force_N'],guide_static_gate=s['guide_static_gate'],**s['pre_candidate']))
        for start,end in [(1.5,2.),(0.,2.4166666666666665)]:
            mask=(x['source_time_s']>=start)&(x['source_time_s']<=end)&(x['time_s']>=3);z=x[mask]
            if len(z):focus.append(dict(run=folder.name,start_source_s=start,end_source_s=end,contact_ratio=float(np.mean(z['contacts']>0)),max_drift_mm=float(max(z['drift_m'])*1000),arm_saturation_ratio=float(np.mean(z['arm_torque_saturation_fraction'])),orientation_rmse_deg=float(np.sqrt(np.mean(z['orientation_error_deg']**2))),source_arm_rmse_rad=float(np.sqrt(np.mean(z['source_arm_rmse_rad']**2))),mean_force_z_N=float(np.mean(z['force_z_N'])),slip_rms_ms=float(np.sqrt(np.mean(z['slip_ms']**2)))))
        if (folder/'contacts.csv').exists():
            rows=list(csv.DictReader((folder/'contacts.csv').open()));held=[r for r in rows if float(r['time_s'])>=1 and (s['first_loss_sim_s'] is None or float(r['time_s'])<s['first_loss_sim_s'])];states=np.load(folder/'states.npz');samples=sum(1 for t in states['time_s'] if t>=1 and (s['first_loss_sim_s'] is None or t<s['first_loss_sim_s']))
            for geom in sorted(set(r['geom'] for r in held)):
                g=[r for r in held if r['geom']==geom];N=np.array([float(r['normal_force_N']) for r in g]);normal=np.average([[float(r['normal_'+k]) for k in 'xyz'] for r in g],axis=0,weights=N)
                forces.append(dict(run=folder.name,geom=geom,mean_normal_force_N=float(N.sum()/max(samples,1)),mean_vertical_force_N=sum(float(r['force_z_N']) for r in g)/max(samples,1),mean_normal_x=normal[0],mean_normal_y=normal[1],mean_normal_z=normal[2],**{'mean_position_'+k+'_m':float(np.average([float(r['position_'+k+'_m']) for r in g],weights=N)) for k in 'xyz'}))
    write_csv(out/'loss_diagnostics.csv',loss_rows);write_csv(out/'per_joint_comparison.csv',joint_summary);write_csv(out/'comparison.csv',summaries);write_csv(out/'source_1p5_2p0_diagnostics.csv',focus);write_csv(out/'finger_contact_statistics.csv',forces)
    gates=json.loads((out/'candidate_gates.json').read_text());write_csv(out/'candidate_gate_comparison.csv',[{k:s.get(k) for k in ['candidate','static_gate','lifting_gate','max_relative_drift_m','lifting_relative_drift_m','supported_force_z_N','supported_finger_force_z_N','supported_slip_rms_ms']} for s in gates])
    fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True,constrained_layout=True)
    for condition in ['A','B','C','D']:
        folder=out/(condition+'_1')
        if not (folder/'telemetry.csv').exists():continue
        x=np.genfromtxt(folder/'telemetry.csv',delimiter=',',names=True);mask=(x['time_s']>=3)&(x['source_time_s']<=2.667)
        for ax,key in zip(axes,['drift_m','orientation_error_deg','force_z_N']):ax.plot(x['source_time_s'][mask],x[key][mask],label=condition)
    for ax,label in zip(axes,['Ball drift (m)','Wrist goal error (deg)','Right hand vertical force (N)']):ax.set_ylabel(label);ax.axvspan(1.5,2,color='orange',alpha=.1);ax.axvline(2.4166666667,color='gray',ls='--');ax.grid(alpha=.2);ax.legend()
    axes[-1].set_xlabel('Source time (s)');fig.suptitle('1x SPL dynamics | dashed line is a release proxy, not ground truth');fig.savefig(out/'dynamic_comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for cond in ['A','B','C','D']:
        r=sorted([s for s in summaries if s['condition']==cond],key=lambda s:s['speed']);axes[0].plot([s['speed'] for s in r],[s['first_loss_source_s'] if s['first_loss_source_s'] is not None else 3.98333 for s in r],'o-',label=cond);axes[1].plot([s['speed'] for s in r],[s['source_arm_rmse_rad'] for s in r],'o-',label=cond)
    for ax in axes:ax.set_xlabel('Playback speed');ax.grid(alpha=.2);ax.legend()
    axes[0].axhline(2.41666667,color='gray',ls='--');axes[0].set_ylabel('First contact loss (source s); 3.983 = no loss');axes[1].set_ylabel('Raw-source arm RMSE before candidate (rad)');fig.savefig(out/'speed_sensitivity.png',dpi=150);plt.close(fig)
    # Actual settled FK geometry and force-bearing contacts, not anatomy from names.
    selected=json.loads((out/'selected.json').read_text())['name'];fig=plt.figure(figsize=(12,6),constrained_layout=True);geometry=[]
    for i,name in enumerate(['fk_fit_2',selected]):
        folder=out/'candidates'/name;states=np.load(folder/'trials'/name/'states.npz');m=mujoco.MjModel.from_xml_path(str(folder/'contact.xml'));d=mujoco.MjData(m);d.qpos[:]=states['qpos'][min(30,len(states['qpos'])-1)];mujoco.mj_forward(m,d)
        q=int(m.joint('basketball_free').qposadr[0]);center=d.qpos[q:q+3];ax=fig.add_subplot(1,2,i+1,projection='3d')
        u=np.linspace(0,2*np.pi,30);v=np.linspace(0,np.pi,20);ax.plot_surface(center[0]+.121*np.outer(np.cos(u),np.sin(v)),center[1]+.121*np.outer(np.sin(u),np.sin(v)),center[2]+.121*np.outer(np.ones_like(u),np.cos(v)),alpha=.18,color='orange')
        for rec in lift.hand.distances(m,d,center):
            g=m.geom(rec['geom']).id;a,b,r=lift.hand.capsule(m,d,g);ax.plot(*np.array([a,b]).T,lw=max(2,r*350),label=rec['geom'] if rec['geom']=='R_Wrist' else None)
            geometry.append(dict(candidate=name,**rec))
        for r in contact_records(m,d):
            p=np.array([r['position_'+k+'_m'] for k in 'xyz']);n=np.array([r['normal_'+k] for k in 'xyz']);ax.scatter(*p,color='red');ax.quiver(*p,*(.035*n),color='red')
        ax.set_xlim(center[0]-.16,center[0]+.16);ax.set_ylim(center[1]-.16,center[1]+.16);ax.set_zlim(center[2]-.17,center[2]+.15);ax.set_box_aspect([1,1,1]);ax.set_title(name+' settled capsule geometry');ax.set_xlabel('world X (m)');ax.set_ylabel('world Y (m)');ax.set_zlabel('world Z (m)');ax.view_init(20,110)
    fig.savefig(out/'grasp_geometry_comparison.png',dpi=150);plt.close(fig);write_csv(out/'settled_geometry.csv',geometry)
    guide_geometry=[]
    for name in ['E_1','F_1']:
        folder=out/name
        if not (folder/'states.npz').exists():continue
        m=mujoco.MjModel.from_xml_path((folder/'scene_path.txt').read_text());d=mujoco.MjData(m);ball=m.geom('basketball_geom').id;depth=0.
        for q in np.load(folder/'states.npz')['qpos']:
            d.qpos[:]=q;mujoco.mj_forward(m,d)
            for ct in d.contact:
                if ball not in (ct.geom1,ct.geom2):continue
                other=int(ct.geom2 if ct.geom1==ball else ct.geom1)
                if m.geom(other).name.startswith('L_'):depth=max(depth,-float(ct.dist))
        guide_geometry.append(dict(run=name,max_sampled_left_penetration_mm=depth*1000))
    (out/'guide_geometry_audit.json').write_text(json.dumps(guide_geometry,indent=2))
    before=json.loads((out/'preservation_before.json').read_text());before.update(json.loads((ROOT/'outputs/m4_2d_contact/preservation_before.json').read_text()));before.update(json.loads((ROOT/'outputs/m4_2d_1_hand/previous_results_hashes.json').read_text()));changed=[p for p,h in before.items() if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];audit=dict(checked_files=len(before),changed_files=changed);(out/'preservation_audit.json').write_text(json.dumps(audit,indent=2))
    if changed:raise RuntimeError(audit)
    print(json.dumps(audit))


def render(out,name):
    from PIL import Image,ImageDraw
    folder=out/name;scene=Path((folder/'scene_path.txt').read_text());m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m);states=np.load(folder/'states.npz');s=json.loads((folder/'summary.json').read_text());cam=mujoco.MjvCamera();renderer=mujoco.Renderer(m,540,960);frames=[]
    stride=max(1,round(1/s['speed']))
    for q,t in zip(states['qpos'][::stride],states['time_s'][::stride]):
        d.qpos[:]=q;mujoco.mj_forward(m,d);cam.lookat[:]=[.25,0,1.1];cam.distance=2.5;cam.azimuth=110;cam.elevation=-10;renderer.update_scene(d,camera=cam);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);draw.text((12,12),f'{name} | simulation {t:.2f}s | contact-only, no release command',fill='yellow')
        if s['first_loss_sim_s'] is not None and t>=s['first_loss_sim_s']:draw.text((12,32),'UNINTENDED CONTACT LOSS: not successful release',fill='red')
        frames.append(im)
    frames[0].save(out/(name+'.gif'),save_all=True,append_images=frames[1:],duration=33*stride,loop=0);frames[min(90,len(frames)-1)].save(out/(name+'_preview.png'));renderer.close()


def render_grasps(out):
    from PIL import Image,ImageDraw
    selected=json.loads((out/'selected.json').read_text())['name'];images=[]
    for name in ['fk_fit_2',selected]:
        folder=out/'candidates'/name;m=mujoco.MjModel.from_xml_path(str(folder/'contact.xml'));d=mujoco.MjData(m);states=np.load(folder/'trials'/name/'states.npz');d.qpos[:]=states['qpos'][30];mujoco.mj_forward(m,d);q=int(m.joint('basketball_free').qposadr[0]);cam=mujoco.MjvCamera();cam.lookat[:]=d.qpos[q:q+3];cam.distance=.65;cam.azimuth=125;cam.elevation=-15
        renderer=mujoco.Renderer(m,540,960);renderer.update_scene(d,camera=cam);im=Image.fromarray(renderer.render());ImageDraw.Draw(im).text((12,12),name+' | saved static Holding, original capsules',fill='yellow');images.append(im);renderer.close()
    canvas=Image.new('RGB',(1920,540));canvas.paste(images[0],(0,0));canvas.paste(images[1],(960,0));canvas.save(out/'grasp_visual_comparison.png')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--gif');p.add_argument('--grasps',action='store_true');args=p.parse_args()
    if args.grasps:render_grasps(args.output)
    elif args.gif:render(args.output,args.gif)
    else:report(args.output)
