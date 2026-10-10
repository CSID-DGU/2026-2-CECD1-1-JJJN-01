"""Plot/replay actual M4-2D-2 trajectories and audit preserved assets."""
import argparse,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from examples.m4_contact_lifting import ARM,PREVIOUS,hand
from examples.m3_control import write_csv


def table(out):
    m=mujoco.MjModel.from_xml_path(str(out/'contact.xml'));d=mujoco.MjData(m);A=out/'trials/fk_fit_2';states=np.load(A/'states.npz');w=m.body('R_Wrist').id
    x=np.genfromtxt(A/'telemetry.csv',delimiter=',',names=True);static=json.loads((A/'summary.json').read_text());zs=[];static_errors=[];static_rotations=[]
    inputs=json.loads((out/'static_inputs.json').read_text());original=Path(next(p for p in inputs if p.endswith('/body_01.xml')));source=Path(next(p for p in inputs if p.endswith('.npz')))
    bridge=hand.HumanoidControllerBridge(original,m,source);candidate=next(c for c in json.loads((PREVIOUS/'candidates.json').read_text()) if c['name']=='fk_fit_2');adapter=hand.GraspTargets(bridge,candidate)
    names=ARM+list(candidate['angles']);adrs=[int(m.joint(n).qposadr[0]) for n in names];R=np.load(out/'ik_path.npz')['rotation']
    for q,t in zip(states['qpos'],states['time_s']):
        d.qpos[:]=q;mujoco.mj_forward(m,d);zs.append(d.xpos[w,2])
        if t>=.2:
            goal=adapter.target(0,d);static_errors.append(d.qpos[adrs]-goal[adrs]);static_rotations.append(np.degrees(Rotation.from_matrix(R.T@d.xmat[w].reshape(3,3)).magnitude()))
    static_errors=np.array(static_errors);static['arm_tracking_rmse_rad']=float(np.sqrt(np.mean(static_errors[:,:9]**2)));static['max_orientation_error_deg']=max(static_rotations)
    write_csv(A/'per_joint_rmse.csv',[dict(joint=n,rmse_rad=float(np.sqrt(np.mean(static_errors[:,i]**2)))) for i,n in enumerate(names)])
    static.update(condition='A',success=static['held_3s'],wrist_rise_m=float(np.mean(zs[-15:])-np.mean(zs[:15])),ball_rise_m=float(np.mean(x['ball_z_m'][-225:])-np.mean(x['ball_z_m'][:225])),
                  first_contact_loss_s=static['first_loss_s'],max_contact_gap_s=static['max_contact_gap_after_settle_s'],mean_vertical_force_N=static['supported_force_z_N'],
                  slip_rms_ms=static['supported_slip_rms_ms'],both_feet_ratio=static['both_feet_ratio'])
    # Identical first stepped states: control treatment starts only at lift onset.
    common=states['qpos'][0]
    for c in ['B','C']:
        np.testing.assert_array_equal(np.load(out/c/'states.npz')['qpos'][0],common)
    summaries=[static]+[json.loads((out/c/'summary.json').read_text()) for c in ['B','C']]
    cols=['condition','success','wrist_rise_m','ball_rise_m','max_relative_drift_m','first_contact_loss_s','contact_ratio','max_contact_gap_s','mean_vertical_force_N','slip_rms_ms','active_contact_slip_rms_ms','arm_tracking_rmse_rad','max_orientation_error_deg','both_feet_ratio','min_com_margin_m','fallen']
    write_csv(out/'condition_comparison.csv',[{k:s.get(k,'') for k in cols} for s in summaries])
    fig,axes=plt.subplots(3,2,figsize=(12,11),constrained_layout=True)
    for c in ['B','C']:
        data=np.genfromtxt(out/c/'telemetry.csv',delimiter=',',names=True);base=data[(data['time_s']>=.9)&(data['time_s']<1)]
        axes[0,0].plot(data['time_s'],data['wrist_z_m']-base['wrist_z_m'].mean(),ls='--',label=c+' wrist');axes[0,0].plot(data['time_s'],data['ball_z_m']-base['ball_z_m'].mean(),label=c+' ball')
        for ax,key in zip([axes[0,1],axes[1,0],axes[1,1],axes[2,0],axes[2,1]],['force_z_N','drift_m','orientation_error_deg','contacts','slip_ms']):ax.plot(data['time_s'],data[key],label=c)
    labels=['Rise relative to pre-lift (m)','Vertical contact force (N)','Ball drift in wrist frame (m)','Wrist orientation error (degrees)','Active hand contacts','Contact slip (m/s)']
    for ax,label in zip(axes.flat,labels):ax.set_ylabel(label);ax.set_xlabel('Time (s)');ax.axvline(1,color='gray',ls=':');ax.axvline(5,color='gray',ls=':');ax.grid(alpha=.2);ax.legend()
    axes[0,1].axhline(.6*9.81,color='black',ls=':');fig.suptitle('B: frozen wrist angles | C: world wrist orientation maintained\n0.25m target, 4s slow lift, 600g contact-only basketball');fig.savefig(out/'lifting_comparison.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    for c in ['B','C']:
        rows=list(__import__('csv').DictReader((out/c/'per_joint_rmse.csv').open()));arm=[r for r in rows if r['joint'] in ARM];ax.plot(ARM,[float(r['rmse_rad']) for r in arm],'o-',label=c)
    ax.set_ylabel('Joint tracking RMSE (rad)');ax.tick_params(axis='x',rotation=35);ax.legend();ax.grid(alpha=.2);fig.savefig(out/'arm_tracking_rmse.png',dpi=150);plt.close(fig)
    # Sampled contacts exclude floor/rim/board as supporting forces in successful C.
    states=np.load(out/'C/states.npz');bad=[]
    for q in states['qpos']:
        d.qpos[:]=q;mujoco.mj_forward(m,d);ball=m.geom('basketball_geom').id
        for c in d.contact:
            if ball not in (c.geom1,c.geom2):continue
            name=m.geom(int(c.geom2 if c.geom1==ball else c.geom1)).name
            if name!='R_Wrist' and not hand.old.is_finger(name):bad.append(name)
    if bad:raise RuntimeError('Non-hand support contacts: '+str(bad))
    records=json.loads((out/'d1_preservation_before.json').read_text())
    records.update(json.loads((ROOT/'outputs/m4_2d_contact/preservation_before.json').read_text()))
    records.update(json.loads((PREVIOUS/'previous_results_hashes.json').read_text()))
    changed=[p for p,h in records.items() if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    audit=dict(checked_files=len(records),changed_files=changed,nonhand_ball_contacts_in_successful_C=bad,identical_initial_states_A_B_C=True)
    (out/'preservation_audit.json').write_text(json.dumps(audit,indent=2))
    if changed:raise RuntimeError(audit)
    print(json.dumps(audit))


def render(out,condition):
    from PIL import Image,ImageDraw
    m=mujoco.MjModel.from_xml_path(str(out/'contact.xml'));d=mujoco.MjData(m);folder=out/condition if condition!='A' else out/'trials/fk_fit_2'
    z=np.load(folder/'states.npz');rows=np.genfromtxt(folder/'telemetry.csv',delimiter=',',names=True);renderer=mujoco.Renderer(m,540,960);cam=mujoco.MjvCamera()
    for close in [False,True]:
        frames=[]
        for q,t in zip(z['qpos'],z['time_s']):
            d.qpos[:]=q;mujoco.mj_forward(m,d);cam.lookat[:]=[.25,0,1.10];cam.distance=2.4;cam.azimuth=110;cam.elevation=-10
            if close:cam.lookat[:]=(d.xpos[m.body('R_Wrist').id]+d.xpos[m.body('basketball').id])/2;cam.distance=.75;cam.elevation=-25
            renderer.update_scene(d,camera=cam)
            if close:
                for g in renderer.scene.geoms[:renderer.scene.ngeom]:
                    if g.objtype==mujoco.mjtObj.mjOBJ_GEOM and g.objid>=0:
                        name=m.geom(g.objid).name
                        if name not in ('R_Wrist','basketball_geom') and not hand.old.is_finger(name):g.rgba[3]=0
            r=rows[int(np.argmin(np.abs(rows['time_s']-t)))];force=r['force_z_N'];contacts=r['contacts'];im=Image.fromarray(renderer.render())
            ImageDraw.Draw(im).text((12,12),f'{condition} | t={t:.2f}s | contacts={contacts:g} | Fz={force:.2f}N',fill='yellow');frames.append(im)
        tag=condition+('_hand' if close else '_full');frames[0].save(out/(tag+'.gif'),save_all=True,append_images=frames[1:],duration=33,loop=0)
        frames[-1].save(out/(tag+'.png'))
    renderer.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/m4_2d_2_lifting');p.add_argument('--gif',choices=['A','B','C']);args=p.parse_args()
    if args.gif:render(args.output,args.gif)
    else:table(args.output)
