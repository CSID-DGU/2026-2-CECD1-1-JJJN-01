"""M4-2D-3A source/physics plots, failure diagnostics, GIF and preservation audit."""
import argparse,csv,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from examples.m4_spl_contact_integration import lift,geodesic_goal,smooth
from examples.m3_control import write_csv


def report(out):
    source=np.genfromtxt(out/'source_kinematics.csv',delimiter=',',names=True);a=json.loads((out/'source_analysis.json').read_text())
    fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True,constrained_layout=True)
    for side in ['R','L']:
        for part in ['Shoulder','Elbow','Wrist']:axes[0].plot(source['source_time_s'],source[side+'_'+part+'_z_m'],label=side+' '+part)
        axes[1].plot(source['source_time_s'],source[side+'_Wrist_vz_ms'],label=side+' wrist vertical speed')
    axes[2].plot(source['source_time_s'],source['R_Wrist_az_ms2'],label='R wrist vertical acceleration');ax2=axes[2].twinx();ax2.plot(source['source_time_s'],source['extension_rate_deg_s'],color='red',alpha=.6,label='Elbow extension rate');ax2.set_ylabel('Extension rate (deg/s)')
    for ax,label in zip(axes,['Height (m)','Vertical velocity (m/s)','Vertical acceleration (m/s2)']):
        ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(fontsize=8);ax.axvspan(*a['shooting_candidate_interval_s'],color='orange',alpha=.12)
    axes[-1].set_xlabel('SOURCE time (s), not simulation time');fig.suptitle('Fitted SPL body_01 FK: shooting region is an estimate; derivatives filtered only for analysis');fig.savefig(out/'source_motion_analysis.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(12,11),constrained_layout=True)
    summaries=[];failure=[]
    for name in ['B','C']:
        x=np.genfromtxt(out/name/'telemetry.csv',delimiter=',',names=True);s=json.loads((out/name/'summary.json').read_text())
        for ax,key in zip(axes.flat,['ball_z_m','ball_speed_ms','drift_m','source_orientation_error_deg','contacts','force_z_N']):ax.plot(x['time_s'],x[key],label=name)
        if s['first_contact_loss_s'] is not None:
            i=int(np.argmin(abs(x['time_s']-s['first_contact_loss_s'])));dt=float(np.median(np.diff(x['time_s'])));P=np.column_stack([x['wrist_'+k+'_m'] for k in 'xyz']);vel=np.gradient(P,dt,axis=0)
            for j in [max(0,i-23),i,min(len(x)-1,i+23)]:
                row={key:float(x[key][j]) for key in ['time_s','source_time_s','contacts','force_z_N','drift_m','palm_normal_up','prescribed_orientation_error_deg','source_orientation_error_deg','ball_speed_ms']};row.update(condition=name,wrist_speed_ms=float(np.linalg.norm(vel[j])),wrist_vertical_velocity_ms=float(vel[j,2]));failure.append(row)
            q=np.load(out/name/'states.npz');m=mujoco.MjModel.from_xml_path(str(out/'contact.xml'));ads=[int(m.joint(n).qposadr[0]) for n in lift.ARM];errors=list(csv.DictReader((out/name/'joint_tracking_torques.csv').open()));jumps=[]
            for k,n in enumerate(lift.ARM):
                goals=q['qpos'][:,ads[k]]-np.array([float(r[n+'_error_rad']) for r in errors]);dgoal=np.diff(goals);ind=int(np.argmax(abs(dgoal)));jumps.append(dict(joint=n,max_sampled_goal_step_rad=float(abs(dgoal[ind])),simulation_time_s=float(q['time_s'][ind+1])))
            write_csv(out/name/'goal_step_diagnostics.csv',jumps)
        for scope in ['entry','full']:summaries.append(dict(condition=name,scope=scope,first_loss_sim_s=s['first_contact_loss_s'],first_loss_source_s=s['source_time_at_loss_s'],**s[scope]))
    for ax,label in zip(axes.flat,['Ball height (m)','Ball speed (m/s)','Ball drift in wrist frame (m)','Raw source orientation error (deg)','Active contacts','Vertical hand contact force (N)']):
        ax.set_ylabel(label);ax.set_xlabel('SIMULATION time (s)');ax.axvline(1,color='gray',ls=':');ax.axvline(3,color='gray',ls=':');ax.axvline(3.5,color='gray',ls='--');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('B direct SPL | C bounded time-varying wrist correction\nEntry gate: blend and first 0.5s source; loss afterward still fails full playback');fig.savefig(out/'integration_comparison.png',dpi=150);plt.close(fig)
    write_csv(out/'integration_metrics.csv',summaries);write_csv(out/'contact_loss_diagnostics.csv',failure)
    caps=[]
    for name in ['C_limit_5','C_limit_10','C','C_limit_20']:
        s=json.loads((out/name/'summary.json').read_text());caps.append(dict(cap_deg=s['max_wrist_angle_deg'],entry_success=s['entry']['success'],full_success=s['full']['success'],first_loss_source_s=s['source_time_at_loss_s'],entry_max_drift_mm=s['entry']['max_relative_drift_m']*1000,entry_source_arm_rmse_rad=s['entry']['source_arm_rmse_rad']))
    write_csv(out/'wrist_limit_sensitivity.csv',caps)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True);axes[0].plot([s['cap_deg'] for s in caps],[s['first_loss_source_s'] for s in caps],'o-');axes[0].set_ylabel('First unintended contact loss (source s)');axes[1].plot([s['cap_deg'] for s in caps],[s['entry_source_arm_rmse_rad'] for s in caps],'o-');axes[1].set_ylabel('Raw source arm RMSE in entry (rad)')
    for ax in axes:ax.set_xlabel('Wrist orientation excursion cap (deg)');ax.grid(alpha=.2)
    fig.suptitle('All caps pass entry; all fail full playback');fig.savefig(out/'wrist_sensitivity.png',dpi=150);plt.close(fig)
    records=json.loads((out/'previous_hashes.json').read_text())
    records.update(json.loads((ROOT/'outputs/m4_2d_contact/preservation_before.json').read_text()));records.update(json.loads((ROOT/'outputs/m4_2d_1_hand/previous_results_hashes.json').read_text()))
    changed=[p for p,h in records.items() if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    A=np.load(out/'baseline/C/states.npz')['qpos'][0]
    for name in ['B','C']:np.testing.assert_array_equal(A,np.load(out/name/'states.npz')['qpos'][0])
    audit=dict(checked_files=len(records),changed_files=changed,identical_initial_states_A_B_C=True)
    (out/'preservation_audit.json').write_text(json.dumps(audit,indent=2))
    if changed:raise RuntimeError(audit)
    print(json.dumps(audit))


def render(out,name):
    from PIL import Image,ImageDraw
    folder=out/name;states=np.load(folder/'states.npz');rows=np.genfromtxt(folder/'telemetry.csv',delimiter=',',names=True);s=json.loads((folder/'summary.json').read_text())
    m=mujoco.MjModel.from_xml_path(str(out/'contact.xml'));d=mujoco.MjData(m);renderer=mujoco.Renderer(m,540,960);cam=mujoco.MjvCamera();frames=[]
    for q,t in zip(states['qpos'],states['time_s']):
        d.qpos[:]=q;mujoco.mj_forward(m,d);cam.lookat[:]=[.25,0,1.1];cam.distance=2.6;cam.azimuth=110;cam.elevation=-10;renderer.update_scene(d,camera=cam);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
        r=rows[np.argmin(abs(rows['time_s']-t))];draw.text((12,12),f'{name} | sim={t:.2f}s | source={r["source_time_s"]:.2f}s | contacts={r["contacts"]:g}',fill='yellow')
        if s['first_contact_loss_s'] and t>=s['first_contact_loss_s']:draw.text((12,32),'CONTACT LOST: failure, not a commanded release',fill='red')
        frames.append(im)
    frames[0].save(out/(name+'_motion.gif'),save_all=True,append_images=frames[1:],duration=33,loop=0);frames[min(90,len(frames)-1)].save(out/(name+'_entry.png'));renderer.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/m4_2d_3a_integration');p.add_argument('--gif',choices=['B','C']);args=p.parse_args()
    if args.gif:render(args.output,args.gif)
    else:report(args.output)
