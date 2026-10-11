"""3C source/FK/physics comparison and recorded-state GIF rendering."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from examples import m4_forearm_rotation as exp
from examples.m3_control import write_csv

def report(out):
    s=np.load(out/'source_analysis.npz');t=s['times'];source=np.genfromtxt(out/'source_rotations.csv',delimiter=',',names=True);raw=np.genfromtxt(out/'original_spl_observations.csv',delimiter=',',names=True)
    fig,axs=plt.subplots(3,1,figsize=(11,10),sharex=True,constrained_layout=True)
    axs[0].plot(t,source['forearm_full_from_1p5_deg'],label='Full body orientation from 1.5s');axs[0].plot(t,source['bone_direction_from_1p5_deg'],label='Fitted elbow-wrist vector');axs[0].plot(t,raw['raw_bone_direction_from_1p5_deg'],'--',label='SPL measured limb vector');axs[0].plot(t,raw['raw_hand_plane_from_1p5_deg'],':',label='SPL hand plane (not forearm twist)');axs[0].set_ylabel('Change (deg)')
    for mode in ['A','B','C']:
        x=np.load(out/(mode+'_targets.npz'));axs[1].plot(t,np.degrees(s['roll']+x['twist_correction_rad']),label=mode)
    axs[1].set_ylabel('Transported material roll (deg)')
    for n in exp.NAMES:axs[2].plot(t,source[n+'_matrix_step_deg']*60,label=n)
    axs[2].set_ylabel('SO(3) step / dt (deg/s)');axs[2].set_xlabel('Source time (s)')
    for ax in axs:ax.set_xlim(1,2.5);ax.axvspan(1.5,2,color='orange',alpha=.1);ax.grid(alpha=.2);ax.legend()
    fig.savefig(out/'rotation_decomposition.png',dpi=150);plt.close(fig)
    comparison=[];positionrows=[];fig,axs=plt.subplots(3,1,figsize=(11,10),sharex=True,constrained_layout=True)
    for mode in ['A','B','C']:
        folder=out/mode/'C_1';summary=json.loads((folder/'summary.json').read_text());x=np.genfromtxt(folder/'telemetry.csv',delimiter=',',names=True);mask=x['time_s']>=3
        for ax,key in zip(axs,['drift_m','orientation_error_deg','force_z_N']):ax.plot(x['source_time_s'][mask],x[key][mask],label=mode)
        m=mujoco.MjModel.from_xml_path((folder/'scene_path.txt').read_text());d=mujoco.MjData(m);model=exp.dyn.prior.HumanoidControllerBridge(exp.ROOT.parents[2]/'basketball-r1/SMPLSim/outputs/smplh_physics/body_01.xml',m,exp.ROOT/'outputs/m3_spl_fitting/spl_P0002_T0049.npz');probe=mujoco.MjData(model.model)
        states=np.load(folder/'states.npz');joints=list(csv.DictReader((folder/'joint_tracking_torques.csv').open()));assert len(joints)==len(states['qpos'])
        records=[]
        for q,row in zip(states['qpos'],joints):
            time=float(row['time_s']);st=float(row['source_time_s'])
            if time<3:continue
            src=model.target(st);probe.qpos[:]=src;mujoco.mj_forward(model.model,probe);P=np.array([probe.xpos[model.model.body(n).id].copy() for n in exp.NAMES]);R=probe.xmat[model.model.body('R_Elbow').id].reshape(3,3).copy()
            goal=src.copy()
            for name in exp.dyn.lift.ARM+list(json.loads((exp.PREVIOUS/'selected.json').read_text())['angles']):goal[int(model.model.joint(name).qposadr[0])]=float(row[name+'_goal_rad'])
            probe.qpos[:]=goal;mujoco.mj_forward(model.model,probe);Pg=np.array([probe.xpos[model.model.body(n).id].copy() for n in exp.NAMES]);Rg=probe.xmat[model.model.body('R_Elbow').id].reshape(3,3).copy()
            d.qpos[:]=q;mujoco.mj_forward(m,d);Pa=np.array([d.xpos[m.body(n).id].copy() for n in exp.NAMES]);root_delta=q[:3]-src[:3]
            r=dict(condition=mode,time_s=time,source_time_s=st,forearm_goal_source_rotation_deg=np.degrees(Rotation.from_matrix(R.T@Rg).magnitude()))
            for k,n in enumerate(exp.NAMES):r[n+'_goal_source_error_mm']=np.linalg.norm(Pg[k]-P[k])*1000;r[n+'_actual_source_error_mm']=np.linalg.norm(Pa[k]-P[k])*1000;r[n+'_root_translation_removed_error_mm']=np.linalg.norm(Pa[k]-root_delta-P[k])*1000
            records.append(r);positionrows.append(r)
        focus=[r for r in records if 1<=r['source_time_s']<=2.4166666667];held=x[(x['time_s']>=3)&(x['time_s']<summary['first_loss_sim_s'])] if summary['first_loss_sim_s'] is not None else x[mask]
        common=x[(x['source_time_s']>=1)&(x['source_time_s']<=1.8)];focusx=x[(x['source_time_s']>=1.5)&(x['source_time_s']<=2.)]
        cross=x[(x['time_s']>=3)&(x['drift_m']>=.05)]
        comparison.append(dict(first_drift_50mm_source_s=float(cross['source_time_s'][0]) if len(cross) else None,max_pre_loss_drift_mm=float(max(held['drift_m'])*1000),common_window_orientation_rmse_deg=float(np.sqrt(np.mean(common['orientation_error_deg']**2))),focus_orientation_rmse_deg=float(np.sqrt(np.mean(focusx['orientation_error_deg']**2))),contact_ratio_1p5_2p0=float(np.mean(focusx['contacts']>0)),max_goal_step_rad=summary['max_goal_step_rad'],slip_rms_ms=summary['pre_candidate']['slip_rms_ms'],condition=mode,first_loss_source_s=summary['first_loss_source_s'],success=summary['pre_candidate']['success'],max_relative_drift_m=summary['pre_candidate']['max_relative_drift_m'],pre_loss_orientation_rmse_deg=np.sqrt(np.mean(held['orientation_error_deg']**2)),pre_loss_mean_force_z_N=np.mean(held['force_z_N']),goal_wrist_position_rmse_mm=np.sqrt(np.mean([r['R_Wrist_goal_source_error_mm']**2 for r in focus])),actual_wrist_position_rmse_mm=np.sqrt(np.mean([r['R_Wrist_actual_source_error_mm']**2 for r in focus])),root_translation_removed_wrist_rmse_mm=np.sqrt(np.mean([r['R_Wrist_root_translation_removed_error_mm']**2 for r in focus])),arm_tracking_rmse_rad=summary['pre_candidate']['arm_rmse_rad'],both_feet_ratio=summary['pre_candidate']['both_feet_ratio'],min_com_margin_m=summary['pre_candidate']['min_com_margin_m'],fallen=summary['pre_candidate']['fallen'],warnings=sum(summary['warnings'])))
    write_csv(out/'physics_comparison.csv',comparison);write_csv(out/'physics_position_errors.csv',positionrows)
    for ax,label in zip(axs,['Ball relative drift (m)','Wrist goal error (deg)','Vertical contact force (N)']):ax.set_ylabel(label);ax.set_xlim(1,2.5);ax.axvspan(1.5,2,color='orange',alpha=.1);ax.grid(alpha=.2);ax.legend()
    axs[0].set_ylim(0,.15);axs[-1].set_xlabel('Source time (s)');fig.savefig(out/'physics_comparison.png',dpi=150);plt.close(fig)
    prior=np.load(exp.PREVIOUS/'C_1/states.npz');baseline=np.load(out/'A/C_1/states.npz');same=bool(np.array_equal(prior['qpos'],baseline['qpos']))
    before=json.loads((out/'preservation_before.json').read_text());changed=[p for p,h in before.items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    assets=list(exp.PREVIOUS.glob('*.gif'))+[exp.ROOT.parents[2]/'SPL-Open-Data/basketball/freethrow/data/2025-12-18/P0002/BB_FT_P0002_T0049.json']
    audit=dict(checked_files=len(before),changed_files=changed,baseline_states_exactly_reproduced=same,read_only_asset_current_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in assets});(out/'preservation_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit));print(json.dumps(comparison,indent=2))
    if changed or not same:raise RuntimeError('Preservation or baseline reproduction failed')

def render(out):
    from PIL import Image,ImageDraw
    for mode in ['A','B','C']:
        folder=out/mode/'C_1';m=mujoco.MjModel.from_xml_path((folder/'scene_path.txt').read_text());d=mujoco.MjData(m);states=np.load(folder/'states.npz');summary=json.loads((folder/'summary.json').read_text());cam=mujoco.MjvCamera();renderer=mujoco.Renderer(m,540,960);frames=[]
        for q,t in zip(states['qpos'],states['time_s']):
            # Recorded-state visualisation only, never used in physics experiments.
            d.qpos[:]=q;mujoco.mj_forward(m,d);cam.lookat[:]=[.25,0,1.1];cam.distance=2.5;cam.azimuth=110;cam.elevation=-10;renderer.update_scene(d,camera=cam);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);draw.text((12,12),f'3C {mode} | source {max(0,t-3):.2f}s | no release command',fill='yellow')
            if summary['first_loss_sim_s'] is not None and t>=summary['first_loss_sim_s']:draw.text((12,32),'UNINTENDED CONTACT LOSS',fill='red')
            frames.append(im)
        frames[0].save(out/(mode+'.gif'),save_all=True,append_images=frames[1:],duration=33,loop=0);frames[145].save(out/(mode+'_preview.png'));renderer.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=exp.OUT);p.add_argument('--render',action='store_true');a=p.parse_args()
    if a.render:render(a.output)
    else:report(a.output)
if __name__=='__main__':main()
