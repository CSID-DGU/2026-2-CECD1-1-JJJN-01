"""Offline export and readiness analysis, never changes a running physical ball."""
import sys,json,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from examples.m5_coordinated_reference import OUT,unit,elbow_bend
from examples.m5_coordinated_eval import report,audit
from examples.m3_control import write_csv
from scipy.spatial.transform import Rotation
import mujoco,numpy as np


def main():
    model=mujoco.MjModel.from_xml_path(str(ROOT/'outputs/m5_1_upper_body/bimanual_contact.xml'));d=mujoco.MjData(model);diagnostics=[]
    for folder in sorted(OUT.iterdir()):
        if not (folder/'readiness.csv').exists():continue
        table=list(csv.DictReader((folder/'readiness.csv').open()));joints=list(csv.DictReader((folder/'joint_targets_torques.csv').open()));names=[k[:-14] for k in joints[0] if k.endswith('_reference_rad')]
        t=np.array([float(r['time_s']) for r in joints]);ref=np.array([[float(r[n+'_reference_rad']) for n in names] for r in joints]);target=np.array([[float(r[n+'_target_rad']) for n in names] for r in joints]);velocity=np.gradient(ref,t,axis=0);acc=np.gradient(velocity,t,axis=0)
        np.savez_compressed(folder/'reference_motion.npz',time_s=t,fps=30.,joint_names=names,reference_angles_rad=ref,controller_target_angles_rad=target,reference_velocity_rad_s=velocity,reference_acceleration_rad_s2=acc,format='MuJoCo hinge XYZ reference; not SMPL-H axis-angle; free root uncommanded')
        states=np.load(folder/'states.npz');trunk=[];forward_reserve=[]
        for q in states['qpos']:
            d.qpos[:]=q;mujoco.mj_kinematics(model,d)
            hips=np.mean([d.xpos[model.body(n).id] for n in ('L_Hip','R_Hip')],axis=0);shoulders=np.mean([d.xpos[model.body(n).id] for n in ('L_Shoulder','R_Shoulder')],axis=0);direction=unit(shoulders-hips);trunk.append(np.degrees(np.arccos(np.clip(direction[2],-1,1))))
        summary=json.loads((folder/'summary.json').read_text());physical=np.genfromtxt(folder/'physics.csv',delimiter=',',names=True);peak_left=float(np.max(physical['left_normal']));peak_drift=float(np.max(physical['drift_m'])*1000)
        diag=dict(name=folder.name,peak_left_normal_450Hz_N=peak_left,peak_drift_450Hz_mm=peak_drift,guide_whole_trial_force_budget_passed=peak_left<3,anatomical_trunk_tilt_final_deg=trunk[-1],anatomical_trunk_tilt_max_deg=max(trunk),max_reference_velocity_rad_s=float(np.max(abs(velocity))),max_reference_acceleration_rad_s2=float(np.max(abs(acc))),left_drift_push_max_N=max((float(r.get('left_drift_push_N',0)) for r in table),default=0),right_hand_setpoint_gate=summary['physical_readiness_success'],bimanual_physical_reference_gate=bool(summary['physical_readiness_success'] and summary['config']['guide']!='off' and summary['left_contact_fraction']>.01 and peak_left<3),release_success=False)
        if summary['probe_executed']:
            start=(summary['config'].get('prep_seconds',1.5)+10.)*summary['config']['duration_scale'];window=[r for r in table if float(r['time_s'])>=start];before=[r for r in table if start-.2<=float(r['time_s'])<start];v0=np.mean([float(r['ball_forward_velocity_m_s']) for r in before]);vmax=max(float(r['ball_forward_velocity_m_s']) for r in window);diag.update(probe_initial_forward_velocity_m_s=float(v0),probe_peak_forward_velocity_m_s=vmax,probe_peak_delta_forward_m_s=float(vmax-v0),probe_directional_acceleration_observed=bool(vmax-v0>.01))
        (folder/'readiness_diagnostic.json').write_text(json.dumps(diag,indent=2));diagnostics.append(diag)
    fields=list(dict.fromkeys(k for r in diagnostics for k in r));write_csv(OUT/'readiness_diagnostics.csv',[{k:r.get(k,'') for k in fields} for r in diagnostics])
    report()
    matched=dict(compared=False)
    if (OUT/'R_control_extended/states.npz').exists():
        a=np.load(OUT/'R_control/states.npz')['qpos'];b=np.load(OUT/'R_control_extended/states.npz')['qpos'][:len(a)];delta=float(np.max(abs(a-b)));matched=dict(compared=True,maximum_qpos_difference=delta,exact=delta==0,n_samples=len(a),note='450Hz dynamics unchanged by 30Hz FK telemetry cache')
    (OUT/'telemetry_equivalence.json').write_text(json.dumps(matched,indent=2))
    gate=dict(right_hand_setpoint_passed=any(d['right_hand_setpoint_gate'] for d in diagnostics),bimanual_shooting_reference_passed=any(d['bimanual_physical_reference_gate'] for d in diagnostics),large_ppo_allowed=False,physical_release_implemented=False);(OUT/'gate.json').write_text(json.dumps(gate,indent=2));print(json.dumps(gate));print(json.dumps(audit()))
if __name__=='__main__':main()
