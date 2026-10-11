"""M5-1 measured comparison, repeated evaluation, plots and offline GIF replay."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from examples.m5_bimanual_contact import OUT,derived_scene
from examples.m5_upper_body_ppo_env import UpperBodyResidualEnv
from examples.m5_upper_body_ppo_eval import evaluate,save
from examples.m3_control import write_csv
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
REFERENCE=OUT/'references/guide_safe/reference.npz'


def repeat_evaluation():
    from stable_baselines3 import PPO
    rows=[]
    for condition in 'ABCD':
        env=UpperBodyResidualEnv(condition=condition,stage=1,reference=REFERENCE,record=True)
        policy=PPO.load(OUT/'smoke'/f'{condition}_stage1_seed7/ppo_smoke.zip',device='cpu')
        for seed in (0,1,2):
            summary,terms=evaluate(env,policy,seed);save(env,OUT/'evaluations'/condition/f'seed{seed}',summary,terms);rows.append(summary)
        env.close();print('Repeated evaluation:',condition,flush=True)
    (OUT/'repeated_evaluation.json').write_text(json.dumps(rows,indent=2))
    write_csv(OUT/'repeated_evaluation.csv',rows)


def offline_orientation(folder):
    state=np.load(folder/'states.npz');m=mujoco.MjModel.from_xml_path(str(derived_scene().resolve()));d=mujoco.MjData(m);ref=np.load(REFERENCE);rows=[]
    for q,t in zip(state['qpos'],state['time_s']):
        d.qpos[:]=q;mujoco.mj_forward(m,d)
        rows.append(dict(time_s=float(t),right_world_orientation_error_deg=float(np.degrees(Rotation.from_matrix(ref['right_rotation'].T@d.xmat[m.body('R_Wrist').id].reshape(3,3)).magnitude()))))
    write_csv(folder/'world_orientation.csv',rows)
    return dict(right_world_orientation_rmse_deg=float(np.sqrt(np.mean([r['right_world_orientation_error_deg']**2 for r in rows]))),right_world_orientation_max_deg=max(r['right_world_orientation_error_deg'] for r in rows))


def report():
    folders=[OUT/'B_stage1_seed7',OUT/'feasibility/final_guide_neg002',OUT/'feasibility/guide_feedback',OUT/'feasibility/guide_safe',OUT/'feasibility/late_guide_shooting']
    summaries=[]
    for folder in folders:
        s=json.loads((folder/'summary.json').read_text());s['trial']=folder.name;s.update(offline_orientation(folder));summaries.append(s)
    write_csv(OUT/'feasibility_comparison.csv',summaries)
    smoke=[]
    for condition in 'ABCD':
        s=json.loads((OUT/'smoke'/f'{condition}_stage1_seed7/summary.json').read_text())
        for label in ('before','after'):
            r=dict(s[label],evaluation=label)
            # A is unilateral control: its historical `success` is not a bimanual gate.
            r['bimanual_stage_success']=bool(r['success'] and r['left_final_contact_ratio']>=.5)
            smoke.append(r)
    write_csv(OUT/'ppo_comparison.csv',smoke)
    gate=json.loads((OUT/'baseline_gate.json').read_text());late=json.loads((OUT/'late_guide_diagnostic.json').read_text())
    guide_passed=any(s['stage']==1 and s['condition']!='A' and s['success'] for s in summaries);shooting_passed=any(s['stage']>=4 and s['success'] for s in summaries)
    feasibility=dict(baseline_passed=gate['passed'],bimanual_holding_passed=guide_passed,shooting_reference_physics_passed=shooting_passed,large_curriculum_training_allowed=guide_passed and shooting_passed,ik_only_success=True,closest_pre_guide_goal_distance_mm=late['pre_left_min_goal_distance_mm'],reason='Guide holding failed; late-guide path missed 30mm goal and drift gate.')
    (OUT/'feasibility/final_gates.json').write_text(json.dumps(feasibility,indent=2))
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True,constrained_layout=True)
    for folder in folders[1:4]:
        x=np.genfromtxt(folder/'physics.csv',delimiter=',',names=True)
        axes[0].plot(x['time_s'],x['drift_m']*1000,label=folder.name)
        axes[1].plot(x['time_s'],x['left_normal'],label=folder.name)
        axes[2].plot(x['time_s'],x['ball_y_m']-x['ball_y_m'][0],label=folder.name)
    for ax,label in zip(axes,['Ball relative drift (mm)','Left normal force (N)','Ball lateral displacement (m)']):ax.set_ylabel(label);ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes[0].axhline(50,color='red',ls='--');axes[-1].set_xlabel('Simulation time (s)');fig.savefig(OUT/'guide_comparison.png',dpi=140);plt.close(fig)
    x=np.genfromtxt(OUT/'feasibility/late_guide_shooting/physics.csv',delimiter=',',names=True)
    fig,axes=plt.subplots(4,1,figsize=(10,10),sharex=True,constrained_layout=True)
    for ax,key,label,scale in zip(axes,['ball_z_m','goal_distance_m','drift_m','left_normal'],['Ball height (m)','Ball goal error (mm)','Relative drift (mm)','Left normal force (N)'],[1,1000,1000,1]):
        ax.plot(x['time_s'],x[key]*scale);ax.set_ylabel(label);ax.axvline(late['first_left_contact_s'],color='orange',ls='--',label='First physical left contact');ax.grid(alpha=.2)
    axes[1].axhline(30,color='green',ls='--');axes[2].axhline(50,color='red',ls='--');axes[0].legend();axes[-1].set_xlabel('Simulation time (s)');fig.savefig(OUT/'late_guide_shooting.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for condition in 'ABCD':
        r=list(csv.DictReader((OUT/'smoke'/f'{condition}_stage1_seed7/step_rewards.csv').open()));axes[0].plot([int(z['step']) for z in r],[float(z['reward']) for z in r],label=condition)
        r=next(x for x in smoke if x['condition']==condition and x['evaluation']=='after');axes[1].bar(condition,r['duration_s'])
    axes[0].set_xlabel('64-step PPO smoke');axes[0].set_ylabel('Reward');axes[0].legend();axes[1].set_ylabel('After-evaluation duration (s)');fig.savefig(OUT/'ppo_smoke_comparison.png',dpi=140);plt.close(fig)
    before=json.loads((OUT/'preservation_before.json').read_text());changed=[p for p,h in before.items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];metadata_changed=[p for p in changed if Path(p).name=='.DS_Store'];changed=[p for p in changed if Path(p).name!='.DS_Store'];audit=dict(checked_files=len(before),scientific_files_checked=sum(Path(p).name!='.DS_Store' for p in before),changed_files=changed,os_metadata_changed=metadata_changed);(OUT/'preservation_audit.json').write_text(json.dumps(audit,indent=2))
    if changed:raise RuntimeError('Protected files changed')
    print(json.dumps(audit),flush=True)


def gif():
    import imageio.v2 as imageio
    m=mujoco.MjModel.from_xml_path(str(derived_scene().resolve()));d=mujoco.MjData(m);renderer=mujoco.Renderer(m,400,640)
    camera=mujoco.MjvCamera();camera.lookat[:]=[.2,0,1.25];camera.distance=2.3;camera.azimuth=110;camera.elevation=-12
    state=np.load(OUT/'feasibility/late_guide_shooting/states.npz');frames=[]
    # Offline renderer probe only: no episode/controller/physics advancement.
    for q in state['qpos'][::6]:
        d.qpos[:]=q;mujoco.mj_forward(m,d);renderer.update_scene(d,camera=camera)
        for g in renderer.scene.geoms[:renderer.scene.ngeom]:
            if g.objtype==mujoco.mjtObj.mjOBJ_GEOM:
                name=m.geom(g.objid).name or ''
                if name.startswith('L_') and any(z in name for z in ['Wrist','Index','Middle','Pinky','Ring','Thumb']):g.rgba[:]=[.2,.6,1,1]
        frames.append(renderer.render().copy())
    renderer.close();imageio.mimsave(OUT/'late_guide_shooting_2x.gif',frames,duration=100,loop=0);print('GIF saved',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',action='store_true');p.add_argument('--gif',action='store_true');a=p.parse_args()
    if a.seeds:repeat_evaluation()
    if a.gif:gif()
    else:report()
if __name__=='__main__':main()
