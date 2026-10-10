"""M3-2A: identical fitted free throw on four existing XML bodies; no retargeting."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from examples.m3_control import write_csv
from examples.m3_spl_physics import run as physics_run, skeleton
from examples.smplh_bodies import SHAPES
from smpl_sim.motion.smplh_motion import SMPLHMotion


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def dimensions(models, model_dir):
    """SMPL mesh standing height; XML mass and joint-to-joint limb lengths."""
    import torch
    from smpl_sim.smpllib.smpl_parser import SMPLH_Parser
    torch.set_num_threads(1)
    parser = SMPLH_Parser(str(model_dir), gender='male', num_betas=16,
                          use_pca=False, create_transl=False)
    rows = []
    with torch.no_grad():
        for body, betas in SHAPES.items():
            m = mujoco.MjModel.from_xml_path(str(models/f'{body}.xml'))
            verts, _ = parser.get_joints_verts(torch.zeros(1,156), torch.tensor([betas], dtype=torch.float32))
            height = float(verts[0,:,1].max()-verts[0,:,1].min())
            row = {'body': body, 'mesh_height_m': height, 'mass_kg': float(m.body_mass.sum()),
                   'betas': json.dumps(betas), 'model_sha256': hashlib.sha256((models/f'{body}.xml').read_bytes()).hexdigest()}
            for side in ('L','R'):
                row[side+'_arm_length_m'] = float(sum(np.linalg.norm(m.body(side+'_'+n).pos) for n in ('Elbow','Wrist')))
                row[side+'_leg_length_m'] = float(sum(np.linalg.norm(m.body(side+'_'+n).pos) for n in ('Knee','Ankle')))
            row['arm_length_m'] = (row['L_arm_length_m']+row['R_arm_length_m'])/2
            row['leg_length_m'] = (row['L_leg_length_m']+row['R_leg_length_m'])/2
            rows.append(row)
    return rows


def compare(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summaries = [json.loads((output/body/'summary.json').read_text()) for body in SHAPES]
    rows, joints = [], []
    for s in summaries:
        body = s['body']
        row = {k:v for k,v in s.items() if k!='warnings'}
        row['mujoco_warning_count'] = sum(w['count'] for w in s['warnings'].values())
        # Equal-foot mean of loaded temporal mean-squared slip, not pooled contact forces.
        row['both_foot_slip_rms_m_s'] = float(np.sqrt((s['L_loaded_contact_slip_rms_m_s']**2+s['R_loaded_contact_slip_rms_m_s']**2)/2))
        rows.append(row)
        for j in read_csv(output/body/'joint_summary.csv'):
            joints.append({'body':body, **j})
    write_csv(output/'comparison_summary.csv',rows)
    write_csv(output/'joint_comparison.csv',joints)
    bodies = list(SHAPES)
    fig,axes = plt.subplots(2,2,figsize=(11,8))
    for ax,key,scale,label in zip(axes.flat,
        ('body_joint_rmse_rad','root_position_rmse_m','original_source_root_position_rmse_m','both_foot_slip_rms_m_s'),
        (1,1000,1000,1000),('Body tracking RMSE (rad)','Root RMSE: floor-aligned (mm)',
                           'Root RMSE: original source (mm)','Equal-foot loaded slip RMS (mm/s)')):
        values = [r[key]*scale for r in rows]
        ax.bar(bodies, values)
        ax.set_title(label);ax.grid(axis='y',alpha=.2)
        for i,v in enumerate(values):ax.text(i,v,f'{v:.4f}' if scale==1 else f'{v:.2f}',ha='center',va='bottom')
        ax.set_ylim(0,max(values)*1.2)
    fig.tight_layout();fig.savefig(output/'tracking_comparison.png',dpi=160);plt.close(fig)
    fig,axes = plt.subplots(3,1,figsize=(11,9),sharex=True)
    for body in bodies:
        ts = read_csv(output/body/'timeseries_450hz.csv')
        t = [float(r['time_s']) for r in ts]
        axes[0].plot(t,[float(r['body_joint_rmse_rad']) for r in ts],label=body)
        axes[1].plot(t,[float(r['root_position_error_m'])*1000 for r in ts],label=body)
        axes[2].plot(t,[float(r['support_margin_m'])*1000 if int(r['any_contact']) else np.nan for r in ts],label=body)
    for ax,label in zip(axes,('Body angle RMSE (rad)','Aligned root error (mm)','COM support margin (mm)')):
        ax.set_ylabel(label);ax.legend();ax.grid(alpha=.2)
    axes[-1].axhline(0,color='k',linewidth=.5);axes[-1].set_xlabel('Simulation time (s)')
    fig.tight_layout();fig.savefig(output/'timeseries_comparison.png',dpi=160);plt.close(fig)
    # Every hinge with same anatomical name, body joints only for readable heatmap.
    names=[j['joint'] for j in joints if j['body']=='body_01' and int(j['is_body'])]
    matrix=np.array([[float(next(j['rmse_rad'] for j in joints if j['body']==body and j['joint']==name)) for name in names] for body in bodies])
    fig,ax=plt.subplots(figsize=(17,4));im=ax.imshow(matrix,aspect='auto',cmap='viridis')
    ax.set_xticks(range(len(names)),names,rotation=90,fontsize=7);ax.set_yticks(range(4),bodies)
    fig.colorbar(im,ax=ax,label='Joint RMSE (rad)');fig.tight_layout()
    fig.savefig(output/'joint_rmse_heatmap.png',dpi=160);plt.close(fig)
    return rows


def render(args):
    """Render saved physical states; rendering does not re-run/alter physics."""
    from PIL import Image, ImageDraw
    motion = SMPLHMotion.load(args.source)
    snapshots = {}
    times = None
    for body in SHAPES:
        model = mujoco.MjModel.from_xml_path(str(args.models/f'{body}.xml'))
        data,ref=mujoco.MjData(model),mujoco.MjData(model)
        bound=motion.bind(model)
        with np.load(args.output/body/'physical_states.npz') as states:
            qpos,qvel,times=states['qpos'],states['qvel'],states['time_s']
            offset=float(states['floor_offset_m'])
            if str(states['source_sha256'])!=hashlib.sha256(args.source.read_bytes()).hexdigest():
                raise RuntimeError('Saved states refer to another motion')
            if str(states['model_sha256'])!=hashlib.sha256((args.models/f'{body}.xml').read_bytes()).hexdigest():
                raise RuntimeError('Saved states refer to another XML')
        camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
        camera.lookat[:]=[0,0,1.2];camera.distance=3.8;camera.azimuth=135;camera.elevation=-15
        selected={int(np.argmin(abs(times-t))):t for t in (0.,1.,2.,3.)}
        frames=[];snapshots[body]={}
        with mujoco.Renderer(model,height=480,width=640) as renderer:
            for i,t in enumerate(times):
                if i%2 and i not in selected:continue
                data.qpos[:]=qpos[i];data.qvel[:]=qvel[i];mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera=camera)
                if i in selected:
                    picture=Image.fromarray(renderer.render())
                    ImageDraw.Draw(picture).text((12,12),f'{body} | t={t:.6f}s | physics',fill='white',stroke_width=1,stroke_fill='black')
                    snapshots[body][selected[i]]=picture
                if i%2==0:
                    ref.qpos[:]=bound.sample(float(t));ref.qpos[2]+=offset;mujoco.mj_forward(model,ref)
                    skeleton(renderer.scene,model,ref)
                    picture=Image.fromarray(renderer.render())
                    ImageDraw.Draw(picture).text((12,12),f'{body} | t={t:.3f}s | green=reference',fill='white',stroke_width=1,stroke_fill='black')
                    frames.append(picture.quantize(colors=256))
        frames[0].save(args.output/body/'reference_vs_physics.gif',save_all=True,append_images=frames[1:],
                       duration=[30 if i%3<2 else 40 for i in range(len(frames))],loop=0,optimize=False,disposal=2)
        print(f'{body}: GIF rendered from saved states',flush=True)
    for t in (0.,1.,2.,3.):
        grid=Image.new('RGB',(1280,960))
        for i,body in enumerate(SHAPES):grid.paste(snapshots[body][t],((i%2)*640,(i//2)*480))
        grid.save(args.output/f'pose_comparison_{t:.0f}s.png')
    streams=[Image.open(args.output/body/'reference_vs_physics.gif') for body in SHAPES]
    frames=[]
    for f in range(min(s.n_frames for s in streams)):
        grid=Image.new('RGB',(1280,960))
        for i,s in enumerate(streams):s.seek(f);grid.paste(s.convert('RGB'),((i%2)*640,(i//2)*480))
        frames.append(grid.quantize(colors=256))
    frames[0].save(args.output/'four_body_comparison.gif',save_all=True,append_images=frames[1:],
                   duration=[30 if i%3<2 else 40 for i in range(len(frames))],loop=0,optimize=False,disposal=2)
    for s in streams:s.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'outputs/m3_spl_fitting/spl_P0002_T0049.npz')
    p.add_argument('--models',type=Path,default=ROOT.parents[2]/'basketball-r1/SMPLSim/outputs/smplh_physics')
    p.add_argument('--smpl-models',type=Path,default=ROOT.parents[2]/'basketball-r1/SMPLSim/assets/body_models/smplh')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/m3_body_baseline')
    p.add_argument('--render-only',action='store_true',help='Render existing saved physical states into GIFs and grids')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    before=hashlib.sha256(args.source.read_bytes()).hexdigest()
    motion=SMPLHMotion.load(args.source)
    if motion.fps!=60 or len(motion.poses)!=240:raise ValueError('Expected fitted 240-frame, 60 FPS motion')
    if args.render_only:
        render(args)
    else:
        for body in SHAPES:
            opts=SimpleNamespace(source=args.source,model=args.models/f'{body}.xml',output=args.output/body,
                                 body=body,root_mode='aligned',gui=False,video=False)
            physics_run(opts)
        write_csv(args.output/'body_dimensions.csv',dimensions(args.models,args.smpl_models))
        compare(args.output)
    if hashlib.sha256(args.source.read_bytes()).hexdigest()!=before:raise RuntimeError('Source NPZ changed')
    print('Baseline complete; source NPZ unchanged.',flush=True)


if __name__=='__main__':main()
