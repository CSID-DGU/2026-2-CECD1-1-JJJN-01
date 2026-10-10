"""M4-2D-1 scientific plots and preservation audit; no simulation/model edits."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from examples.m3_control import write_csv
out=ROOT/'outputs/m4_2d_1_hand'

def main():
    data=np.genfromtxt(out/'axis_perturbations.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
    names=list(dict.fromkeys(r['joint'][:-2] for r in data));values=np.zeros((15,3))
    for r in data:
        if r['delta_rad']>0:values[names.index(r['joint'][:-2]),'xyz'.index(r['joint'][-1])]=r['tip_motion_mm']
    fig,ax=plt.subplots(figsize=(7,8),constrained_layout=True);im=ax.imshow(values,cmap='viridis');ax.set_xticks(range(3),['X','Y','Z']);ax.set_yticks(range(15),names)
    for i in range(15):
        for j in range(3):ax.text(j,i,f'{values[i,j]:.1f}',ha='center',va='center',color='white' if values[i,j]<values.max()*.6 else 'black')
    ax.set_title('Measured fingertip displacement after +0.2 rad FK (mm)');fig.colorbar(im,ax=ax,label='mm');fig.savefig(out/'fk_tip_displacement.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for folder,label in [('old_z_reference','Original Z curl'),('fk_fit_2','FK grasp + finger PD'),('fk_fit_2_no_pd','Same grasp, finger PD off')]:
        x=np.genfromtxt(out/'trials'/folder/'telemetry.csv',delimiter=',',names=True)
        for ax,key in zip(axes.flat,['ball_z_m','force_z_N','drift_m','contacts']):ax.plot(x['time_s'],x[key],label=label)
    for ax,title in zip(axes.flat,['Ball height (m)','Vertical hand contact force (N)','Ball drift in wrist frame (m)','Active hand contacts']):ax.set_ylabel(title);ax.set_xlabel('Time (s)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0,1].axhline(.6*9.81,ls=':',color='black');fig.suptitle('body_01 static contact-only Holding: 600 g, radius 121 mm');fig.savefig(out/'holding_comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    geom=np.genfromtxt(out/'trials/fk_fit_2/initial_geometry.csv',delimiter=',',names=True,dtype=None,encoding='utf-8');axes[0].barh(geom['geom'],geom['surface_gap_mm']);axes[0].set_xlabel('Initial sphere-capsule surface gap (mm)');axes[0].set_title('Geometric distances before gravity settling')
    con=np.genfromtxt(out/'trials/fk_fit_2/contacts.csv',delimiter=',',names=True,dtype=None,encoding='utf-8');con=con[(con['time_s']>=.2)&(con['normal_force_N']>.01)]
    for name in np.unique(con['geom']):
        c=con[con['geom']==name];axes[1].scatter(c['normal_x'],c['normal_y'],s=np.maximum(c['normal_force_N'],.01)*15,alpha=.4,label=name)
    axes[1].set_xlabel('Normal X (world)');axes[1].set_ylabel('Normal Y (world)');axes[1].set_title('Contact normals on ball; point area ~ normal force');axes[1].legend();axes[1].set_aspect('equal');axes[1].grid(alpha=.2)
    fig.savefig(out/'contact_geometry.png',dpi=150);plt.close(fig)
    thumb=np.genfromtxt(out/'thumb_opposition.csv',delimiter=',',names=True)
    fig,ax=plt.subplots(figsize=(7,5),constrained_layout=True);s=ax.scatter(thumb['palm_z_m']*1000,-thumb['palm_y_m']*1000,c=thumb['index_tip_distance_mm'],s=60,cmap='viridis')
    ax.set_xlabel('Across palm Z (mm)');ax.set_ylabel('Palm-facing height -Y (mm)');ax.set_title('Thumb proximal XYZ combinations: +/-0.35 rad');fig.colorbar(s,ax=ax,label='Thumb-index tip distance (mm)');fig.savefig(out/'thumb_opposition.png',dpi=150);plt.close(fig)
    summaries=[]
    for name in ['old_z_reference','fk_fit_2','fk_fit_2_no_pd']:
        s=json.loads((out/'trials'/name/'summary.json').read_text());summaries.append(s)
    write_csv(out/'holding_validation.csv',summaries)
    pergeom=[]
    for name in np.unique(con['geom']):
        c=con[con['geom']==name];pergeom.append(dict(geom=name,mean_normal_force_when_active_N=float(c['normal_force_N'].mean()),mean_normal_x=float(c['normal_x'].mean()),mean_normal_y=float(c['normal_y'].mean()),mean_normal_z=float(c['normal_z'].mean())))
    write_csv(out/'contact_normals_summary.csv',pergeom)
    files={**json.loads((ROOT/'outputs/m4_2d_contact/preservation_before.json').read_text()),**json.loads((out/'previous_results_hashes.json').read_text())}
    changed=[p for p,h in files.items() if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    audit={'checked_files':len(files),'changed_files':changed};(out/'preservation_audit.json').write_text(json.dumps(audit,indent=2))
    if changed:raise RuntimeError(audit)
    print(json.dumps(audit))

if __name__=='__main__':main()
