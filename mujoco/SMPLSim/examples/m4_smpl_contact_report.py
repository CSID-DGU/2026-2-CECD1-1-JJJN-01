"""Plot staged contact results and run regression with excluded XMLs referenced in memory."""
import argparse
import json
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from examples.m3_control import write_csv

def report(out):
    summaries=[];fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for body in ['body_01','body_02','body_03','body_04']:
        for condition in ['hold','hold_no_pd']:
            folder=out/body/condition
            s=json.loads((folder/'summary.json').read_text());summaries.append(s)
            x=np.genfromtxt(folder/'telemetry.csv',delimiter=',',names=True)
            if condition=='hold':
                t=x['time_s'];axes[0,0].plot(t,x['ball_z_m'],label=body)
                axes[0,1].plot(t,x['vertical_force_N'],label=body)
                axes[1,0].plot(t,x['hand_contacts'],label=body)
                axes[1,1].plot(t,x['contact_slip_ms'],label=body)
    columns=['body','stage','finger_PD_enabled','held_3s','holding_success_time_s','first_sustained_contact_loss_s','contact_ratio','finger_contact_ratio',
             'supported_mean_vertical_force_N','supported_relative_drift_rms_m','contact_slip_rms_ms','max_penetration_mm','both_feet_contact_ratio','min_human_support_margin_m','fall_count']
    write_csv(out/'four_body_holding.csv',[{k:s[k] for k in columns} for s in summaries])
    labels=['Ball height (m)','Hand vertical contact force (N)','Active hand contacts','Contact tangential slip (m/s)']
    for ax,label in zip(axes.flat,labels):ax.set_ylabel(label);ax.set_xlabel('Time (s)');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('SMPL+H contact Holding: failures are retained');fig.savefig(out/'four_body_holding.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True,constrained_layout=True)
    for stage in ['hold','lift','motion','release']:
        x=np.genfromtxt(out/'body_01'/stage/'telemetry.csv',delimiter=',',names=True)
        axes[0].plot(x['time_s'],x['ball_z_m'],label=stage+' ball')
        axes[0].plot(x['time_s'],x['palm_z_m'],ls='--',label=stage+' wrist')
        axes[1].plot(x['time_s'],x['palm_normal_up'],label=stage)
        axes[2].plot(x['time_s'],x['hand_contacts'],label=stage)
    for ax,label in zip(axes,['Height (m)','Palm normal dot world up','Contact points']):ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[2].set_xlabel('Time (s)');fig.suptitle('body_01 later stages: diagnostic only after failed D2')
    fig.savefig(out/'stage_diagnostics.png',dpi=140);plt.close(fig)

def regression(out):
    sys.path.insert(0,str(ROOT/'tests'))
    import test_smplh_motion
    test_smplh_motion.ROOT=ROOT.parents[2]/'basketball-r1/SMPLSim'
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    with (out/'regression_with_local_assets.log').open('w') as f:
        result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
    print((out/'regression_with_local_assets.log').read_text())
    return result.wasSuccessful()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/m4_2d_contact');p.add_argument('--regression',action='store_true')
    args=p.parse_args()
    if args.regression:raise SystemExit(0 if regression(args.output) else 1)
    report(args.output)
