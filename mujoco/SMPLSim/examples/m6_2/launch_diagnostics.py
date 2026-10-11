"""Plots and invariant checks for timing/acceleration sweeps."""
import csv,json
import numpy as np
from examples.m6_2.release_acceleration import OUT,BASE,audit

def run():
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 r=json.loads((OUT/'comparison.json').read_text());pick=json.loads((OUT/'recommended.json').read_text())['condition'];base=np.load(BASE)
 for n,z in r.items():
  assert z['continuous_release_pass'] and z['ball_writes_after_reset']==0
  ref=np.load(z['reference']);assert np.isfinite(ref['qpos']).all()
  if z['group']=='timing_only':
   for k in base.files:
    if k!='release_time_s':np.testing.assert_array_equal(base[k],ref[k])
 eligible=[(n,z) for n,z in r.items() if z['pose_order_pass'] and z['stable_tracking_pass'] and z['forward_m_s']>0 and z['vertical_m_s']>0]
 assert eligible
 # Show high-speed horizontal/downward failures separately, not as success.
 names=['A_baseline','B_release_1.56',pick];fig,ax=plt.subplots(3,2,figsize=(12,10));metrics={}
 for n in names:
  rows=list(csv.DictReader((OUT/'runs'/n/'physics.csv').open()));t=np.array([float(x['time_s']) for x in rows]);z=r[n]
  for a,key in zip(ax.flat,['vx','vz','ball_z','R_elbow_bend_deg','R_knee_deg','body_rmse_rad']):
   a.plot(t,[float(x[key]) for x in rows],label=n);a.axvline(z['release']['actual_time_s'],ls='--',alpha=.3);a.set_title(key);a.set_xlabel('time (s)');a.grid(alpha=.2);a.legend(fontsize=7)
  metrics[n]={k:z[k] for k in ['release_speed_m_s','forward_m_s','lateral_m_s','vertical_m_s','release_angle_deg','first_flight_minimum_rim_distance_m','minimum_ballistic_speed_m_s','body_tracking_rmse_rad','falls','warnings','score','setpoint_elbow_bend_deg','setpoint_knee_bend_deg','setpoint_ball_height_m']};metrics[n]['release_time_s']=z['release']['actual_time_s'];metrics[n]['velocity_decomposition']=z['velocity_decomposition']
 fig.tight_layout();fig.savefig(OUT/'timing_acceleration_comparison.png');plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,6))
 for group in sorted(set(z['group'] for z in r.values())):
  v=[z for z in r.values() if z['group']==group];ax.scatter([z['forward_m_s'] for z in v],[z['vertical_m_s'] for z in v],label=group)
 ax.axhline(0,color='gray');ax.axvline(0,color='gray');ax.set(xlabel='towards hoop velocity (m/s)',ylabel='up velocity (m/s)');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'release_velocity_components.png');plt.close(fig)
 metrics['experiment_count']=len(r);metrics['recommended']=pick;metrics['all_continuous_release_pass']=True;metrics['timing_only_joint_targets_identical']=True;(OUT/'key_results.json').write_text(json.dumps(metrics,indent=2));audit();print(json.dumps(metrics,indent=2))

if __name__=='__main__':run()
