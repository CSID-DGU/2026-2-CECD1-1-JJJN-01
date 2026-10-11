"""Read-only video decoding, sparse manual 2D annotation and conservative ball tracking."""
import sys,json,csv,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,cv2
from examples.m3_control import write_csv
OUT=ROOT/'outputs/m5_2_video_reference'
# Digitized visible near-side silhouette only. Clothing/occlusion prevents side identity.
# Raw image 926x1316; approximate manual uncertainty +/-20 px, NOT a pose estimator.
LANDMARKS={
 215:dict(shoulder=(506,846),elbow=(526,966),wrist=(426,1008),hip=(563,1024),knee=(521,1150),ankle=(576,1257),head=(482,752)),
 219:dict(shoulder=(506,842),elbow=(504,967),wrist=(374,994),hip=(563,1024),knee=(521,1150),ankle=(576,1257),head=(484,752)),
 223:dict(shoulder=(502,839),elbow=(412,923),wrist=(356,829),hip=(565,1040),knee=(511,1153),ankle=(575,1260),head=(494,753)),
 227:dict(shoulder=(487,798),elbow=(374,812),wrist=(438,689),hip=(555,1010),knee=(520,1150),ankle=(575,1260),head=(514,736)),
 231:dict(shoulder=(483,719),elbow=(385,726),wrist=(450,605),hip=(511,971),knee=(510,1124),ankle=(560,1250),head=(519,661)),
 235:dict(shoulder=(485,666),elbow=(382,620),wrist=(448,525),hip=(493,929),knee=(512,1100),ankle=(550,1230),head=(537,575)),
 239:dict(shoulder=(477,620),elbow=(425,547),wrist=(421,449),hip=(478,844),knee=(486,1010),ankle=(500,1170),head=(533,570)),
}
BALL_MANUAL={215:(388,1016,52),219:(339,991,51),223:(327,785,49),227:(450,658,47),231:(478,567,47),235:(464,477,48),239:(340,232,47)}

def analyze():
 v=json.loads((OUT/'video_access.json').read_text())['path'];cap=cv2.VideoCapture(v);rows=[];frames=[];previous=np.array([360.,970.])
 while True:
  ok,f=cap.read()
  if not ok:break
  i=len(rows);t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000;frames.append(f);hsv=cv2.cvtColor(f,cv2.COLOR_BGR2HSV);mask=((hsv[:,:,0]<12)|(hsv[:,:,0]>165))&(hsv[:,:,1]>65)&(hsv[:,:,2]>45);mask[:300]=False;mask[1150:]=False;mask[:,:220]=False;mask[:,680:]=False
  cs,_=cv2.findContours(mask.astype('uint8')*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);candidates=[]
  for c in cs:
   area=cv2.contourArea(c)
   if 250<area<13000:
    (x,y),r=cv2.minEnclosingCircle(c)
    if 13<r<75:candidates.append((np.linalg.norm(np.array([x,y])-previous),x,y,r,area))
  found=min(candidates) if candidates else None;valid=found is not None and found[0]<180
  if valid:_,x,y,r,area=found;previous=np.array([x,y])
  else:x=y=r=np.nan
  rows.append(dict(frame=i,pts_s=t,ball_x_px=x,ball_y_px=y,ball_radius_px=r,ball_detected=valid,confidence='medium_color_only' if valid else 'missing',overlay_occlusion=bool(.2<t<1.4 or 8.3<t<9.5)))
 write_csv(OUT/'ball_detector_rejected_raw.csv',rows)
 for r in rows:
  i=r['frame'];r['ball_detected']=i in BALL_MANUAL;r['ball_x_px'],r['ball_y_px'],r['ball_radius_px']=BALL_MANUAL.get(i,(np.nan,np.nan,np.nan));r['confidence']='manual_medium_uncertainty_10px' if i in BALL_MANUAL else 'unobserved';r['method']='sparse_manual_visual_digitization' if i in BALL_MANUAL else 'missing_rejected_automatic_color_track'
 write_csv(OUT/'ball_2d.csv',rows);write_csv(OUT/'frame_timestamps.csv',[dict(frame=r['frame'],pts_s=r['pts_s']) for r in rows])
 annotations=[];features=[]
 for i,points in LANDMARKS.items():
  for n,(x,y) in points.items():annotations.append(dict(frame=i,pts_s=rows[i]['pts_s'],landmark='visible_near_'+n,x_px=x,y_px=y,uncertainty_px=40,confidence='manual_medium' if n not in ('wrist','hip') else 'manual_low_clothing_occlusion',side_identity='unresolved_near_side',method='manual_visual_digitization'))
  hip=np.array(points['hip']);knee=np.array(points['knee']);ankle=np.array(points['ankle']);a=knee-hip;b=ankle-knee;bend=np.degrees(np.arccos(np.clip(a@b/np.linalg.norm(a)/np.linalg.norm(b),-1,1)));trunk=np.array(points['shoulder'])-hip;tilt=np.degrees(np.arctan2(abs(trunk[0]),-trunk[1]));features.append(dict(frame=i,pts_s=rows[i]['pts_s'],knee_bend_2d_deg=bend,trunk_tilt_2d_deg=tilt,pelvis_y_px=hip[1],wrist_y_px=points['wrist'][1],elbow_y_px=points['elbow'][1],ball_y_px=rows[i]['ball_y_px']))
 write_csv(OUT/'manual_landmarks_2d.csv',annotations);write_csv(OUT/'motion_features_2d.csv',features)
 # Frame boundaries visually bracketed from actual decoded frames, not nominal FPS.
 phases=[('preparation',191,217),('ball_raising',218,228),('set_point',229,232),('shooting_acceleration',233,238),('release_candidate',239,241),('follow_through',242,251)]
 write_csv(OUT/'observed_phases.csv',[dict(phase=n,start_frame=a,end_frame=b,start_pts_s=rows[a]['pts_s'],end_pts_s=rows[b]['pts_s'],uncertainty_frames=3,method='manual_visual_phase_boundary',release_ground_truth=False) for n,a,b in phases])
 for i in (215,219,223,227,231,235,239):
  f=frames[i].copy();p=LANDMARKS[i]
  for a,b in [('head','shoulder'),('shoulder','hip'),('shoulder','elbow'),('elbow','wrist'),('hip','knee'),('knee','ankle')]:cv2.line(f,p[a],p[b],(0,255,0),3)
  for n,xy in p.items():cv2.circle(f,xy,10,(0,255,255),2);cv2.putText(f,n,xy,cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),2)
  r=rows[i]
  if r['ball_detected']:cv2.circle(f,(round(r['ball_x_px']),round(r['ball_y_px'])),round(r['ball_radius_px']),(0,0,255),3)
  cv2.imwrite(str(OUT/f'annotated_{i:04d}.png'),f)
 metadata=dict(source=v,decoded_frames=len(rows),last_pts_s=rows[-1]['pts_s'],timestamp_basis='decoder presentation timestamps; variable capture cadence',representative_trial=[191,251],other_trials='First staged demonstration has long pauses; third has playback overlays. Neither is used for fitting.',not_observable=['unique 3D joints','wrist flexion/extension','forearm pronation/supination','individual fingers','far-side joint identity'],manual_uncertainty_px=40,body_tracking='7 sparse near-side manually digitized frames; no automatic pose model installed',ball_detection='Color tracker rejected after visual checks: court markings mistaken for ball. Final feature curves use 7 manually digitized ball centers/radii (+/-10px assumed reading uncertainty), not per-frame automated tracking.')
 (OUT/'video_analysis.json').write_text(json.dumps(metadata,indent=2));print(metadata)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,3,figsize=(13,4),constrained_layout=True);times=[x['pts_s'] for x in features]
 ax[0].plot(times,[-x['ball_y_px'] for x in features],'o-',label='ball height (pixels)');ax[0].plot(times,[-x['pelvis_y_px'] for x in features],'o-',label='pelvis height');ax[0].plot(times,[-x['wrist_y_px'] for x in features],'o-',label='wrist height');ax[1].plot(times,[x['knee_bend_2d_deg'] for x in features],'o-',label='near knee projected bend');ax[2].plot(times,[x['trunk_tilt_2d_deg'] for x in features],'o-',label='projected trunk tilt')
 for a in ax:a.set_xlabel('Source PTS (s)');a.legend();a.grid(alpha=.2)
 fig.savefig(OUT/'video_features.png',dpi=140);plt.close(fig)
if __name__=='__main__':analyze()
