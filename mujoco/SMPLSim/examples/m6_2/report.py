"""Recompute diagnostics and an honest M6-2 experiment report."""
import json,csv
import numpy as np,mujoco
from examples.m6_2.experiment import OUT,ROOT,model,initial,geometry,unit,audit,RIM_CENTER
from examples.m3_control import write_csv

def report():
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 results=json.loads((OUT/'comparison.json').read_text());best='G_stable_0.14_0.8_0.65';z=results[best];extra=[]
 for n,r in results.items():
  p=OUT/'runs'/n;rows=list(csv.DictReader((p/'physics.csv').open()));t=np.array([float(x['time_s']) for x in rows]);q=np.array([[float(x[k]) for k in ['ball_x','ball_y','ball_z']] for x in rows]);v=np.array([[float(x[k]) for k in ['vx','vy','vz']] for x in rows]);first=t>=r['release']['actual_time_s'];hit=np.flatnonzero(first&(q[:,2]<.14)&(v[:,2]<0));end=hit[0] if len(hit) else len(t);valid=first&(np.arange(len(t))<end);r['first_flight_minimum_rim_distance_m']=float(np.linalg.norm(q[valid]-np.array([4.57,0,3.04]),axis=1).min()) if valid.any() else None
  if valid.any():r['first_flight_minimum_rim_distance_m']=float(np.linalg.norm(q[valid]-RIM_CENTER,axis=1).min())
  s=np.load(p/'states.npz');kin=list(csv.DictReader((p/'kinematics.csv').open()));ri=int(np.argmin(abs(t-r['release']['actual_time_s'])));r['release_ball_relative_head_m']=(q[ri]-[float(kin[ri]['Head_'+a]) for a in 'xyz']).tolist();ref=np.load(r['reference']);targetvel=np.gradient(s['target_qpos'][:,7:160],s['time_s'],axis=0)
  m=model();d=mujoco.MjData(m);d.qpos[:]=s['qpos'][0,:160];mujoco.mj_forward(m,d);r['observed_target_max_velocity_rad_s']=float(abs(targetvel).max());r['wrist_proxy_start_deg']=float(rows[0]['wrist_extension_proxy_deg']);r['wrist_proxy_release_deg']=float(rows[ri]['wrist_extension_proxy_deg']);r['wrist_proxy_range_deg']=[min(float(x['wrist_extension_proxy_deg']) for x in rows),max(float(x['wrist_extension_proxy_deg']) for x in rows)]
  if 'push_start' in ref:
   r['pocket_to_planned_setpoint_s']=float(ref['raise_s']);r['planned_push_to_release_s']=float(r['release']['actual_time_s']-ref['push_start']);r['planned_rise_push_overlap_s']=float(max(0,.18+float(ref['raise_s'])-float(ref['push_start'])))
  extra.append(dict(condition=n,**{k:r.get(k) for k in ['release_ball_relative_head_m','first_flight_minimum_rim_distance_m','observed_target_max_velocity_rad_s','wrist_proxy_start_deg','wrist_proxy_release_deg','pocket_to_planned_setpoint_s','planned_push_to_release_s','planned_rise_push_overlap_s']}))
 (OUT/'comparison.json').write_text(json.dumps(results,indent=2));write_csv(OUT/'phase_metrics.csv',extra)
 # Measured world axes in the actual starting pose, with sign-neutral geometric roles.
 m=model();d=mujoco.MjData(m);d.qpos[:]=initial();mujoco.mj_forward(m,d);offset=np.load(OUT/'references/B_pose.npz')['offset'];p,W,f=geometry(m,d,offset);normal=-W[:,1];flex=unit(np.cross(f,normal));arm=unit(d.xpos[m.body('R_Wrist').id]-d.xpos[m.body('R_Elbow').id]);roles=[]
 for n in [f'R_{b}_{a}' for b in ('Elbow','Wrist') for a in 'xyz']:
  axis=d.xaxis[m.joint(n).id];scores={'flexion_extension':abs(axis@flex),'axial_twist':abs(axis@arm),'radial_ulnar_proxy':abs(axis@normal)};roles.append(dict(joint=n,**scores,dominant_geometric_role=max(scores,key=scores.get)))
 write_csv(OUT/'axis_roles.csv',roles)
 fig=plt.figure(figsize=(12,5));a=fig.add_subplot(121,projection='3d');b=fig.add_subplot(122)
 for n in ['A_old',best,'H_fast_0.65']:
  rows=list(csv.DictReader((OUT/'runs'/n/'physics.csv').open()));t=np.array([float(x['time_s']) for x in rows]);q=np.array([[float(x[k]) for k in ['ball_x','ball_y','ball_z']] for x in rows]);a.plot(*q.T,label=n)
  r=results[n];p=np.array(r['release']['position_m']);v=np.array(r['release']['velocity_world_m_s']);a.quiver(*p,*(v*.15),color='red');b.plot(t,q[:,2],label=n)
 a.scatter([4.57],[0],[3.04],color='black',label='rim');a.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)');a.legend(fontsize=7);b.axhline(3.04,color='black',ls='--');b.set(xlabel='time (s)',ylabel='ball height (m)');b.legend();fig.tight_layout();fig.savefig(OUT/'ball_trajectory_velocity.png');plt.close(fig)
 speedref=1.92;table='| 조건 | 속도 m/s | 골대 방향 | 측면 | 수직 | RMSE rad | 발 접촉 | 낙상/경고 | 득점 |\n|---|---:|---:|---:|---:|---:|---:|---:|---|\n'
 for n,r in results.items():table+=f"| {n} | {r['release_speed_m_s']:.3f} | {r['forward_m_s']:.3f} | {r['lateral_m_s']:.3f} | {r['vertical_m_s']:.3f} | {r['body_tracking_rmse_rad']:.4f} | {r['both_feet_contact_rate']:.2%} | {r['falls']}/{r['warnings']} | {r['score']} |\n"
 audit();pres=json.loads((OUT/'preservation_audit.json').read_text());smoke=json.loads((OUT/'ppo_smoke/verification.json').read_text());anchor=json.loads((OUT/'anchor.json').read_text())
 text=f'''# M6-2 Natural Free Throw Motion Redesign

2026-10-11 / Apple Silicon / Python 3.12 / MuJoCo 3.15.0 / body_01

## 판정

**부분 개선, 전체 성공 기준 미충족.** 실제 물리 엔진에서 공이 위·앞으로 발사되며 안정적인 속도는 {z['release_speed_m_s']:.3f}m/s까지 개선됐다. 득점은 없고, 측면 속도와 낮은 릴리스 높이, 전방으로 팔을 내미는 경향이 남아 있다. 기하학적 목표 달성을 자연스러운 자유투 성공으로 처리하지 않았다. 대규모 PPO 학습은 하지 않았다.

## A. 기존 동작 문제

원본 M6 A_old를 이번 환경에서 재실행한 속도는 {results['A_old']['release_speed_m_s']:.3f}m/s이다. 사용자 제시 1.92m/s는 다른 빠른 후보의 기록이므로 동일 조건 baseline 값으로 대체하지 않았다. 이번 최선 후보는 1.92m/s 대비 {(z['release_speed_m_s']/speedref-1)*100:.1f}% 증가다.

기존 동작은 공의 앞쪽 Cartesian 목표와 고정된 전체 팔 Orientation을 따라가는 IK라 팔꿈치 아래 위치/굴곡 상태를 명시적으로 보존하지 못했다. 직전 palm-to-hoop 실험은 자세 종료 후 0.6초 뒤 릴리스하여 안정화된 공을 떨어뜨렸다. 새 실험에서는 가속 구간 중 Weld를 해제한다. Bridge의 기존 Stable PD는 목표 위치만 받아 목표 각속도는 0으로 계산하므로 빠른 Reference에서 지연이 발생한다. 이 사실과 IK 잔차는 별개로 기록했다.

## B. 재설계 구조

Preparation: 원래 Holding에서 추가 22도 knee-bend FK를 생성하고 최초 상태를 배치한다. 초기 root 한 번의 바닥 정렬 이후 root는 freejoint이며 직접 제어하지 않는다. Raising과 하체/몸통 신전은 0.18초에 함께 시작한다. Set Point는 머리와 오른쪽 어깨를 기준으로 정의하고 팔꿈치 굽힘 목표 95도 및 공 아래 elbow soft constraint를 사용한다. 기존 몸통을 고정하지 않고 5도 tilt 목표로 순차 IK를 수행한다. 왼팔은 측면 Guide 시각 Reference이며 실제 접촉 지지 성공을 주장하지 않는다. M6에서 공-인체 충돌은 기존과 같이 비활성이다.

Arm acceleration: Set Point 전부터 rising과 겹쳐 팔꿈치 신전과 위·앞 push를 생성한다. Wrist cock 목표는 손바닥/손가락 기저로 SO(3) 보간한다. Snap은 scene 골대 방향과 Z-up이 만드는 sagittal 축 기준으로 ±25/35/65도 후보를 비교한다. 이 각도는 가설이며 해부학적 정답이 아니다. XML hinge 이름을 flexion 방향으로 간주하지 않는다. 최종 Follow-through에도 별도 정지 없이 관절 목표가 이어진다. 초당 120 FK/IK 샘플, CubicSpline runtime, 450Hz physics/30Hz action과 평가 기록을 사용한다. 초기 0.18초 preparation 이후 장시간 Holding은 없다.

높은 Set Point H는 초기 crouch 대신 선 자세의 머리 기준으로 정의했다. 높인 후보는 시각적인 공 높이는 개선하지만 발사 속도가 감소하고 양의 Snap 후보 H_high_+25는 실제 수직 속도가 음수라 실패다. 자연스러움과 속도를 동시에 만족하는 Reference는 아직 확보하지 못했다.

## C. 조건 비교

A 원본, B 새 pose/느린 팔, C 하체·몸통 가속, D 팔 가속, E 손목 snap, F timing scan, G implicit target-velocity controller, H 높은 Set Point. C 이후 하체와 팔의 시작/겹침 시간도 달라지므로 완벽한 단일 변수 실험은 아니다.

{table}

## D. 발사·Weld 물리

선택한 발사 진단 후보: **{best}**. 전체 성공 후보를 의미하지 않는다.

- 실제 Release: {z['release']['actual_time_s']:.6f}s, world position {z['release']['position_m']}m.
- World velocity: {z['release']['velocity_world_m_s']}m/s.
- 골대 방향 {z['forward_m_s']:.3f}, 측면 {z['lateral_m_s']:.3f}, 수직 {z['vertical_m_s']:.3f}m/s, 발사각 {z['release_angle_deg']:.2f}도.
- 공 최고 높이 {z['max_ball_height_m']:.3f}m, 림 높이 3.04m. 실제 궤적의 림 최소 거리 {z['minimum_rim_distance_m']:.3f}m. 첫 비행의 최소 거리는 phase_metrics.csv에 별도 기록한다.
- 수평거리 {z['release_horizontal_distance_m']:.3f}m, 높이차 {z['height_difference_m']:.3f}m. 공기저항 없는 점질량 최소 속도 하한은 sqrt(g*(h+sqrt(d²+h²)))={z['minimum_ballistic_speed_m_s']:.3f}m/s. 이는 농구공 크기를 고려한 득점 보장이 아니다. 현재 속도와 최고 높이는 도달에 부족하다.
- Release qpos/qvel 및 각속도 점프 모두 정확히 0. 초기 이후 공 상태 쓰기 0, 외부 힘 0을 step 내부 assert로 확인. 공의 실제 속도를 Weld 해제 후 유지한다. 자유 비행 중 중력 속도 잔차 최대 {z['max_freeflight_velocity_residual_m_s']:.3g}m/s.
- 공 반지름 .121m, 질량 .600kg. 기존 림/바닥/백보드 모델 유지. 유한 공 반지름을 고려하는 기존 ScoringDetector를 재사용하여 실제 득점 False.
- Wrist-local Anchor {anchor['local_offset_m']}m, 손목에서 거리 {anchor['distance_from_wrist_m']:.3f}m. 손목 회전은 omega × R offset 속도를 만들므로 스냅 방향이 잘못되면 공에 아래/측면 속도가 생긴다. offset/반지름/질량은 수정하지 않았다. 최대 Weld 위치 오차 {z['max_weld_error_m']*1000:.2f}mm.

## E. Controller와 관절축

최선 발사 후보 body RMSE {z['body_tracking_rmse_rad']:.5f}rad, 선택 관절 속도 RMSE {z['selected_velocity_rmse_rad_s']:.3f}rad/s, torque saturation 최대 {z['max_torque_saturation']:.2%}. 양발 접촉 {z['both_feet_contact_rate']:.2%}, 최소 COM support margin {z['min_com_margin_m']*1000:.2f}mm, foot slip RMS {z['foot_slip_rms_m_s']*1000:.2f}mm/s. 낙상·경고 0.

원본 Controller 파일은 보존했다. 새로운 StableVelocityBridge는 기존 Stable PD mass/bias/implicit acceleration 계산에 목표 각속도 항을 넣고 COM 보정과 torque limit을 유지한다. amount .25/.5/.8을 대조했다. 단순 additive velocity 보상 controller_ff_0.35는 body RMSE .272rad로 실패, .7은 QACC 경고 및 낙상으로 실패다. 이 높은 속도 수치를 성공 후보로 선정하지 않았다.

axis_probes.csv에는 Shoulder/Elbow/Wrist/Hip/Knee/Ankle/Torso/Spine/Chest의 실제 xaxis, XML range, ±.1rad FK 공 중심/손가락/손바닥 변화가 있다. XML range 때문에 한쪽 probe가 clip된 경우 actual_delta_rad를 함께 기록했다. forearm의 별도 관절은 없으며 R_Elbow 세 hinge가 관여한다. axis_roles.csv는 실제 시작 pose에서 flexion-axis=cross(finger,palm), forearm twist-axis=elbow→wrist, radial/ulnar proxy의 축 정렬 점수를 기록한다. 복합 회전 pose에 따라 축은 변하므로 단일 XYZ를 고정 해부학적 축으로 확정하지 않는다. 손바닥 법선은 -Wrist local Y proxy이며 실제 mesh anatomical normal로 검증된 것이 아니다.

원본/목표의 주요 joint angle, 실제/목표 velocity, torque, root/pelvis quaternion, shoulder/elbow/wrist world 위치·quaternion, 공 위치·속도, holding 및 release 상태는 각각 joints.csv, joint_evaluation.csv, kinematics.csv, physics.csv, states.npz에 있다. peak velocity/acceleration/jerk는 30Hz 저장 데이터로 평가하여 매우 짧은 실제 peak를 놓칠 수 있다. 자유 root의 reference translation은 planner/진단용이며 actuator 목표가 아니다. 전체 body RMSE가 작아도 wrist/set-point의 오차를 숨길 수 있으므로 IK 잔차와 직접 시각 판정을 병행했다.

## F. 시각 판정 및 산출물

실제 저장 physics로 1배속 재생한다. Replay의 qpos 설정은 기록 재현만을 위한 것이며 실험 중 공 상태를 덮어쓴 것이 아니다.

- 기존 같은 카메라: outputs/m6_2_natural_shooting/A_old_side_1x.gif
- 발사 진단 측면: {best}_side_1x.gif, 45도: {best}_front_1x.gif
- 높은 Set Point 비교: H_high_-25_side_1x.gif
- 각 *_phases.png에 초기/상승/신전/Set Point 근처/Release 직전/순간/직후/Follow-through/비행/실패 10 프레임.
- 각 run/diagnostics.png: 관절각, 공 위치·속도, tracking/saturation, COM/slip.
- ball_trajectory_velocity.png: 실제 공 궤적 및 릴리스 속도 화살표.

영상 검토: 무릎/몸통의 상승 동시 신전은 개선. 팔꿈치 굽힘→신전은 존재. 공의 위·앞 자유 비행도 존재. 그러나 공이 이마 위를 충분히 거치는 모습, 해부학적으로 명확한 cock-back→finger-down snap, 자연스러운 Follow-through는 충족으로 확정하지 않았다. 팔을 앞으로 내미는 경향이 남아 있으므로 **자연스러운 자유투 완성은 실패/미완료**다. H를 포함한 결과도 속도와 자연스러움을 동시에 통과하지 못했다.

## G. PPO 및 테스트

{len(results)}개 실제 physics 조건 비교. 원본 회귀 140개 및 기존 M6 17개 통과(regression.log). PPO env checker, 기존 checkpoint load 및 action/observation {(smoke['action_dim'],smoke['observation_dim'])} 호환, 128-step smoke + 128-step resume 수행. 이전 counter 256→384→512. 저장/로드의 deterministic action 정확히 일치. 이것은 shooting 학습 성공이 아니다. Observation/Action 공간은 같지만 Reference SHA가 바뀌어 원래 run.py의 엄격한 resume config 검사에는 불일치한다. 이번 smoke는 명시적인 새 환경에서 weights를 가져오는 방식으로 수행했다. Release timing은 실험 파라미터로 바꾸며 기존 30차원 residual action에 timing action을 추가하지 않았다.

보호 파일 {pres['protected']}개 SHA256 변화 0. 기존 checkpoint roundtrip 및 audit의 쓰기 경로도 새 출력 폴더로 리다이렉트했다. Git 명령, 원본 모델 변경, 기존 NPZ 변경, 대규모 학습 없음.

## 실행

팀 저장소 mujoco/SMPLSim에서:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore VECLIB_MAXIMUM_THREADS=1 ../../../basketball-r1/.venv/bin/python -m examples.m6_2.experiment experiment
PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore VECLIB_MAXIMUM_THREADS=1 ../../../basketball-r1/.venv/bin/python -m examples.m6_2.refine
PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore ../../../basketball-r1/.venv/bin/mjpython -m examples.m6_2.visualize gui --condition {best} --view side --repeats 0
PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore ../../../basketball-r1/.venv/bin/mjpython -m examples.m6_2.visualize gui --condition H_high_-25 --view side --repeats 0
PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore ../../../basketball-r1/.venv/bin/python -m examples.m6_2.verify regression
PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore ../../../basketball-r1/.venv/bin/python -m examples.m6_2.verify smoke --reference outputs/m6_2_natural_shooting/references/{best}.npz --ff .8
```

新実験ソース: examples/m6_2/experiment.py, refine.py, high_setpoint.py, visualize.py, verify.py, report.py. 결과는 outputs/m6_2_natural_shooting/ 아래. 기존 성공 실험은 수정하지 않았다.

## 다음 병목

1. 높은 Set Point와 팔꿈치 아래 위치, wrist cock orientation을 동시에 실현하는 상완/전완 pose와 Anchor 위치를 먼저 분리 검증한다. 현재 FK 잔차는 발사 끝 구간에서 0.1m 이상 발생할 수 있다.
2. 부자연스러운 전방 reach와 측면 속도를 줄이는 world elbow-path 및 실제 head-relative ball-path 제약이 필요하다. 지금의 soft elbow bend만으로 충분하지 않다.
3. 안정적인 source angular velocity와 실제 tracking을 확보한 뒤 실제 릴리스 수평·수직 성분을 늘린다. 높은 불안정 속도는 배제한다.
4. mesh 기준 손바닥 법선/손가락 방향 및 anatomical wrist extension calibration을 확정하고, 손목 운동의 omega×offset 기여와 팔목 선속도를 분해한다.
5. Reference 자연스러움과 림 도달 가능한 속도 모두 확보 전까지 대규모 PPO를 진행하지 않는다. 득점 실패는 보상/학습시간으로 해결됐다고 주장하지 않는다.
'''
 text=text.replace('新実験ソース','새 실험 소스')
 text+='''

## 사용자 피드백: 큰 손목 방향 전환 복원

새 Cock-back 목표가 기존 upright → forearm roll → palm-to-hoop 큰 회전 순서를 대체해 버렸다. IK가 회전을 어깨·팔꿈치에 분산하여 사용자가 기대한 큰 손 회전이 약해졌다. 이 문제를 별도 restored_turn.py / restored_turn_early.py에서 수정했다. 원래 palm-to-hoop Reference의 SO(3) 손 방향과 손목 hinge 경로를 상승 중 참고하여 큰 방향 전환을 보존하고 Cock-back/Snap을 별도로 겹쳤다. Wrist-local prior는 발사 구간에서 점차 해제하는 조건도 비교했다.

I_restored_turn_35 실제 손목 XYZ hinge 운동 범위는 약 81.6/197.9/72.1도, 1.0초의 손가락 world 방향 변화는 시작 대비 약 120.1도다. 이는 해부학적 flexion 198도라는 뜻이 아니다. 큰 손 방향 전환은 다시 보이지만 I 조건의 발사는 골대 반대·아래 방향으로 실패했다. J는 큰 회전을 팔 가속 전에 완료하여 안정화 시간을 주었으며 J_early_turn_0.35는 골대 방향 .231m/s, 수직 .050m/s로 거의 떨어뜨리는 수준이었다. 손목 회전 복원과 자연스러운 고속 발사 동시 성공을 주장하지 않는다.

화살표: 초록=실제 wrist→Middle1 손가락 방향, 파랑=-Wrist local Y 손바닥 법선 proxy. 이전 GIF를 그대로 보여주는 대신 새로운 실제 물리 결과를 생성했다.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore ../../../basketball-r1/.venv/bin/mjpython -m examples.m6_2.visualize gui --condition I_restored_turn_35 --view front --repeats 0
PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore ../../../basketball-r1/.venv/bin/mjpython -m examples.m6_2.visualize gui --condition J_early_turn_0.35 --view side --repeats 0
```

추가 시각화: I_restored_turn_35_front_1x.gif, J_early_turn_0.35_side_1x.gif. joint_tracking_angle_velocity.png는 주요 hip/knee/torso/shoulder/elbow/wrist XYZ의 실제·목표 관절각 및 각속도 비교다. 신규 M6-2 검증 테스트 3개도 통과하여 기존 157개와 합계 160개다.
'''
 (ROOT/'M6_2_NATURAL_FREE_THROW.md').write_text(text);print('Report written:',ROOT/'M6_2_NATURAL_FREE_THROW.md')

if __name__=='__main__':report()
