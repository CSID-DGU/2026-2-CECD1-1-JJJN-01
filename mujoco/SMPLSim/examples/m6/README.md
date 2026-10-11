# M6 Weld Shooting — 구현 및 검증 결과

2026-10-11 / macOS Apple Silicon / MuJoCo 3.15.0 / Python 3.12

## 상태

- Weld Holding/Release: **검증 통과**. 네 실험 모두 해제 순간 공 qpos/qvel 완전 동일, 낙상·경고 0.
- 전신 합성 Reference/PD 추종: 실행 가능. **자연스러운 Shooting-ready Reference 성공은 미달**.
- 득점 Detector: 단위 테스트 및 독립 물리 fixture 검증 통과.
- Gym/Residual PPO/Checkpoint/Resume: **짧은 Smoke Test 통과**.
- 실제 Reference 득점: **0회**. PPO 전후 같은 seed 7/8, 두 episode 모두 0/2.
- 실제 PPO 득점 학습: **미검증**. 대규모 학습 수행하지 않음.

## 측정 결과

| condition | score | ball_rise_mm | release_speed_m_s | release_vx | release_vy | release_vz | all_153_joint_rmse_rad | root_position_rmse_m | root_orientation_rmse_deg | both_feet_rate | min_com_margin_mm | slip_mm_s | weld_max_error_mm | fall_count | warnings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_verified/seed_7 | False | 431.5543 | 0.5229 | 0.4827 | -0.1625 | 0.1185 | 0.0279 | 0.0161 | 2.3943 | 1.0 | 119.6038 | 2.2699 | 0.2725 | 0 | 0 |
| policy_verified/seed_7 | False | 428.4819 | 0.5204 | 0.48 | -0.1672 | 0.1112 | 0.028 | 0.0165 | 2.71 | 1.0 | 119.5236 | 2.4048 | 0.2597 | 0 | 0 |
| faster_reference/seed_7 | False | 427.959 | 1.3013 | 0.9695 | -0.3093 | 0.811 | 0.0388 | 0.0173 | 2.306 | 1.0 | 118.068 | 2.3737 | 0.2805 | 0 | 0 |
| coordinated/seed_7 | False | 194.7376 | 0.8636 | -0.015 | 0.8559 | -0.1143 | 0.0518 | 0.0145 | 2.3077 | 1.0 | 118.0448 | 2.1223 | 0.4634 | 0 | 0 |
| fastest_reference/seed_7 | False | 446.4345 | 1.9219 | 1.2755 | -0.3202 | 1.4014 | 0.0462 | 0.0188 | 2.2912 | 1.0 | 119.6457 | 2.4531 | 0.52 | 0 | 0 |

`comparison.csv`의 Body RMSE는 모든 153 hinge의 실제 qpos와 적용된 target 차이다. Root 오차는 자유 Root의 Reference 대비 오차이며 Root를 강제 추종시키는 actuator가 있다는 뜻이 아니다. 기존 파일 일부 `summary.json`은 선택된 30 residual 관절 기준이므로 두 집계를 섞지 않는다. `joints.csv`는 선택된 30 관절의 각도·속도·토크이며 나머지 관절은 기존 Controller가 계속 제어한다.

기본 Reference는 공 431.55mm 상승, 릴리스 속도 0.523m/s, 양발 접촉 100%, 낙상·경고 0. 그러나 림 높이 3.04m에 도달하려면 현재 릴리스 높이 1.48m에서 수직 속도만 최소 약 5.54m/s가 필요하다. 현재 수직 속도는 약 0.118m/s다. 골대까지 수평 진행도 부족하다. 작은 residual만으로 이 차이가 해결된다는 근거는 없다.

## M6-0 물리

기존 M4 `build_holding_xml` 및 `WeldBallRelease`를 호출한다. 원본 XML은 읽기만 한다. 600g, 반지름 0.121m, 얇은 구면 관성 I=2/3 mr²를 유지한다. `R_Wrist`와 공의 상대 offset은 M5 성공 Reference의 `rel0`; 방향은 초기 wrist 방향과 일치한다. Weld `solref=.02 1`, `solimp=.95 .99 .001`, `torquescale=.121`, 초기에는 비활성화한다. Reset에서 공 qpos/qvel을 **한 번** 정렬한 뒤 `data.eq_active[id]=1`. 제약은 공의 질량과 관성을 손에 전달한다. 해제 시 `data.eq_active[id]=0`, 위치나 속도를 보정하지 않는다.

A 정지, B 수직 상승, C 위·앞 이동, D 손목 회전: 초기 1.45초 요청, 실제 1.451111초(450Hz 다음 스텝) 해제. 위치·선속도·각속도 점프 모두 0. 최대 Holding 상대 위치 오차 0.288mm 이하. 충돌 전 free-flight 선속도 중력 residual 약 2e-13m/s 이하. 회전은 freejoint raw 각속도가 ball-local, 보고된 각속도는 world. 정확한 전후 qpos(wxyz)/qvel은 각 `physics/A~D/summary.json`에 기록한다. Toggle 전후 동일성이 스텝 전후 속도 동일성을 뜻하지 않는다: 다음 스텝에는 중력이 작용한다.

공과 Humanoid의 충돌은 Holding 및 Release 후에도 제외한다. 이는 M6에서 손가락의 추가 접촉 가속을 제외하기 위한 명시적 설계다. 바닥·림·백보드 충돌은 유지한다. 자유 Root, 원래 actuator153개, 기존 PD+COM gain/torque limit 보존. 각 physics substep에서 ball 상태 덮어쓰기 및 외력 없음 assert를 수행한다.

## M6-1 전신 Reference

`core.generate_reference`: 발 위치·방향을 보존하는 하체 FK 최적화로 초기 무릎 굽힘 12° 추가. 초기 free-root 위치만 정렬하며 시뮬레이션 Root는 계속 자유 상태다. 무릎/고관절 신전, 몸통 목표 신전 7°, 양팔 wrist position/orientation IK, 실제 림 방향 head IK를 중첩한다. 오른손 world orientation은 고정 축 SO(3) exponential, 관절 목표는 30Hz sequential IK + clamped cubic spline으로 연속 생성한다. 팔 가속 및 follow-through도 별도 공 상태 제어 없이 관절 목표로 표현한다. Release 이후 약 2.8초 관찰한다.

실제 무릎 굽힘 감소는 확인됐지만 기본 조건의 몸통 기울기는 약 13.89→14.97°로 기대한 신전이 아니다. 상대 손목 회전 proxy는 최대 약 134°로 과도하다. 이 수치는 해부학적 Wrist Extension 정답 각도로 해석하지 않는다. 팔꿈치 위치·손목 위치 IK가 해부학적 협응을 보장하지 않는다. 양손은 Guide IK로 표현할 뿐 실제 양손 Holding은 이번 범위에서 요구하지 않는다.

`refine.py`는 기존 실제 FK/Jacobian 기반 whole-body IK로 몸통 축과 상대 손목 회전을 추가 제약했다. 이 후보는 상승량 약 195mm, 옆 방향 릴리스 속도 약 0.86m/s로 오히려 악화했다. 채택하지 않았다. 자연스러운 전신 Reference 및 충분한 발사 에너지를 먼저 확보해야 한다. 초기 선언된 Reference 후보 시간은 2.4초/1.2초/0.6초, 전체 episode 시간과 GUI 배속은 서로 다르다. 원본 SPL NPZ를 수정하거나 이 합성 궤적을 실제 사람 Ground Truth로 주장하지 않는다.

## M6-2 득점 판정

`ScoringDetector`: Release 후 위 평면(zrim+radius)→아래 평면(zrim-radius) 순서의 하강을 요구한다. 림 64 capsule의 보수적 내접 반지름을 사용하고, 구의 rim-plane 단면 반지름 `sqrt(r²-dz²)`까지 고려한다. 전체 구체가 아래로 통과한 뒤 한 번만 득점한다. 물리 450Hz 및 스텝 사이 보간된 단면을 검사하며, 극단적인 큰 timestep에서 완전한 연속 swept-volume 인증을 제공하지는 않는다.

중앙 통과, 바깥 통과, 림 튕김 이탈, 백보드 반사 후 진입, 상향 통과, 정지, 미릴리스, 빠른 교차, 중복 득점 방지, 기울어진 출구를 검사한다. 독립 물리 fixtures의 중앙/백보드 반사 통과는 실제로 검출됐다. 이 fixture는 M4 테스트 전용 초기 위치·속도 입력을 사용하며 **Humanoid 슈팅/PPO 득점으로 계산하지 않는다**. `fixtures.py`는 주 환경과 분리돼 있다. 상향 fixture는 상승 구간만 평가한다: 이후 다시 하강해 정상 통과하면 그때 득점하는 것이 올바른 동작이다.

## M6-3 PPO 설계

Action 30차원: 양쪽 Hip/Knee 각 XYZ(12), Torso/Spine/Chest XYZ(9), 오른쪽 Shoulder/Elbow/Wrist XYZ(9). [-1,1]을 ±0.15rad 잔차로 변환하고, 최대 잔차 변화율 0.8rad/s로 연속 적용한다. XML 한계로 clip한다. 원래 Controller가 torque를 계산한다. 고정 Release 시점이며 Stage3 timing action은 아직 구현하지 않았다.

Observation 169차원 순서:

| 항목 | 차원 | 단위/정규화 |
|---|---:|---|
| 선택 관절 q, qvel, target error | 90 | rad/π, rad/s÷10, rad/π |
| Root displacement, quaternion | 7 | 초기 대비 m, unit wxyz |
| COM-root, support margin, 좌우 발힘 | 6 | m, m, N÷1000 |
| 오른손 world 위치·quaternion·각/선속도 | 13 | m, wxyz, rad/s·m/s÷10 |
| 공 world 위치·quaternion·선속도·각속도 | 13 | m, wxyz, m/s·rad/s÷10 |
| 림-공 | 3 | m |
| phase one-hot, Weld flag, release remaining | 7 | 5 phase, 0/1, 초÷3 |
| 적용 중 residual | 30 | rad÷0.15 |

COM은 Weld 활성 중 human+ball 질량 중심, Release 후 Humanoid 질량 중심이다. Support margin도 같은 기준으로 계산한다. Quaternion 순서는 항상 wxyz, scene Z-up, SI. 속도 scaling은 clipping이 아니며 Observation Box는 무한 bounds, 값은 finite 검사한다. timestep450Hz/control30Hz, seed reset은 재현되지만 Stage4 물리 randomization을 아직 사용하지 않는다.

Reward = 0.5·motion + 0.2·balance + Stage2에서 (2·ballistic + 0.1·horizontal_direction + 0.3·release_velocity_alignment + 100·실제 score_event) - residual/action-change/jerk/limit/saturation/root-drift penalty. Ballistic은 실제 post-release 위치·속도에서 중력 포물선의 하강 림 평면 교차 오차로 계산한다. 속도 정렬용 ballistic 목표는 비행시간1.1초의 **보상 진단값**이며 공 속도에 입력하지 않는다. 실제 score_rate는 별도 기록한다. 자유 비행 중 충돌은 이후 궤적을 바꿀 수 있어 shaping이 실제 점수를 대체하지 않는다. Jerk는 30Hz 실제 qvel 차분이며 매우 빠른 contact transient를 모두 표현하지 못한다.

Stage0: 물리·Detector·finite Gym 검증. Stage1: motion/balance 보상만 사용. Stage2: 동일 환경에 실제 scoring/ballistic 보상 추가. Stage1 전신 자연스러움 및 발사 준비 Gate는 아직 미달이므로 Stage2 Smoke Test는 소프트웨어 검증만 의미한다. Stage3 timing/Stage4 robustness는 fixed release와 reset options를 확장할 자리만 있으며 미구현 단계는 명시적으로 거절한다.

## 실행

아래 명령은 **팀 저장소 `mujoco/SMPLSim`**에서 실행한다. 기존 개발 폴더는 수정하지 않는다.

```bash
cd "/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim"
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore VECLIB_MAXIMUM_THREADS=1
M6PY=../../../basketball-r1/.venv/bin/python

# 물리, 득점 fixture, 새 테스트, 기존 회귀 검증
$M6PY -m examples.m6.run validate --output outputs/m6_weld_shooting/physics_new
$M6PY -m examples.m6.fixtures
$M6PY -m examples.m6.test_core
$M6PY -m examples.m6.run regression
$M6PY -m examples.m6.run audit

# 무잔차 물리 실행 / 학습 모델 물리 실행
$M6PY -m examples.m6.run evaluate --episodes 2 --seed 7 --output outputs/m6_weld_shooting/new_before
$M6PY -m examples.m6.run evaluate --model outputs/m6_weld_shooting/smoke_verified/final.zip --episodes 2 --seed 7 --output outputs/m6_weld_shooting/new_after

# GUI 실시간 물리 실행 (GUI 배속만 2배)
../../../basketball-r1/.venv/bin/mjpython -m examples.m6.run evaluate --gui --speed 2 --output outputs/m6_weld_shooting/new_gui

# 저장 상태 반복 GUI: 원본 검증 물리를 시각 재생, 공 제어가 아님
../../../basketball-r1/.venv/bin/mjpython -m examples.m6.run replay --states outputs/m6_weld_shooting/baseline_verified/seed_7/states.npz --gui --repeats 0 --speed 1

# 짧은 새 학습 (headless, CPU, 외부 API 없음)
$M6PY -m examples.m6.run train --stage 2 --steps 256 --seed 7 --output outputs/m6_weld_shooting/my_smoke

# 장시간 학습 명령 예시: 실행 가능하지만 현재 Reference의 발사 속도·자연스러움 미달
$M6PY -m examples.m6.run train --stage 1 --steps 100000 --seed 7 --checkpoint 10000 --output outputs/m6_weld_shooting/my_stage1
# Stage 전환은 새 환경에서 가중치 초기화; Resume과 구분
$M6PY -m examples.m6.run train --stage 2 --init-model outputs/m6_weld_shooting/my_stage1/final.zip --steps 100000 --seed 7 --checkpoint 10000 --output outputs/m6_weld_shooting/my_stage2
# 동일 Stage/Reference에서 중단 후 재개: 새 출력 폴더 사용
$M6PY -m examples.m6.run train --stage 2 --resume outputs/m6_weld_shooting/my_stage2/checkpoints/m6_10000_steps.zip --steps 50000 --checkpoint 10000 --output outputs/m6_weld_shooting/my_resume

# 새 Reference 비교 및 학습: 동일 --reference를 eval/replay/train/resume에 전달
$M6PY -m examples.m6.refine
$M6PY -m examples.m6.run evaluate --reference outputs/m6_weld_shooting/reference_coordinated_1.2s.npz --output outputs/m6_weld_shooting/new_candidate
# 정상 속도 GIF
../../../basketball-r1/.venv/bin/mjpython -m examples.m6.run replay --states outputs/m6_weld_shooting/baseline_verified/seed_7/states.npz --gif outputs/m6_weld_shooting/new_reference_1x.gif
```

`--steps`는 추가 학습 Step이며 PPO rollout128 단위로 올림될 수 있다. Checkpoint는 주기별 `checkpoints/*.zip`, best는 평가 평균 reward 기준 `best/best_model.zip`, 최종은 `final.zip`. **best reward가 best 득점률을 의미하지 않는다.** Ctrl+C 후 마지막 자동 Checkpoint에서 재개한다. 학습 파라미터·optimizer·step counter를 복원하지만 중단된 episode의 물리 state/RNG를 완전히 이어가는 bit-exact Resume은 아니다. Resume에서 Stage, motion_seconds, action/obs 크기와 Reference SHA256이 일치해야 한다. CSV logger `progress.csv`, episode `monitor.csv`, `training.json`, EvalCallback `eval/evaluations.npz`를 저장한다. CLI 장기 학습은 구현했지만 이번에는 실행하지 않았다.

## 검증 및 보호

기존 테스트 **140개 통과**, 새 테스트 **17개 통과**(최종 로그 확인). Gym checker, spaces, finite reward, seed, 정확한 release continuity, scoring, NPZ 연속성, PPO save/load, checkpoint 및 256→384 step Resume를 확인했다. 보호 파일 **3100개 SHA256 변경 0개**. 구형 회귀 테스트가 이전 audit 결과를 쓰는 부작용은 새 M6 폴더로 redirect한다. Git 명령은 사용하지 않았다. 원본 모델/라이선스 파일/원본 NPZ도 복사·수정하지 않았다.

Smoke256 steps: CPU 8.89초, 28.79env steps/s. 이 속도에는 128 step마다 evaluation과 checkpoint 저장 비용이 포함된다. 단독 headless rollout은 evaluation.json wall_seconds로 별도 비교할 수 있다. Seed7/8은 같은 결정적 초기 상태이므로 아직 다양한 초기 조건의 일관성 검증이 아니다.

## 시각화 및 후속

- `reference_1x.gif`: 무잔차 자유투 Reference의 **실패** 비행.
- `ppo_smoke_failure_1x.gif`: 짧은 학습 후 **실패** 비행.
- `physics_comparison.png`: 공 높이/속도, 무릎, 몸통, 손목 proxy, COM.
- `detector_physical_fixtures.json`: 실제 림·백보드 충돌 및 Detector 검증.

성공한 Humanoid 슈팅이 없으므로 성공 슈팅 GIF는 없다. detector 테스트의 성공을 슈팅 성공으로 포장하지 않는다. GUI 실행 및 GIF 렌더링을 macOS mjpython에서 검증했다. GIF/GUI에서 자연스러운 농구 자유투가 확보됐다고 판정하지 않는다.

우선 과제는 충분한 상향 발사 속도와 해부학적으로 적절한 팔/손목 협응을 동시에 만들 수 있는 Reference다. 그 뒤 Stage1 물리 및 자연스러움 Gate를 통과하고 Stage2 장기 학습을 진행해야 한다. 현재 source 영상의 2D 특징은 3D 회전 정답이 아니므로 별도 영상 기반 복원도 필요하다. 다음에는 wrist relative orientation과 forearm twist, elbow-extension 방향을 실제 FK로 제한하면서 속도 목표를 관절 궤적 최적화의 항으로 사용해야 한다. 공 속도를 직접 바꾸는 해결책은 사용하지 않는다.

변경/생성 파일: `examples/m6/core.py`, `run.py`, `fixtures.py`, `refine.py`, `test_core.py`, `report.py`, 본 `README.md`, `outputs/m6_weld_shooting/*`. 기존 M5-2 작업은 기준 동작 재현·2D 특징 분석까지만 검증된 상태로 보존했으며 M6 작업에서 덮어쓰지 않았다.
