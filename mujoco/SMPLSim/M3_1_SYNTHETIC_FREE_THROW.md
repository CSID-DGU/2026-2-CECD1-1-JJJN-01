# M3-1: 합성 자유투 기준 모션과 연속 추종

2026-10-10 / macOS Apple Silicon / Python 3.12 / MuJoCo 3.15.0.

**이 데이터는 물리 시뮬레이션·모션 입력 파이프라인 검증용 임시 합성 데이터다. 실제 농구 자유투의 정답, 모션 캡처, 선수 동작 또는 성공률 데이터로 사용하지 않는다.** 농구공/골대/볼 release를 구현한 단계도 아니다.

## 구현과 결과

기존 M3 SMPLHMotion NPZ 인터페이스를 그대로 사용해 하나의 3초 기준 모션을 생성했다. 30 FPS의 0~3초 양끝 샘플을 포함하므로 **91개 pose 샘플/90개 시간 간격**이다. 각 체형에서 450 Hz 물리 1350스텝을 실행했고 30 Hz로 90개 지표 샘플을 기록했다.

하나의 synthetic_free_throw.npz를 네 모델에 입력한다. 관절 pose, root orientation과 timing은 동일하고 체형별 초기 floor clearance를 맞추기 위해 root 위치에 일정한 offset만 더한다. 각 실행 후 angular pose가 source와 완전히 같은지, root offset이 모든 프레임에서 일정한지, source SHA-256이 변하지 않았는지 검사한다. 기준 동작을 체형별로 다시 만들지 않는다.

| 체형 | body RMSE rad | 동작 10관절 RMSE rad | 균형/양발 접촉 | 골반 최대 기울기 | 낙상 |
|---|---:|---:|---:|---:|---:|
| body_01 | 0.03243 | 0.07318 | 100% / 100% | 3.33° | 0 |
| body_02 | 0.03204 | 0.07355 | 100% / 100% | 1.67° | 0 |
| body_03 | 0.03441 | 0.07286 | 100% / 100% | 5.44° | 0 |
| body_04 | 0.03093 | 0.07232 | 100% / 100% | 1.70° | 0 |

네 headless 테스트와 네 native GUI 테스트 모두 --check 통과, warning 카운터 0, 프로그램 exit 0. GUI/headless 결과의 모든 summary 수치가 일치했다. 화면 캡처/시각 판독은 수행하지 않았다. 전 구간을 측정하며 M3-0에서 제외했던 최초 1초도 이번에는 제외하지 않는다. 첫 측정 시점은 1/30초이며 t=0 초기 상태는 지표에 포함하지 않는다.

body RMSE는 root와 손가락을 제외한 63개 hinge, 동작 RMSE는 양쪽 Hip_x/Knee_x/Shoulder_x/Elbow_y/Wrist_x 10개에 대한 기준 목표각 오차이다. 손가락을 포함한 전체 153개 RMSE는 summary.csv의 joint_rmse_rad에 별도 저장한다.

관절별 RMSE 범위는 Knee_x 0.053~0.066rad, Shoulder_x 0.100~0.102rad, Elbow_y 0.096rad, Wrist_x 0.054rad이다. 빠른 단계 전환에서 PD 지연과 오차가 남는다. 평균 손목 높이는 처음보다 약 0.405~0.492m 상승하고, 무릎 최대 실제 각도는 약 0.409~0.424rad였다 (목표 최대 0.35rad). 최대 발 drift는 약 4.09mm. 동일 관절 목표라도 신체 길이/질량/관성이 달라 결과는 동일하지 않다.

## 단계와 목표

| 시간 s | 단계 | 목표 변화 |
|---|---|---|
| 0.00~0.35 | 기본 서기 | M3 서기: Hip_x=-.05, Knee_x=.10, Ankle_x=-.05, Shoulder_z 좌 -1.35/우 +1.35 |
| 0.35~0.85 | 무릎 굽히기 | Knee_x .10→.35, Hip_x -.05→-.175, Ankle_x -.05→-.175 |
| 0.85~1.30 | 무릎 펴기 | 하지 목표를 초기 서기 각으로 복귀 |
| 1.30~1.95 | 팔 올리기/준비 | Shoulder_x 0→-1.15, Elbow_y 좌 0→-.90 / 우 0→+.90 |
| 1.95~2.50 | 팔꿈치 펴기 | Elbow_y 좌 -.90→-.10 / 우 +.90→+.10, Shoulder_x -1.15→-1.30, Wrist_x 0→-.30 |
| 2.50~3.00 | 손목 움직임/후속 자세 | Wrist_x -.30→+.20 (2.75초), 이후 +.10까지 복귀 |

양손을 사용한 간단한 검증 동작이다. 투구 손/보조 손의 농구 기술 차이나 공을 잡는 손가락 동작은 모델링하지 않았다. 팔꿈치를 펴기 전에 팔 올리기 단계에서 굽힌 준비 자세를 만들고, 손목 flick 전에 손목을 cock한다.

원본 목표곡선은 quintic smoothstep으로 경계의 속도/가속도가 0이 되도록 생성한다. NPZ를 읽는 기존 BoundMotion은 root orientation SLERP 및 위치/hinge 각 선형 보간을 사용한다. 따라서 시뮬레이터에 제공하는 각도는 연속이지만, 30 FPS 샘플 경계의 속도 연속성까지 보장하는 C2 재생기는 아니다.

## 제어와 측정

M3 BalancedPD의 gain, mass/leg-length scaling, COM Jacobian 및 ankle/hip 목표 보정식을 그대로 재사용했다. PPO, root actuator/고정, 외부 지지력, 중간 episode reset, 추종 중 qpos 덮어쓰기를 사용하지 않는다. 초기 pose를 설정한 뒤 3초 동안 mj_step으로 물리 상태를 진행한다. root reference는 초기화/진단 용도이며 전체 root 경로를 servo하지 않는다.

접촉은 floor와 Ankle/Toe 충돌의 normal force 합으로 측정한다. 한 발 force>1N이면 접촉이다. 힘이 있는 foot box의 평면 투영 hull로 COM support_margin을 근사한다.

낙상은 초기 대비 root 높이<65%, pelvis tilt>35°, 또는 힘이 있는 nonfoot-floor 접촉 중 하나. 균형은 낙상 없음, pelvis tilt<25°, 한 발 이상 접촉, COM projection이 근사 support hull 밖 2cm 이내이다. 이 값은 30 Hz의 근사 지표이며 CoP/ZMP·외란 안정성·실제 자유투의 안정성을 보장하지 않는다. 물리 상태의 유한 여부는 모든 physics step에서 확인한다.

--check 기준: 전체 3초 완료, 낙상/엔진 warning 없음, balance/양발 접촉≥95%, body RMSE≤.10rad, 동작 RMSE≤.20rad. 실제 각도 변화가 Knee≥.10, Shoulder≥.65, Elbow≥.40, Wrist≥.15rad인지 확인한다. 이 기준은 파이프라인 sanity check이며 농구 기술의 정답 기준이 아니다.

## 실행 명령

SMPLSim 디렉터리에서:

```sh
# 공통 NPZ 생성 + 네 체형 연속 추종 + 전체 CSV/Markdown report
../.venv/bin/python examples/m3_free_throw.py --all --check

# 공통 source 파일을 그대로 사용한 단일 체형 GUI
../.venv/bin/mjpython examples/m3_free_throw.py --body body_01 --gui --check   --source outputs/m3_free_throw/synthetic_free_throw.npz   --output outputs/m3_free_throw_gui

# 네 체형 순차 GUI (체형당 3초, 창 종료 시 다음 프로세스)
../.venv/bin/mjpython examples/m3_free_throw.py --all --gui --check   --source outputs/m3_free_throw/synthetic_free_throw.npz   --output outputs/m3_free_throw_gui

# 저장된 source를 재생성하지 않고 headless 재검증
../.venv/bin/python examples/m3_free_throw.py --all --check   --source outputs/m3_free_throw/synthetic_free_throw.npz   --output outputs/m3_free_throw_replay

# 회귀 테스트 8개
../.venv/bin/python -m unittest discover -s tests -v
```

--source 생략 시 output/synthetic_free_throw.npz를 갱신한다. --source 지정 시 그 파일은 읽기 전용으로 사용한다. --models/--output으로 다른 경로를 지정할 수 있으며 기본값은 스크립트 위치 기준 절대 경로이다. GUI 결과는 headless 결과와 다른 output 디렉터리를 권장한다. --check로 실행하다 창을 일찍 닫으면 불완전한 rollout으로 실패한다. macOS GUI는 mjpython을 사용하고 --all은 창 종료 경합을 피하도록 체형별 프로세스를 실행한다.

## 출력 파일

outputs/m3_free_throw/:

- synthetic_free_throw.npz: 단일 공통 기준 모션, 기존 schema_version=1.
- reference_metadata.json: SYNTHETIC_PIPELINE_VALIDATION_ONLY, basketball_ground_truth=false, source hash, phase 시간표, 엔진 버전, 제어/측정 설정.
- summary.csv: 체형별 전체 3초 비교.
- phase_summary.csv: 6단계×4체형의 오차/균형/발 접촉/낙상.
- task_joint_summary.csv: 동작 10관절×4체형의 RMSE/최대오차/목표·실제 각도 변화량.
- REPORT.md: 실행 결과에서 자동 생성하는 Markdown 보고서, 임시 데이터임을 명시.
- body_XX_synthetic_free_throw_timeseries.csv: 30 Hz 시간/phase/추종오차/COM/접촉력/골반/발 drift/낙상.
- body_XX_synthetic_free_throw_joint_tracking.csv: 모든 153 hinge의 30 Hz 원래 목표/COM 보정 목표/실제각/오차/토크.
- body_XX_synthetic_free_throw_reference.npz: 동일 source pose + 체형별 일정한 floor offset.
- body_XX_phase_summary.csv, body_XX_task_joint_summary.csv, body_XX_synthetic_free_throw_summary.csv: 개별 체형 결과.
- body_XX_joints.csv: qpos/dof/actuator 주소 대응.

phase 경계는 source 일정에 따라 시간으로 지정한다. 추종이 지연되어도 phase 이름을 actual pose에 맞춰 바꾸지 않으므로 각 단계의 지연/오차를 확인할 수 있다. 최초 0.35초 구간도 버리지 않는다.

## 변경 파일

- smpl_sim/motion/synthetic_free_throw.py: 6단계 합성 곡선, phase 정의, 관절 목록, SMPLHMotion 생성.
- examples/m3_free_throw.py: 하나의 source를 생성/로딩하고 네 체형에 적용, phase/관절 CSV 집계, source 불변 검사, 자동 Markdown report, 검증 및 GUI orchestration.
- examples/m3_control.py: 기존 실행기를 재사용하도록 optional motion label, 측정 시작 시간, 관심 관절 목록, joint log stride를 추가. M3-0 기본값(label/imported, 최초 1초 제외, 기존 관심 관절, 10 Hz joint log)은 유지. 제어 로직은 변경하지 않았다.
- tests/test_synthetic_free_throw.py: 3초/30 FPS/91샘플과 NPZ round trip, 단계 순서·각도, 매끄러운 목표곡선 및 입력 오류 검증.
- M3_1_SYNTHETIC_FREE_THROW.md: 이 문서.

## 검증 범위

8개 unittest 통과 (기존 5 + 합성 모션 3), 네 headless --check 통과, 네 GUI --check 통과. source 동일성과 각도 배열 동일성 검사 통과. compileall과 git diff --check 통과.

이번 단계는 연속 목표 입력→관절 변환→PD+COM→물리 실행→정량 평가를 확인했다. 실제 영상 기반 동작 추출, 공의 release timing, 골대/공 충돌, 발 IK, 정확한 hand rest frame 정렬, root 궤적 추종 및 PPO 최적화는 후속 단계이다.
