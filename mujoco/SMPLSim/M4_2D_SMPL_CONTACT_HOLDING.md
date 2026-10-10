# M4-2D: SMPL+H Contact-based Basketball Holding

2026-10-10 / macOS Apple Silicon / Python 3.12 / MuJoCo 3.15.0

## 결론과 단계별 상태

**SMPL+H용 접촉 실험 코드와 측정 파이프라인을 구현했지만, 현재 설정에서 3초 Holding은 네 체형 모두 실패했다.** 따라서 Contact-based 자유투 발사 기능이 완성됐다고 판정하지 않는다. 실패 데이터를 보존했고, D3–D5는 body_01에서 원인 분석용으로만 실행했다. 네 체형의 성공한 Lifting/Release 검증은 아직 미완료다.

| 단계 | 구현·검증 | 결과 |
|---|---|---|
| D1 손 구조 | 네 체형 관절·액추에이터·충돌 형상 CSV/JSON 분석 | 완료 |
| D2 정지 Holding | 네 체형, PD on/off, 3.5초 물리 실행 | **3초 유지 실패** |
| D3 Lifting | 기존 PD로 느린 팔 목표 추종, body_01 진단 | 손은 상승하나 공은 상승 전에 떨어짐 |
| D4 실제 Motion | 원본 NPZ를 기존 Loader로 입력, body_01 진단 | Humanoid 동작 유지, 공 Holding 실패 |
| D5 Release | 손가락 목표 펴기, 상태 덮어쓰기 없음 | 공이 이미 떨어져 **유효 릴리스 없음** |
| D6 네 체형 | 동일 D2 조건 및 PD off 비교, 각각 GIF 저장 | Holding 실패 비교 완료; 이후 단계 성공 검증 보류 |

일반 실행에서는 D2 성공 기록이 없으면 후속 단계를 차단한다. `--diagnostic`은 실패 원인 분석용 실행만 허용한다. 실패한 공의 낙하/바닥 이동 속도를 릴리스 속도로 보고하지 않는다.

## D1. 손 구조와 Shadow Hand 차이

| 항목 | 현재 SMPL+H | 독립 Shadow Hand |
|---|---|---|
| 손가락 구조 | 5손가락 × 3분절 × XYZ hinge = **45 hinge/손** | 손당 22 finger hinge, MCP 등 선택 축 |
| 손목 | XYZ 3 hinge/손 | 2 hinge/손 |
| 손가락 구동 | 45 motor/손; Humanoid 전체 153 motor | 18 finger actuator; 손목 포함 전체 20 |
| 말단 구동 | 각 XYZ 축 독립 | 4 fixed tendon이 PIP+DIP 합 구동 |
| 손바닥 형상 | `R_Wrist` capsule **1개** | 여러 box 및 실제 palm mesh 시각화 |
| 손가락 형상 | 분절마다 capsule 1개, 총 15개/손 | capsule·mesh·일부 primitive 조합 |
| 회전 범위 | 모든 손가락/손목 축 **−π~π** | 관절별 해부학적 범위 |
| joint armature | 0.01 kg·m² | 기본 0.0002 kg·m² |
| 기본 passive damping | 0 | 손가락 기본 0.05 |
| 손가락 PD | 기존 BalancedPD의 kp=5, kd=0.5, limit=2 × 체형 gain_scale | 별도 실험의 position PD |
| 공 관성 | M4-1 얇은 구형 shell: 각 축 **0.0058564 kg·m²** | 독립 실험의 실구: 0.00351384 kg·m² |

SMPL+H finger 이름은 `R_Index1~3`, `R_Middle1~3`, `R_Ring1~3`, `R_Pinky1~3`, `R_Thumb1~3`이고 각각 `_x/_y/_z` motor가 존재한다. 네 체형 모두 같은 관절 수·이름·구동 구조다. 회전축은 body 로컬 XYZ이며 **SMPL rest skeleton의 형상을 고려해야 하므로 +Z를 모든 손가락의 해부학적 flexion과 동일시하지 않는다.** 현재 목표 프로파일은 시험용 로컬 회전이다. 엄지에는 별도 목표를 적용한다.

body_01의 손바닥 capsule 반지름은 32 mm, 중심은 wrist local [−44.5, −5.7, −0.8] mm이고 축의 반길이는 약 26.92 mm다. 손가락 capsule 반지름은 약 5.4–12.9 mm다. 이것은 피부 표면을 정밀하게 재현한 충돌 모델이 아니다. 자세한 체형별 치수·회전축은 `outputs/m4_2d_contact/hand_geoms.csv`, `hand_joints.csv`에 기록했다.

기존 M4-1 공은 contype=8, conaffinity=16이며 원본 손 geom은 1/1이므로 손과 충돌하지 않는다. 파생 장면에서 오른손의 contype에 bit 32를 추가하고 공 conaffinity를 **16|32=48**로 설정했다. 바닥·림·백보드 접촉은 유지한다. 왼손 보조 Holding과 공-몸통 충돌은 이번 시험에서 활성화하지 않았다.

## D2. 정지 Holding 방법

1. 원본 Humanoid와 M4-1 농구공·골대 모델을 **새 XML로 복제**한다. Weld/equality가 없으며 `neq=0`이다.
2. 원본 Humanoid만 1.5초 사전 안정화한다. 측정 시작 시 Humanoid 속도는 한 번 0으로 초기화하여 정지 조건을 만든다. 측정 중에는 Humanoid나 공의 상태를 덮어쓰지 않는다.
3. wrist local −Y를 월드 +Z로 향하게 하는 목표를 계산한다. 실제 forearm 방향을 읽어 wrist 목표를 보정하며 동등한 XYZ 회전 중 가까운 표현을 선택한다. 이는 **별도 목표 생성기**이며 기존 BalancedPD·StablePD 코드는 변경하지 않는다.
4. 기본 오른손 4손가락의 local Z 목표는 proximal/middle 0.3 rad, distal 0.21 rad다. 엄지 Z 목표는 0이며 `--thumb`으로 따로 변경할 수 있다. 나머지 손가락 축은 원본 Motion의 값이다.
5. 공 중심의 wrist local X/Z는 [−0.10, −0.007] m이다. 월드 수직 높이를 탐색해 손과 겹치지 않는 위치에 0.2 mm 여유를 두고 **초기 배치**한다. 공의 초기 선속도·각속도는 0이다.
6. 이후에는 기존 153개 motor의 제어값만 갱신한다. 공 freejoint에 actuator나 외력을 추가하지 않는다. Controller 호출 전후 공 `qpos/qvel`이 동일한지 매 step 검사한다.

PD off 대조군은 동일한 초기 자세에서 오른손 손가락 45개 motor 토크만 0으로 만든다. 손목·팔·하체의 기존 PD+COM 제어는 유지한다. PD off에서도 공이 떨어졌지만 PD on도 실패하므로 **손가락 PD가 성공적인 Holding에 필수라는 결론은 아직 낼 수 없다.** 일부 체형에서는 PD off가 조금 더 오래 버텼다.

### 측정 정의

- 기본 timestep=1/450초. 원본 gain_scale 및 토크 제한 규칙 유지.
- 초기 0–0.2초는 착좌 구간으로 제외한다. `holding_success_time_s`는 0.2초부터 **처음 50 ms 이상 연속 손 접촉 상실이 시작된 시점**까지다. `held_3s`는 이후 3초 유지 조건이다.
- 힘 0.01 N 이상인 손-공 접촉만 활성 접촉으로 센다. 접촉 좌표계 힘을 월드로 변환해 공에 작용하는 수직 힘을 합산한다.
- Slip은 각 접촉점에서 공/손의 Jacobian 속도 차이 중 접촉 법선에 수직인 성분의 RMS다. 접촉 없는 구간을 Slip=0으로 표시하되 RMS 요약에서 제외한다.
- 상대 위치는 실제 wrist local 좌표이며 자세와 무관한 위치 비교에 사용한다. `supported_relative_drift_rms_m`는 첫 접촉 상실 이전, `relative_drift_rms_m`는 전체 평가 구간을 포함한다. 후자는 낙하 이후 큰 이동도 포함하므로 Holding 성능으로 해석하지 않는다.
- COM은 Humanoid 자체와 Humanoid+ball을 구분한다. 공 포함 margin은 실제 손 접촉이 있을 때만 요약한다. 지지 영역은 기존 발 box의 투영 영역 기반이며 정확한 동적 ZMP 판정이 아니다.

## D2/D6. 네 체형 정량 결과

각 조건 3.5초. 아래 지지력은 착좌 종료부터 첫 지속 접촉 상실까지의 평균이다. 전체 0.2–3초 평균은 CSV에 별도로 보존하며 낙하 이후 0 N이 포함된다.

| 체형 | 질량 kg | gain_scale | PD on 유지 s | PD off 유지 s | on 수직 지지력 N | on Slip m/s | on 상대 위치 RMS mm | 최소 human COM margin m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| body_01 | 79.598 | 0.978 | **0.433** | 0.464 | 5.605 | 0.0562 | 68.87 | 0.1218 |
| body_02 | 61.417 | 0.618 | **0.711** | 0.667 | 5.347 | 0.0410 | 74.36 | 0.1094 |
| body_03 | 101.331 | 1.492 | **0.311** | 0.356 | 5.570 | 0.0609 | 76.79 | 0.1345 |
| body_04 | 63.758 | 0.861 | **0.564** | 0.602 | 5.157 | 0.0500 | 85.20 | 0.0976 |

공 무게는 5.886 N이다. 일시적인 지지력은 생기지만 접촉점 배치와 상대 위치가 안정화되지 않는다. PD on 접촉점은 최대 2/3/2/2개이고 네 체형 모두 양발 접촉률 **100%**, Humanoid 낙상 **0회**, MuJoCo 경고 **0회**였다. 따라서 이 실패는 현재 시험에서 전신 균형 붕괴보다 **손 안에서 공이 굴러 접촉 영역을 벗어나는 현상**에 가깝다.

body_02가 가장 오래 유지하고 Slip이 가장 작았으며 body_03의 유지 시간이 가장 짧았다. 체중뿐 아니라 손 형상과 관절 동역학도 함께 달라지므로 체중 하나로 원인을 단정하지 않는다. 한 시나리오의 결정적 실행값이며 통계적 견고성 검증은 아니다. 네 체형의 정상 릴리스 위치·속도는 **N/A**다.

기본 PD on의 최대 손 관통은 0 mm, PD off body_04는 0.124 mm였다. 기본 geom margin 1 mm에 의해 표면 간 거리가 양수일 때도 접촉력이 생길 수 있으므로 0 mm 관통이 안정적인 grasp를 뜻하지 않는다.

## 추가 D2 실패 분석

body_01에서 공의 초기 X 위치 −0.05~−0.12 m, finger Z 목표 −0.35~0.6 rad, proximal/중간 목표 분리, 엄지 목표 −0.6/0 rad, 사전 안정화 및 초기 정지 조건을 비교했다. 시험한 프로파일에서 3초 성공은 없었다. 이것이 모델에서 접촉 Holding이 수학적으로 불가능하다는 증명은 아니다.

`--palm-pad`는 원본 capsule을 삭제하지 않고 wrist에 **80×8×60 mm의 mass=0 box 충돌 proxy**를 추가하는 별도 진단이다. 원본 손 질량·관성은 유지되지만 충돌 표면은 달라진다. 이것도 3초 Holding을 달성하지 못했고, 기본 네 체형 비교에는 사용하지 않았다.

M4-1의 손 접촉에는 공의 반발 설정 `solref=[−40000, −45]`가 사용된다. 이를 손-공 **명시적 contact pair에만** `solref=[0.005, 1]`로 바꾸는 추가 시험도 수행했다. 공 자체와 바닥·림·백보드 설정은 그대로 보존했다.

| body_01, cup=0.3, X=−0.10 m | 평가 유지 s | Slip RMS m/s | 최대 관통 mm |
|---|---:|---:|---:|
| 기본 접촉, 450 Hz | 0.433 | 0.0562 | 0 |
| 감쇠 hand pair, 450 Hz | 0.511 | 0.0142 | 0.378 |
| 감쇠 hand pair, 2,000 Hz | 0.511 | 0.0111 | 0.388 |

감쇠로 접촉 떨림/Slip은 줄었지만 3초 유지에는 실패했다. 따라서 물리 timestep만 바꾸는 것으로 해결된다고 볼 수 없다.

## D3–D5. body_01 진단

정상 단계 진입 조건인 D2가 실패했으므로 다음 결과는 성공 검증이 아닌 **실패 위치를 분리하는 진단**이다.

- D3: 1초부터 2초 동안 첫 자세와 source 2.5초 자세 사이의 관절 목표를 부드럽게 연결하고 palm-up wrist 목표를 사용했다. 손목 높이는 약 **0.999 m 상승**했지만 공은 약 **0.00069 m**만 상승했다. 공은 **0.633초**에 지속 접촉을 잃어 팔 상승이 시작되기 전에 떨어졌다. 접촉으로 공을 들어 올리는 성공은 확인되지 않았다.
- D4: 60 FPS, 240프레임 원본 NPZ를 기존 Loader로 읽고, 1초 준비 뒤 `source time=t−1`로 기존 Motion 보간/PD+COM을 사용했다. 손가락 목표만 별도로 덧씌웠으며 body/root 데이터 파일은 수정하지 않았다. 공은 **0.458초**에 접촉을 잃었다. 손목은 약 **1.061 m 상승**하지만 공은 따라가지 않는다.
- D5: 3.5초부터 0.2초에 걸쳐 손가락 목표를 0으로 펴는 로직을 구현했다. 공이 이미 떨어졌으므로 `release=null`, `release_attempt.valid_held_until_command=false`다. 바닥에서 움직이던 공의 위치/속도는 진단용 `release_attempt`에만 있으며 정상 릴리스로 사용하지 않는다.

원본 Fitting NPZ의 **손목 2개와 손가락 30개 axis-angle 회전은 모두 정확히 0**이다. 손목의 월드 방향은 어깨·팔꿈치 회전에 따라 변하지만 실제 wrist/finger motion이 fitting된 것은 아니다. D4 palm normal의 world-up 내적은 시작 약 0.355이며, 전체 구간 최소 약 **−0.964**다. 손바닥 아래 지지가 계속 유지되는 기준 Motion이 아니다. body MPJPE가 작았다는 사실만으로 손-공 접촉 자세가 정확하다고 볼 수 없다.

## 구현과 보존

새 파일:

- `examples/m4_smpl_contact.py`: 파생 장면, 손 목표 생성기, 기존 Controller 재사용, 단계 gate, 계측, GUI/GIF.
- `examples/m4_smpl_contact_report.py`: 네 체형/단계 그래프, 로컬 모델 참조 회귀 실행.
- `tests/test_smpl_contact.py`: 원본 질량·관성/구동 보존, 충돌 마스크, palm-up/PD off, 공 상태 비갱신, hand pair의 환경 보존 검사.
- `M4_2D_SMPL_CONTACT_HOLDING.md`: 이 문서.
- `outputs/m4_2d_contact/`: 실험 XML, CSV, NPZ, GIF, 구조 분석 및 보존 감사.

기존 Fitting NPZ, BalancedPD/StablePD, Weld/Kinematic, M4-1 코드/모델 및 독립 실험 코드는 수정하지 않았다. 기존 파일 **148개 SHA-256 일치**, 변경 **0개**다. NPZ SHA-256은 `af4a970bbac834955ea6222f36a2ef3a320b520a40e9fbf17b85b0a32910200d`다. Git 작업을 수행하지 않았다.

기존 모델 파일은 `../../../basketball-r1/SMPLSim/outputs/smplh_physics`에서 읽으며 팀 저장소에 복사하지 않았다. 원본 body XML은 변경하지 않고 파생 XML을 출력 폴더에 생성한다.

회귀 검사에서는 기존 motion test의 모델 경로가 팀 저장소 내 제외된 XML을 가리켜 처음에는 경로 오류가 발생했다. 새 report runner가 **테스트 모듈의 모델 루트만 메모리에서 로컬 원본 경로로 지정**하여 해결했다. 기존 테스트 파일 수정·모델 복사·심볼릭 링크 생성은 하지 않았다. **기존 32개 + 신규 5개, 총 37개 모두 통과**했다. 결과는 `outputs/m4_2d_contact/regression_with_local_assets.log`에 저장했다. 테스트 통과는 모델/코드 계약 검증이며 **3초 Holding 성공과 구분**한다.

## 실행 명령

작업 위치는 팀 저장소의 **`2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim`**이다. MuJoCo 최상위 폴더에서:

```bash
cd 2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim
```

body_01 현재 Holding 조건 GUI: **공이 떨어지는 실패 재현**이다.

```bash
PYTHONDONTWRITEBYTECODE=1 \
../../../basketball-r1/.venv/bin/mjpython examples/m4_smpl_contact.py \
  --stage hold --body body_01 --gui
```

네 체형 Holding 정량 실행/각각 GIF:

```bash
PYTHONDONTWRITEBYTECODE=1 \
../../../basketball-r1/.venv/bin/python examples/m4_smpl_contact.py --stage hold --all

PYTHONDONTWRITEBYTECODE=1 \
../../../basketball-r1/.venv/bin/mjpython examples/m4_smpl_contact.py --stage hold --all --gif
```

PD off 대조군:

```bash
PYTHONDONTWRITEBYTECODE=1 \
../../../basketball-r1/.venv/bin/python examples/m4_smpl_contact.py --stage hold --all --no-finger-pd
```

원본 실제 Motion **실패 진단** GUI/GIF:

```bash
PYTHONDONTWRITEBYTECODE=1 \
../../../basketball-r1/.venv/bin/mjpython examples/m4_smpl_contact.py \
  --stage motion --body body_01 --duration 5.5 --diagnostic --gui --gif
```

단계별 진단, 원본 데이터 수정 없음:

```bash
PYTHONDONTWRITEBYTECODE=1 \
../../../basketball-r1/.venv/bin/python examples/m4_smpl_contact.py \
  --stage lift --duration 4 --diagnostic

PYTHONDONTWRITEBYTECODE=1 \
../../../basketball-r1/.venv/bin/python examples/m4_smpl_contact.py \
  --stage release --duration 5.5 --diagnostic
```

접촉 감쇠/고해상도 비교는 별도 output에 저장:

```bash
PYTHONDONTWRITEBYTECODE=1 \
../../../basketball-r1/.venv/bin/python examples/m4_smpl_contact.py --stage hold \
  --soft-hand-contact --fine-step --output outputs/m4_2d_contact/custom_damping
```

분석 그래프/회귀 테스트:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/contact_holding_mpl \
../../../basketball-r1/.venv/bin/python examples/m4_smpl_contact_report.py

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/contact_holding_mpl \
../../../basketball-r1/.venv/bin/python examples/m4_smpl_contact_report.py --regression
```

GUI는 지정한 duration 후 닫힌다. headless 정량 결과와 GUI/GIF 실행 결과가 동일함을 확인했다. 기존 경고 메시지(SyntaxWarning/torch deprecation)는 기존 라이브러리에서 발생할 수 있으며 MuJoCo 물리 경고와 별개다.

## 결과 파일과 다음 작업

`four_body_holding.csv`는 PD on/off 비교, `body_XX/<stage>/telemetry.csv`는 450 Hz 접촉점·힘·공 위치/속도·Slip·상대 위치·최대 손가락 토크·발 접촉·COM margin, `summary.json`은 판정, `states.npz`는 실제 qpos 재생 상태다. `hand_structure.json` 및 관절/geom CSV가 D1 근거다. 탐색 결과는 `tuning`, `warmup_tuning`, `thumb_tuning`, `stationary_tuning`, `geometry_tuning`, `contact_damping` 하위 폴더에 분리했다.

다음 순서는 **D2의 hand pose/contact 안정화**가 먼저다.

1. 45개 XYZ hinge를 해부학적 flexion/abduction/twist와 대응시켜 허용할 손가락 목표 범위를 정한다. 엄지 opposition을 별도로 다룬다.
2. 손바닥·손가락 충돌 형상에서 공 표면 거리, 접촉 법선 분포, 공 무게를 지탱할 힘 평형을 함께 만족하는 **체형별 초기 grasp 목표**를 찾는다. 같은 Z 각을 모든 손가락에 주는 현재 프로파일을 대체한다.
3. 현실적인 palm collision proxy와 손-공 전용 감쇠를 별도 파생 모델에서 비교한다. 큰 그릇 형태의 인공 지지물로 성공을 대체하지 않는다.
4. D2 3초 이상 성공, 낮은 상대 이동/Slip, PD off 비교를 확보한 후 D3 Lifting을 검증한다.
5. D4에서는 source body motion과 접촉용 wrist/finger 목표를 명시적으로 구분한다. 기존 NPZ를 덮어쓰지 않는다. 실제 손가락 fitting은 별도 결과 파일로 검증한다.
6. 마지막으로 정상 Holding 중 손가락 펴기 전후의 실제 상태 연속성과 자유 비행을 검증하고 네 체형의 릴리스 속도를 비교한다.

## 시각화

![네 체형 초기 접촉](outputs/m4_2d_contact/four_body_initial.png)

![네 체형 접촉 Holding 실패 비교](outputs/m4_2d_contact/four_body_holding.png)

![후속 단계 진단](outputs/m4_2d_contact/stage_diagnostics.png)

각 체형 GIF: [body_01](outputs/m4_2d_contact/body_01/hold/motion.gif), [body_02](outputs/m4_2d_contact/body_02/hold/motion.gif), [body_03](outputs/m4_2d_contact/body_03/hold/motion.gif), [body_04](outputs/m4_2d_contact/body_04/hold/motion.gif).

![body_01 정지 Holding 실패 재현](outputs/m4_2d_contact/body_01/hold/motion.gif)

![body_01 실제 Motion 실패 진단](outputs/m4_2d_contact/body_01/motion/motion.gif)
