# M4-2D-1: 손가락 운동학 및 Grasp 자세 분석

2026-10-10 / macOS Apple Silicon / Python 3.12 / MuJoCo 3.15.0

## 결과

body_01의 원래 손바닥·손가락 capsule만 사용한 `fk_fit_2` 후보가 3.5초 정적 Holding 시험에 통과했다. 초기 0.2초를 제외한 유지 시간은 3.30초이며, 공의 손목 기준 최대 이동은 8.40mm이다. 같은 초기 자세·위치에서 오른손 손가락 PD를 끄면 0.424초에 50ms 이상 접촉을 잃는다. 이전 Z 굽힘 목표는 0.633초에 접촉을 잃는다.

성공 범위는 단일 body_01·고정 손목 방향·지정된 초기 공 위치의 정적 시험이다. 다른 체형, Lifting, 실제 자유투, Release는 이번 단계에서 실행하거나 구현하지 않았다. 매 순간 접촉을 유지한 것은 아니다: 활성 접촉률은 87.47%, 초기 0.2초 이후 최장 접촉 공백은 한 스텝인 2.22ms다. 따라서 동적 Grasp 성공으로 확대 해석하지 않는다.

## 1. 측정 방법과 좌표계

원본 `body_01.xml`과 기존 M4-1 농구공 환경을 참조한다. 기존 M3 PD + COM Controller와 `HumanoidControllerBridge`는 그대로 import한다. SPL NPZ의 첫 body 자세에서 손목을 palm-up으로 설정한다. 손바닥을 향하는 방향은 오른손 손목 로컬 −Y로 정의했으며, 실제 물리 시험의 world +Z와 내적은 0.9998 이상이다. 로컬 −X는 대체로 손가락 쪽, Z는 손바닥의 가로 방향이다. 이는 모델의 접촉 구조를 위한 기준이며 사람 해부학 좌표계의 정답을 의미하지 않는다.

15개 손가락 분절 × XYZ = 45개 hinge 각각을 나머지 관절을 고정한 상태에서 −0.2, +0.2rad로 변경하여 `mj_forward`를 실행했다. 총 90개 측정은 `axis_perturbations.csv`에 world/손목 좌표계 손끝 변화와 world 회전축을 저장한다. 손끝은 해당 손가락 마지막 capsule의 원위 중심선 끝점이다. 피부 표면·실제 손톱 위치가 아니다.

분절의 원위 방향 b는 실제 capsule 중심선으로, 손바닥 법선 n은 실제 손목 FK로 구했다. Flexion 후보 방향은 normalize(b×n), Twist는 b, Abduction은 두 방향의 직교 방향이다. 실제 `data.xaxis`와의 내적을 비교해 3축의 역할을 배정했다. ±0.2rad의 끝점 움직임을 CSV·GIF에서 확인했고, 모든 45개 축에 대해 미소 중앙차분 FK가 `axis × (tip − anchor)`와 일치하는 별도 테스트도 통과했다.

아래 표의 괄호 값은 실제 축과 추정 역할 방향의 정렬 정도다. 역할은 기본 자세 근처의 기하학적 추정이며, 큰 회전 이후에도 같은 해부학적 역할을 보장하지 않는다. 특히 정렬 값이 낮은 엄지·소지 근위 분절은 여러 축의 조합이 필요하다.

| 분절 | Flexion | Abduction | Twist |
|---|---|---|---|
| R_Index1 | +Z (1.000) | −Y (0.942) | −X (0.942) |
| R_Index2 | +Z (0.907) | +X (0.859) | −Y (0.947) |
| R_Index3 | +Z (0.972) | +X (0.944) | −Y (0.971) |
| R_Middle1 | +Z (0.986) | −Y (0.879) | −X (0.867) |
| R_Middle2 | +Z (0.965) | +X (0.938) | −Y (0.973) |
| R_Middle3 | +Z (0.990) | +X (0.977) | −Y (0.987) |
| R_Pinky1 | +Z (0.945) | +X (0.673) | −Y (0.712) |
| R_Pinky2 | +Z (0.945) | +X (0.919) | −Y (0.973) |
| R_Pinky3 | +X (0.832) | −Z (0.808) | −Y (0.971) |
| R_Ring1 | +Z (0.999) | −Y (0.838) | −X (0.837) |
| R_Ring2 | +Z (0.952) | +X (0.937) | −Y (0.984) |
| R_Ring3 | +X (0.838) | −Z (0.825) | −Y (0.984) |
| R_Thumb1 | +Z (0.870) | −Y (0.808) | −X (0.703) |
| R_Thumb2 | +Z (0.963) | −Y (0.954) | −X (0.918) |
| R_Thumb3 | +Z (0.932) | −Y (0.970) | −X (0.904) |


+Z를 일괄 Flexion으로 쓰는 이전 가정은 대부분의 분절에서는 근사적으로 타당하지만, Ring3와 Pinky3에서는 +X가 더 가까웠다. Ring3/Pinky3는 기본 분절이 손바닥 법선과 거의 평행하므로 b×n의 방향이 작은 형상 차이에 민감하다. 따라서 이 +X 추정은 현재 형상에서의 기하학적 역할이며 해부학적 굽힘 축이 +X라고 확정하는 결과가 아니다. `bone_palm_normal_absdot`도 CSV에 기록한다. 또한 기본 capsule 자세 자체가 이미 굽어 있어, 큰 농구공을 지탱할 때 무조건 더 굽히는 것이 유리하지 않았다.

![FK 끝점 변화](outputs/m4_2d_1_hand/fk_tip_displacement.png)

## 2. 엄지 Opposition

엄지 근위 XYZ를 각각 −0.35, 0, +0.35rad로 바꾼 27개 조합을 `thumb_opposition.csv`에 저장했다. 기본 엄지–검지 끝점 거리는 54.82mm이며, 이 제한된 격자에서 (+0.35, −0.35, 0) 조합은 35.69mm로 줄였다. 검지에 가까워지는 동작은 하나의 Z 굽힘만으로 설명하기 어렵다.

농구공 접촉에는 검지 끝점과의 최소 거리보다 공 둘레에서의 지지 방향이 중요하다. 선택된 Grasp의 엄지 근위 XYZ는 (−0.331, +0.247, −0.158)rad이며, 이 조합은 공을 지지하는 Thumb2 접촉을 형성한다. 손가락 끝점을 서로 가깝게 만드는 조합과 농구공에 적합한 Opposition은 별개이다. Thumb1 Y만 ±0.35rad로 바꾸는 비교 후보도 실행했다.

![엄지 조합](outputs/m4_2d_1_hand/thumb_opposition.png)

## 3. 실제 충돌 형상과 공의 거리

오른손에는 R_Wrist capsule 1개, 손가락 capsule 15개가 있다. 원본 손 형상·질량·관성·hinge·actuator는 변경하지 않았다. R_Wrist capsule 반지름은 32mm, 중심선 길이는 약 53.84mm다. 손가락 반지름은 약 5.4–12.9mm다. 농구공 반지름 121mm, 질량 600g 및 M4-1 thin-shell 관성 0.0058564 kg·m²를 유지했다.

각 capsule 중심선의 최근접점 c와 반지름 r에 대해 sphere 표면 간격은 ||ball_center−c||−r−0.121로 계산한다. 이 값과 ball을 향하는 기하학적 법선을 `initial_geometry.csv`에 기록한다. 실제 활성 contact frame 법선·normal force·gap은 `contacts.csv`에 별도로 기록한다. 기하학적 거리 0에 가깝다는 이유만으로 접촉력이 있다고 판단하지 않는다.

손의 충돌 bit 32와 공 affinity 48만 기존 contact 실험의 설정으로 적용한다. 공–바닥·림·백보드 설정은 M4-1 그대로다. 이번 성공 후보에는 Palm Proxy, 추가 접촉 pair, 감쇠 변경이 없다. Equality는 0개이며 외부 힘도 없다.

선택된 자세의 활성 접촉은 주로 R_Wrist, Index3, Middle3, Thumb2다. 정상 구간에서 공에 향하는 평균 법선은 각각 아래와 같다. 힘은 **그 형상이 활성 상태일 때의 평균 normal force**이므로 행별 힘을 합쳐 평균 총 수직력으로 사용하면 안 된다. 접선 마찰력과 활성 빈도도 다르다.

| 형상 | 평균 법선 XYZ (world) | 활성 시 평균 normal force (N) |
|---|---|---:|
| R_Wrist | (0.358, 0.066, 0.931) | 1.426 |
| Index3 | (−0.162, −0.412, 0.897) | 2.503 |
| Middle3 | (0.021, −0.558, 0.830) | 1.343 |
| Thumb2 | (−0.042, 0.345, 0.938) | 4.274 |

엄지와 검지·중지의 법선에 반대 방향 Y 성분이 나타나며 모두 위쪽 성분을 갖는다. 이는 단일 둥근 받침보다 공의 가로 이동을 제한하는 데 유리한 지지 배치다. 정식 force-closure 증명이나 손 전체 파지 검증은 수행하지 않았다.

이전 Z 후보에서 초기 0.1–0.25초 공 중심의 손 기준 상대 속도 평균은 대략 (+0.039, +0.009, +0.066)m/s다. 이후 capsule 경계에서 벗어나 중력 방향으로 낙하한다. 선택 후보는 약 (−0.015, +0.010, +0.022)m/s로 초기 방향이 바뀌며 상대 이동이 작은 범위에 머문다. 실제 접촉점의 접선 Slip은 별도 scalar RMS로 측정한다.

기존 Palm Proxy는 작은 평면 접촉 면을 추가하지만 여러 방향의 법선·엄지 Opposition·손가락 제어를 대신하지 못한다. 과거 proxy 시험도 3초 유지에 실패했다. 이번 단계에서 기존 proxy 코드나 결과를 변경하지 않았다.

![거리와 접촉 법선](outputs/m4_2d_1_hand/contact_geometry.png)

## 4. Grasp 후보 생성

원본 XML의 ±π 가동 범위를 그대로 목표각으로 쓰지 않고, 새 실험 목표에 제한을 적용했다. 추정 Flexion은 −0.25~1.10rad, 근위 Abduction은 ±0.25rad, 엄지 근위의 비-Flexion 조합은 ±0.65rad이다. 기타 Abduction/Twist는 0으로 둔다. 이는 이미 굽은 SMPL capsule 기본 자세에 대한 **보수적인 실험 범위**이며 임상적으로 검증된 해부학 관절 제한은 아니다. 로컬 공개 Shadow Hand XML의 제한된 굽힘·벌림 범위도 참고했지만 관절별 범위를 직접 이식하지 않았다.

SciPy bounded `least_squares`로 21개 손가락 역할 변수와 공 초기 상대 위치 XYZ 3개를 조정했다. 손목은 palm-up으로 고정하고, 손바닥 및 손가락 원위 capsule의 sphere 표면 간격, 모든 capsule의 관통 페널티, 아래쪽 지지 법선, 작은 자세 정규화를 잔차에 포함한다. 세 가지 시작 위치로 기하학 최적화를 수행했다. 안정성 판단은 이 cost가 아니라 별도의 물리 시험으로 한다.

총 8개 후보를 1초 동안 선별했다. 모든 후보의 `contacts.csv`, `joint_torques.csv`, `telemetry.csv`, `initial_geometry.csv`, `summary.json`, `states.npz`를 별도 폴더에 저장했다. 같은 1초까지 유지한 후보 중 상대 이동이 가장 작은 fk_fit_2를 선택했다.

| 후보 | 1초 시험의 첫 지속 접촉 이탈 (s) | 최대 접촉 수 | 최대 상대 이동 (mm) |
|---|---:|---:|---:|
| old_z_reference | 0.6333333333333351 | 2 | 817.9 |
| fk_fit_1 | 0.3533333333333333 | 1 | 1008.4 |
| fk_fit_2 | 1초까지 없음 | 5 | 8.4 |
| fk_fit_3 | 1초까지 없음 | 5 | 8.6 |
| flex_scale_0 | 0.6244444444444461 | 3 | 1071.9 |
| flex_scale_0.5 | 1초까지 없음 | 5 | 26.2 |
| thumb_opposition_-0.35 | 0.8355555555555586 | 3 | 452.8 |
| thumb_opposition_+0.35 | 1초까지 없음 | 4 | 29.6 |


fk_fit_2의 초기 위치 탐색 seed는 손목 로컬 (−0.10622, −0.03473, +0.02998)m다. 여기서 실제 물리적으로 안정화된 손 위에 sphere를 world +Z 방향으로 올려 초기 관통이 없도록 한 번 정렬한다. 따라서 위 seed는 최종 공 중심 offset과 같지 않다. 최종 world 공 중심은 (0.298391, −0.001941, 1.060381)m이며 ball freejoint 초기 속도는 0이다. 초기 정렬 이후에는 공 qpos/qvel을 쓰지 않는다.

선택한 목표각:

| 손가락 | 분절 1 XYZ (rad) | 분절 2 XYZ | 분절 3 XYZ |
|---|---|---|---|
| Index | +0.000, +0.006, -0.250 | +0.000, +0.000, -0.189 | +0.000, +0.000, -0.092 |
| Middle | +0.000, -0.023, -0.233 | +0.000, +0.000, -0.138 | +0.000, +0.000, -0.061 |
| Ring | +0.000, -0.142, -0.250 | +0.000, +0.000, -0.160 | +0.227, +0.000, +0.000 |
| Pinky | +0.147, +0.000, -0.026 | +0.000, +0.000, +0.004 | +0.241, +0.000, +0.000 |
| Thumb | -0.331, +0.247, -0.158 | +0.000, +0.000, +0.103 | +0.000, +0.000, +0.144 |


Index/Middle/Ring 일부 Flexion 값이 음수인 이유는 구부러진 기본 형상을 약간 펴서 큰 구에 맞추기 때문이다. uniform +Z 굽힘을 늘리는 방법과 달랐다. 목표 후보만 새 adapter에서 제공하며 기존 PD gains·COM 로직은 바꾸지 않았다.

## 5. 3초 Holding 검증

Humanoid 단독 1.5초 안정화 후 공을 초기 배치하고 3.5초 물리 시뮬레이션한다. dt=1/450s, 기존 gain_scale=0.977857이다. 손가락 PD off 비교는 동일한 안정화·공 초기 상태에서 측정 시작 후 오른손 45개 actuator 토크만 0으로 만든다.

Holding 판정은 초기 0.2초를 제외하고 3초 이상 유지하며 50ms 이상의 접촉 이탈이 없어야 한다. 공 상대 이동·접촉력·PD off 결과를 함께 확인한다. 한 스텝의 micro-bounce는 연속 접촉 성공으로 취급하지 않고 별도 보고한다.

| 지표 | 이전 Z 후보 | fk_fit_2, PD on | 동일 후보, PD off |
|---|---:|---:|---:|
| 3초 Holding | 실패 | 성공 | 실패 |
| 첫 50ms 이상 접촉 이탈 (s) | 0.6333 | 없음 | 0.4244 |
| 초기 0.2초 제외 유지 (s) | 0.4333 | 3.3000 | 0.2244 |
| 최대 활성 접촉 수 | 2 | 5 | 4 |
| 유지 구간 평균 수직 접촉력 (N) | 5.638 | 5.849 | 4.480 |
| 손가락 평균 수직력 (N), 유지 구간 | 5.169 | 4.957 | 2.939 |
| 최대 상대 이동 (mm), 전체 3.5초 | 2495.2 | 8.40 | 2191.6 |
| 최대 손가락 PD 토크 (N·m) | 0.2203 | 0.1978 | 0 |
| 양발 접촉률 | 100% | 100% | 100% |
| 낙상 / MuJoCo 경고 | 0 / 0 | 0 / 0 | 0 / 0 |

수직력의 평균 구간은 초기 0.1초부터 첫 지속 이탈 전까지이며 실패 조건에서는 시간 구간이 짧다. 유지 시간의 0.2초 제외 기준과 구분한다. 공 무게는 0.6×9.81=5.886N이고 선택 후보의 평균 힘은 이에 가깝다. 손가락이 약 4.96N, 손목 capsule이 나머지를 지지한다. 선택 후보의 접촉 Slip RMS는 0.01023m/s, 최소 human COM support margin은 0.1223m, 최대 pelvis tilt는 10.94도이다.

활성 접촉률 87.47% 및 최대 무접촉 공백 2.22ms는 현재 강성 접촉의 미세 이탈을 보여준다. 최대 활성 관통은 0mm이며 hand geom의 1mm margin 때문에 실제 표면 사이 작은 양의 간격에서도 힘이 생길 수 있다. 이는 물리적 피부 순응성의 정밀한 모델이 아니다. 따라서 stiffness/timestep 민감도, 초기 배치 외란, 더 긴 지속 시간에 대한 안정성은 아직 미검증이다.

![Holding 비교](outputs/m4_2d_1_hand/holding_comparison.png)

![선택 후보](outputs/m4_2d_1_hand/fk_fit_2.gif)

![PD off 실패](outputs/m4_2d_1_hand/fk_fit_2_no_pd.gif)

## 6. GUI와 GIF 실행

실행 위치: 팀 저장소 `2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim`. MuJoCo 상위 디렉터리에서 먼저:

```bash
cd 2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim
```

분석 및 선별·3.5초 재검증:

```bash
PYTHONDONTWRITEBYTECODE=1 ../../../basketball-r1/.venv/bin/python \
  examples/m4_hand_kinematics.py --mode analyze
```

선택 후보 GUI(이번 환경에서 실제 실행 완료):

```bash
PYTHONDONTWRITEBYTECODE=1 ../../../basketball-r1/.venv/bin/mjpython \
  examples/m4_hand_kinematics.py --mode hold --candidate fk_fit_2 --gui
```

동일한 초기 상태에서 손가락 PD off:

```bash
PYTHONDONTWRITEBYTECODE=1 ../../../basketball-r1/.venv/bin/mjpython \
  examples/m4_hand_kinematics.py --mode hold --candidate fk_fit_2 --no-finger-pd --gui
```

45개 축의 ±0.2rad FK 동작을 GUI에서 순차 표시:

```bash
PYTHONDONTWRITEBYTECODE=1 ../../../basketball-r1/.venv/bin/mjpython \
  examples/m4_hand_kinematics.py --mode axes --gui
```

이 FK GUI는 **운동학 분석용**이며 물리 Holding 시험이 아니다. 변경 중인 관절 이름과 각도는 터미널에 표시한다. GIF에서는 X 빨강, Y 초록, Z 파랑의 actual world hinge axis를 arrow로 표시한다. 손가락별 모든 분절/축을 담은 Index/Middle/Ring/Pinky/Thumb.gif가 생성된다.

```bash
PYTHONDONTWRITEBYTECODE=1 ../../../basketball-r1/.venv/bin/mjpython \
  examples/m4_hand_kinematics.py --mode axes
PYTHONDONTWRITEBYTECODE=1 ../../../basketball-r1/.venv/bin/mjpython \
  examples/m4_hand_kinematics.py --mode render --candidate fk_fit_2
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/contact_holding_mpl \
  ../../../basketball-r1/.venv/bin/python examples/m4_hand_kinematics_report.py
```

GUI는 지정한 duration(기본 3.5초) 후 닫힌다. `--duration`으로 늘릴 수 있지만 여기 보고한 검증은 3.5초다. 기존 M4-2D 실행 명령은 그대로 유지된다.

## 7. 테스트 및 보존 확인

기존 테스트 37개 + 새 FK/거리 검증 3개 = 40개 통과했다. 새 테스트는 45축 중앙차분과 spatial FK derivative 비교, capsule 최근접점 clamp, 실제 축 매핑과 원상 복구를 검증한다. 물리 Holding 성공 여부는 별도 실험 CSV로 확인한다.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/contact_holding_mpl \
  ../../../basketball-r1/.venv/bin/python examples/m4_smpl_contact_report.py \
  --regression --output outputs/m4_2d_1_hand
```

Git에서 제외된 기존 로컬 XML은 참조만 한다. 회귀 runner는 테스트의 자산 경로만 메모리에서 원본 환경으로 연결하며 자산 파일을 복사하거나 테스트 코드를 바꾸지 않는다.

원본 Controller·Fitting NPZ·Humanoid XML·독립 실험·기존 M4-2D 코드 및 결과 327개 파일의 SHA256을 검증했고 변경은 0개이다. `preservation_audit.json`에 결과가 있다. 기존 Fitting NPZ SHA256은 af4a970bbac834955ea6222f36a2ef3a320b520a40e9fbf17b85b0a32910200d이다. Git 명령은 실행하지 않았다.

새 파일:

- examples/m4_hand_kinematics.py
- examples/m4_hand_kinematics_report.py
- tests/test_hand_kinematics.py
- M4_2D_1_HAND_KINEMATICS.md
- outputs/m4_2d_1_hand/ 이하 CSV·JSON·NPZ·PNG·GIF·회귀 로그

## 8. 실패 원인과 다음 검증

이전 uniform +Z 목표는 일부 말단 축과 맞지 않았고 기본 capsule 곡률을 더 굽혀 큰 공에 대한 지지 위치를 제한했다. 후보 1은 기하학적 거리 최적화 후에도 최대 접촉 1개와 편향된 지지로 빠르게 실패했다. 엄지 Y를 단독 변경한 −0.35rad 후보도 0.836초에 접촉을 잃었다. 접촉 수나 기하학 cost만으로 성공을 판단할 수 없다는 결과다.

이번 개선은 손가락 자세 **및 공의 초기 배치**가 함께 바뀌었다. 두 변화의 기여를 완전히 분리한 인과 실험은 아니다. PD off 비교는 동일 초기 배치에서 손가락 제어의 필요성을 확인한다. 엄지 개별 ablation, 초기 위치 외란, 실제 해부학적 범위 검증은 아직 하지 않았다.

권장 후속 순서:

1. body_01에서 초기 공 위치·손목 경사 외란과 접촉 timestep/stiffness 민감도를 확인한다.
2. 측정된 signed flexion 축과 보수적 목표 범위를 source pose에서의 해부학적 기준과 추가 비교한다.
3. body_02~04의 각 손 구조에서 FK와 공 초기 상대 위치를 다시 정렬해 D2를 통과시킨다.
4. 이후 별도 요청 범위에서 D3 Lifting으로 넘어간다. 이번 결과를 실제 자유투 Motion·Release 성공으로 취급하지 않는다.
