# M4-2D-3A: SPL Shooting Motion Integration

2026-10-10 · macOS Apple Silicon · Python 3.12 · MuJoCo 3.15.0 · body_01

## 결과와 성공 범위

**Holding에서 SPL 시작 자세로 전환하고 source 첫 0.5초에 진입하는 데는 성공했지만, 전체 SPL 동작에서 공을 유지하는 데는 실패했다.** 정상적인 슈팅, 접촉 기반 Release, 골 성공을 구현하거나 검증한 결과가 아니다.

기존 Lifting C를 재현한 A는 성공했다. B(직접 연결)는 source 재생 이전 전환 중 공을 잃었다. C(시간에 따라 변하는 제한된 손목 방향 목표)는 전환 구간을 통과했으나 source 약 1.86초에서 지속 접촉을 잃었다. 공을 잃은 이후에도 Humanoid 모션을 실행하여 실패 궤적을 기록했으며, 이를 성공 또는 의도된 릴리스로 판정하지 않았다.

## 입력과 SPL 분석

입력은 `outputs/m3_spl_fitting/spl_P0002_T0049.npz` 그대로다. 240프레임, 60FPS, 첫 프레임부터 마지막 프레임까지 239/60=3.983333초다. SHA256:

`af4a970bbac834955ea6222f36a2ef3a320b520a40e9fbf17b85b0a32910200d`

기존 `HumanoidControllerBridge`와 SMPL+H Motion Loader의 좌표계 변환, root 처리, axis-angle → XYZ hinge 변환 및 보간을 재사용한다. 각 native 프레임의 목표 qpos에 `mj_forward`를 적용하여 양쪽 어깨·팔꿈치·손목의 world 위치와 방향을 분석했다. 여기서 높이는 최종 MuJoCo world Z다. 원본 SPL skeleton이나 실제 측정 손 속도가 아니라 **Fitting 결과를 body_01에서 FK한 값**이다.

NPZ의 양쪽 손목 로컬 axis-angle과 모든 손가락 axis-angle은 정확히 0이다. 그렇다고 world 손목 방향이 고정되는 것은 아니다. 상위 어깨·팔꿈치의 회전 때문에 손목의 world 위치와 방향은 변한다. 이 NPZ만으로 실제 손목 스냅 또는 손가락 릴리스를 복원할 수 없다.

속도·가속도 탐지에는 11프레임, 3차 Savitzky–Golay 미분을 사용했다. 분석용 미분 필터이며 입력 모션을 변경하지 않는다. 방향 변화와 각속도는 회전행렬의 상대 회전 벡터로 계산한다. CSV에는 양팔의 XYZ 위치·속도·가속도·각속도와 초기 대비 방향 변화가 모두 들어 있다.

| 분석 항목 | source 시간 / 값 | 해석 |
|---|---|---|
| 주요 오른손 상승 구간 | 1.583–2.467s | 최대 상승 속도의 25% 이상인 연속 구간 |
| 주요 양의 상승 가속 구간 | 1.500–1.817s | 내부 프레임 최대 양의 가속도의 25% 이상 |
| 오른손 상승 속도 최대 | 1.883s / 1.623m/s | 슈팅 구간 후보의 근거 |
| 내부 상승 가속도 최대 | 1.567s / 6.595m/s² | 양 끝 6프레임 제외 |
| 팔꿈치 펴짐 속도 후보 | 2.417s / 513.677°/s | 손이 상승 중인 프레임에서 탐색 |
| 넓은 슈팅 구간 후보 | 1.633–2.667s | 두 속도 후보 주변 ±0.25s |
| 릴리스 시점 후보 | 1.883s, 2.417s | 서로 다른 운동학적 proxy; Ground Truth 아님 |

팔꿈치 펴짐은 어깨–팔꿈치–손목의 내각이 증가하는 것으로 정의한다. 미분의 원시 최대 가속도는 첫 프레임에서 나왔으므로 경계 효과 가능성을 표시하고 주요 가속도 후보에서 제외했다. 위 시간은 릴리스 명령이나 정답 라벨에 사용하지 않는다.

![양팔 source FK 분석](outputs/m4_2d_3a_integration/source_motion_analysis.png)

## Holding과 시작 자세의 차이

`fk_fit_2`의 어깨·팔꿈치 목표는 SPL 첫 프레임과 같다. nominal FK 손목 위치 차이는 0mm지만 손목 방향 차이는 67.490°다. Holding에서 source를 뺀 손목 XYZ 목표각 차이는 (-0.718642, -0.312652, +0.794570)rad다. 손바닥 법선(world Z 성분)은 Holding에서 1.000, source 첫 자세에서 0.383이다. 직접 전환은 공을 지지하던 방향을 크게 기울이는 동작이다.

A는 기존 Lifting C의 0.25m 상승 경로를 별도로 재현한다. B/C는 **초기 Holding에서 SPL 첫 자세로 직접 연결**한다. A의 상승 끝 자세에서 SPL로 연결하는 실험은 아니다. 따라서 A와 B/C의 궤적·시험 길이가 다르며, A는 성공 조건의 재현 및 방향 보정 방법의 기준이다. B/C 간에는 초기 상태, source 및 전환 시간 조건을 동일하게 유지했다.

## 구현과 시간 계획

새 목표 생성 adapter만 추가했다. 기존 BalancedPD의 gains, 토크 제한, COM ankle 보정은 변경하지 않는다. Root는 freejoint이며 source root 목표는 기존 bridge 경로로 읽는다. Root 위치를 핀으로 고정하거나 공의 운동으로 Humanoid를 강제 이동하지 않는다.

| simulation 시간 | B/C 동작 |
|---|---|
| 0–1s | fk_fit_2 정적 Holding |
| 1–3s | minimum-jerk 함수로 Holding → source 첫 자세 전환 |
| 3–6.983s | source 0–3.983s를 1배속 재생 |
| 이후 약 0.5s | 마지막 source 목표 유지 |

오른손 45개 손가락 목표는 전체 시험에서 fk_fit_2를 유지한다. 손가락을 펴는 Release 명령은 없다. 왼팔 및 나머지 몸은 source 목표를 사용한다. B는 source의 로컬 손목 목표를 그대로 사용한다.

C는 D2의 실제 forearm 방향 기반 손목 보정 방법을 재사용하되, world 손목 목표를 고정하지 않는다. Holding 방향 R_H와 매 시점 source FK 방향 R_S(t)의 상대 회전 벡터 δ=Log(R_HᵀR_S)를 구한다. 목표는 다음과 같다.

`R_goal(t) = R_H Exp(s(t) min(1, 15° / ||δ||) δ)`

s는 전환 구간의 minimum-jerk 함수이며 전환 이후 1이다. source 방향에 따라 목표 회전축과 회전량이 변하지만 Holding 대비 회전량은 15°로 제한한다. 실제 parent 방향으로 `R_local = R_parentᵀ R_goal`을 구하고 현재 관절각에 가까운 XYZ Euler 해를 손목 목표로 준다.

15°는 접촉 유지 가능성을 시험하기 위한 임시 제한이다. 실제 슈팅 손목 방향의 정답이나 최적값이 아니다. 이 보정은 source 손목 추종을 의도적으로 희생하며, 전체 모션 성공을 보장하지 않는다.

## 초기 Gate와 물리 조건

새 출력 폴더에서 기존 D1 정적 Holding 3.5초 및 D2 Lifting C 6.5초를 먼저 재현했다. 정적 Gate는 3초 유지, 상대 이동 ≤20mm, 양발 접촉 ≥99%, COM margin ≥0, 낙상·경고 없음을 요구한다. 입력 해시와 baseline 성공도 확인하며 실패하면 SPL 연결을 실행하지 않는다.

공은 반지름 0.121m, 질량 0.600kg의 기존 M4-1 모델이다. 기존 손바닥·손가락 capsule을 사용한다. 초기 안정화와 배치 이후 공 qpos/qvel을 덮어쓰지 않는다. 각 스텝 controller 전후 공 상태 불변, equality 수 0, 외부 generalized/body force 0, 유한 상태를 검사했다. 초기 A/B/C 첫 물리 스텝 상태는 원소별로 동일했다.

## 정량 결과

연결 Gate 구간은 실행 전에 정한 simulation 1–3.5초(전환 2초 + source 첫 0.5초)다. 전체 시험 결과는 별도로 판정한다. 지속 이탈은 이미 접촉한 공의 유효 손 접촉이 50ms 이상 없는 경우다. 접촉률이 100%가 아니어도 최대 공백이 짧을 수 있으므로 두 지표를 함께 기록했다.

A 기존 Lifting C 재현: 손목 +244.993mm, 공 +245.307mm, 최대 상대 이동 8.382mm, 접촉률 87.48%, 최대 공백 2.22ms, 평균 수직 접촉력 5.835N, 낙상·경고 0.

| 연결 구간 지표 | B 직접 연결 | C 방향 보정 |
|---|---:|---:|
| 연결 Gate | 실패 | **통과** |
| 최대 상대 이동 | 1,061.42mm | **8.36mm** |
| 손 접촉률 | 45.20% | 88.19% |
| 최대 접촉 공백 | 1,191.11ms | 2.22ms |
| 평균 수직 접촉력 | 2.854N | 5.847N |
| 9개 팔 관절 목표 추종 RMSE | 0.03861rad | 0.03684rad |
| 원본 source 팔 RMSE | 0.23316rad | 0.34321rad |
| 원본 source 어깨·팔꿈치 RMSE | 0.04156rad | 0.04293rad |
| 보정된 목표 대비 최대 손목 방향 오차 | 17.13° | 6.53° |
| 원본 source 대비 최대 손목 방향 오차 | 67.12° | 67.12° |
| 양발 접촉률 | 100% | 100% |
| 최소 COM support margin | 121.61mm | 124.91mm |

| 전체 시험 지표 | B | C |
|---|---:|---:|
| 전체 모션 Holding | **실패** | **실패** |
| 첫 지속 이탈 simulation 시간 | 2.311s | 4.860s |
| 해당 source 시간 | 0s, 전환 중 | 1.860s |
| 최대 상대 이동 | 3,601.62mm | 2,158.68mm |
| 손 접촉률 | 17.45% | 52.66% |
| 팔 목표 추종 RMSE | 0.10511rad | 0.21294rad |
| 원본 source 팔 RMSE | 0.17736rad | 0.47847rad |
| 최소 COM margin | 94.08mm | 96.27mm |
| 양발 접촉률 | 100% | 100% |
| 낙상 / MuJoCo 경고 | 0 / 0 | 0 / 0 |

상대 이동은초기 공 중심의 손목 로컬 좌표를 기준으로 측정한다. 관절 RMSE는 hinge 목표와 실제 각도의 차이다. 보정 목표 추종 오차와 원본 source 오차를 구분한다. 접촉력은 `mj_contactForce`를 world 좌표로 변환하여 공에 작용하는 손 힘을 합산한다. contact normal force >0.01N을 유효 접촉으로 본다. Slip은 접촉점 상대 선속도의 접선 성분이며 `telemetry.csv`에 기록한다. COM margin은 기존 지지 영역 평가의 기하학적 지표로, 동적 안정성의 충분조건은 아니다.

초기 정적 Holding 구간(0–1s)의 방향 진단은 Holding 기준이며, 원본 source 방향 오차 비교는 연결 이후(1s부터)의 표를 사용한다.

물리 telemetry는 450Hz, 관절별 오차·실제 actuator torque와 접촉 법선은 30Hz다. `per_joint_rmse.csv`는 9개 오른팔 및 45개 오른손 손가락의 관절별 결과를 포함한다. 실패 이후 구간도 전체 지표에 포함되므로 Holding 중 성능과 동일하게 해석하지 않는다.

![공 위치·속도와 접촉 비교](outputs/m4_2d_3a_integration/integration_comparison.png)

## 손목 보정 범위 민감도

| Holding 대비 제한 | 연결 Gate | 최대 연결 상대 이동 | 첫 이탈 source 시간 | 전체 유지 |
|---|---|---:|---:|---|
| 5° | 통과 | 10.54mm | 1.987s | 실패 |
| 10° | 통과 | 9.41mm | 1.896s | 실패 |
| 15° | 통과 | 8.36mm | 1.860s | 실패 |
| 20° | 통과 | 8.36mm | 1.709s | 실패 |

제한을 작게 하면 이 조건에서 이탈을 늦췄지만 source와의 관절 오차가 커졌다. 네 경우 모두 전체 실패이므로 손목 방향 제한만으로 해결했다고 볼 수 없다.

## 실패 원인과 후속 과제

B는 팔이 거의 정지한 전환 구간에서 손바닥이 기울면서 공이 미끄러졌다. 첫 이탈 직전 공의 상대 이동은 이미 약 97mm였다.

C는 source 상승 가속 구간을 지나 손목 최대 상승 속도 시점에 접근하며 이탈했다. 이탈 약 50ms 전에는 손 접촉 1개, 수직 접촉력 4.53N, 상대 이동 45.1mm, 손목 보정 목표 대비 방향 오차 26.98°였다. 첫 이탈 시에는 상대 이동 88.0mm, 방향 오차 33.50°였고 실제 손목 속도는 1.085m/s였다. 접촉 상실 이전부터 방향 추종이 악화되었다. 팔의 빠른 변화, wrist 보정 추종 한계 및 접촉 구조를 함께 조사해야 하며 공 관성 하나만을 원인으로 확정할 수 없다.

추가로 30Hz 목표 로그에서 elbow XYZ 일부는 33ms당 약 0.55–0.61rad 변했다. wrist 보정 목표에는 simulation 5.136초에서 약 3.06–3.25rad의 큰 변화가 발견됐다. 이는 최초 공 이탈(4.86초) **이후**이므로 최초 실패를 이 점프 하나로 설명할 수 없다. Euler 해 선택 및 원본 회전 변화에 대한 연속성 검증이 필요하다. 저장된 샘플 간 변화이며 모든 450Hz 스텝의 최대값을 뜻하지 않는다.

다음 단계에서 먼저 할 일:

1. 접촉 상실 전 source 1.5–1.9초의 목표각 변화, 실제 torque/토크 포화 및 방향 오차를 함께 분석한다.
2. 목표 생성 adapter에서 연속적인 회전 해와 관절 목표를 검증한다. 기존 Controller 변경과 분리한다.
3. 실제 손바닥 지지 방향·접촉 분포를 고려한 시간별 wrist 목표를 설계하고 원본 source 오차와 접촉 유지 사이의 trade-off를 평가한다.
4. 필요하면 원본 NPZ를 보존한 별도 시간 재매개변수화 조건으로 동적 유지 한계를 조사한다.
5. 전체 연결 Holding Gate를 통과한 뒤에만 M4-2D-3B 가속 및 M4-2D-4 Release 검증으로 진행한다.

## 실행 명령어

실행 위치는 팀 저장소의 `2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim`이다. 기존 로컬 가상환경과 XML을 참조하고 팀 저장소로 복사하지 않는다.

```bash
# 분석만 실행
PYTHONDONTWRITEBYTECODE=1 ../../../basketball-r1/.venv/bin/python \
  examples/m4_spl_contact_integration.py --analyze-only

# 정적 Gate, Lifting C baseline, B/C 물리 시험
PYTHONDONTWRITEBYTECODE=1 ../../../basketball-r1/.venv/bin/python \
  examples/m4_spl_contact_integration.py --condition all

# 기존 Gate 결과를 사용하여 C GUI 실행
PYTHONDONTWRITEBYTECODE=1 ../../../basketball-r1/.venv/bin/mjpython \
  examples/m4_spl_contact_integration.py --condition C --gui

# B GUI: 같은 명령의 --condition을 B로 변경
# 그래프·CSV·보존 감사
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/contact_holding_mpl \
  ../../../basketball-r1/.venv/bin/python examples/m4_spl_contact_integration_report.py

# 저장 상태로 GIF 생성: B도 --gif B로 동일하게 실행
PYTHONDONTWRITEBYTECODE=1 ../../../basketball-r1/.venv/bin/mjpython \
  examples/m4_spl_contact_integration_report.py --gif C

# 기존 전체 회귀 + 신규 검증
PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore \
  ../../../basketball-r1/.venv/bin/python examples/m4_smpl_contact_report.py \
  --regression --output outputs/m4_2d_3a_integration
```

macOS의 mjpython으로 C GUI 실행을 완료했고 headless 결과와 일치했다. C GUI도 실패 궤적을 끝까지 보여준다. 실패 이후 공이 날아가거나 떨어지는 장면은 정상 Release가 아니다. GUI를 중간에 닫으면 해당 조건 결과가 짧은 시험으로 다시 저장되므로 정량 비교를 재생성하려면 끝까지 실행하거나 headless 시험을 다시 실행한다.

## 검증과 생성 파일

회귀 및 신규 테스트 **47개 통과**. 신규 테스트는 회전 보간/제한, 연결 경계 연속성, finger Grasp 유지, 원본 목표 보존 및 controller의 공 상태 미변경을 확인한다. 보호된 기존 코드·결과·XML·NPZ 등 **468개 파일의 SHA256 변경 0개**, A/B/C 초기 상태 동일. Git 작업 없음.

신규 파일:

- `examples/m4_spl_contact_integration.py`
- `examples/m4_spl_contact_integration_report.py`
- `tests/test_spl_contact_integration.py`
- `M4_2D_3A_SPL_MOTION_INTEGRATION.md`
- `outputs/m4_2d_3a_integration/`: source FK/분석 JSON·CSV, 새 contact XML, Holding Gate, baseline, B/C 및 제한 민감도 물리 로그, 관절별 RMSE/torque, 접촉 법선, 그림, GIF, 보존 감사.

초기 비연속 제한 방식의 탐색 결과는 새 출력 폴더의 `diagnostics/pre_smooth_blend`에 보존했다. 위 결과표는 수정된 부드러운 전환 방식의 기본 C 및 민감도 시험 결과만 사용한다. 기존 D1/D2 결과는 덮어쓰지 않았다.

GUI 및 저장 상태 GIF:

- [B 직접 연결 실패 GIF](outputs/m4_2d_3a_integration/B_motion.gif)
- [C 연결 통과 후 전체 실패 GIF](outputs/m4_2d_3a_integration/C_motion.gif)

Issue 체크용 상태: SPL 분석, 시작 자세 차이 분석, 부드러운 연결 adapter, 시간별 wrist 보정, A/B/C 비교, 정량 CSV 및 회귀 검증은 완료. **전체 SPL 모션 접촉 유지와 이후 Shooting Acceleration/Release는 미완료**다.
