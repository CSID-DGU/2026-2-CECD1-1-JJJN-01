# M4-2D-2: SMPL+H Contact-based Lifting

2026-10-10 / macOS Apple Silicon / Python 3.12 / MuJoCo 3.15.0 / body_01

## 1. 결과와 적용 범위

`fk_fit_2`의 정적 Holding을 재현한 뒤, 기존 접촉 형상과 PD + COM Controller를 유지하여 느린 0.25m Lifting을 검증했다.

- 손목 방향을 유지한 C: 손목 0.24499m, 공 0.24531m 상승. 최대 손목 기준 상대 이동 8.38mm. 성공.
- 손목 각도를 고정한 B: 손목 0.25360m 상승했지만 공은 3.251초에 50ms 이상 손 접촉을 잃고 떨어졌다. 실패.
- A/B/C 모두 Humanoid 낙상과 MuJoCo 물리 경고 0, 양발 접촉률 100%.

공은 600g, 반지름 121mm이다. Weld, Palm Proxy, 외부 지지력, 공 초기 발사 속도를 사용하지 않았다. 초기 정렬 후 공 qpos/qvel을 변경하지 않는다. 이번 결과는 body_01의 지정된 초기 Grasp와 느린 상승 경로에 한정된다. 실제 자유투 Motion·Release·다른 체형·PPO를 구현하거나 시험하지 않았다.

활성 접촉률은 C에서도 87.48%이며, 한 스텝(2.22ms)의 접촉 공백이 남아 있다. 100% 연속 접촉이나 임의의 속도·경로에 대한 안정성을 의미하지 않는다.

## 2. 초기 Holding 재현과 Gate

M4-2D-1의 `candidates.json`에서 `fk_fit_2`를 읽는다. 원본 Humanoid XML과 기존 M4-1 환경으로 새 출력 폴더의 `contact.xml`을 생성한다. 기존 D1 코드의 `GraspTargets`와 정적 `trial`을 재사용한다.

Humanoid만 1.5초 안정화한 다음, 이전과 같은 손목 로컬 공 위치 seed와 world +Z 관통 방지 정렬을 적용한다. 초기 Humanoid 속도와 공 속도는 측정 시작 전에 한 번 0으로 설정한다. 첫 world 공 중심은 (0.298391, −0.001941, 1.060381)m이며 이전 성공 조건과 일치한다.

정적 A 재현 결과:

| 지표 | 값 |
|---|---:|
| 시험 길이 | 3.5s |
| 초기 0.2s 제외 유지 | 3.30s |
| 50ms 이상 손 접촉 이탈 | 없음 |
| 최대 상대 이동 | 8.40mm |
| 평균 수직 접촉력, 초기 0.1s 제외 | 5.8486N |
| 양발 접촉률 / COM 최소 margin | 100% / 0.1223m |
| 낙상 / MuJoCo 경고 | 0 / 0 |

Gate는 기존 3초 Holding 성공, 상대 이동 ≤20mm, 낙상 없음, 양발 접촉률 ≥99%, COM margin ≥0, 경고 없음을 요구한다. XML·NPZ·후보 파일·기존 Controller·D1 adapter의 SHA256도 저장하고 후속 실행에서 확인한다. 파일이 없거나 성공 조건/입력 해시가 맞지 않으면 IK 및 Lifting을 진행하지 않는다.

A/B/C의 첫 물리 스텝 qpos는 원소별로 정확히 같았고, B/C 초기 공 중심도 A와 1e-10m 허용 오차 안에서 일치했다. 대조 조건의 초기 배치를 따로 유리하게 바꾸지 않았다.

## 3. 오른팔 FK 분석

대상은 R_Shoulder, R_Elbow, R_Wrist의 XYZ hinge 총 9개다. XML의 이름만으로 축을 판단하지 않고, 각각 ±0.2rad를 적용하여 실제 `mj_forward` 손목 위치 변화와 world 회전 벡터 변화를 측정했다.

- 결과: `arm_fk_perturbations.csv`
- 초기 6×9 spatial Jacobian rank: 6
- 특이값: 1.9152, 1.7239, 1.6581, 0.3467, 0.3134, 0.0833
- 초기 nominal 손목 위치: (0.255861, −0.133281, 0.945900)m
- nominal 어깨 위치: (0.103906, −0.188793, 1.394477)m

손목 hinge는 손목 body 원점의 위치를 바꾸지 않지만 방향을 바꾼다. 어깨·팔꿈치 6개 hinge로 손목 위치를 조정하고 손목 3개 hinge로 방향을 보정할 수 있었다. rank 6은 이 초기 자세의 국소적 성질이며 모든 목표 위치가 도달 가능하다는 뜻은 아니다. 위치와 회전 Jacobian의 단위도 서로 다르므로 위 특이값을 순수 위치 조건수로 해석하지 않는다.

손바닥을 향하는 방향은 기존 실험과 같은 손목 로컬 −Y이고, 목표 world +Z를 유지한다. 이번 C는 법선만 맞추는 것보다 강한 조건인 전체 world 손목 회전행렬 유지 목표를 사용한다.

## 4. 팔 목표 생성과 손목 방향 보정

SciPy bounded `least_squares`로 어깨·팔꿈치의 6개 목표각을 구한다. 손목 위치 목표는 초기 nominal XY를 고정하고 world Z를 0.25m 올리는 81개 경로점이다. 잔차는 위치 오차 ×30, 초기 자세 변화 ×0.025, 이전 경로점 변화 ×0.04로 구성한다. 각 관절 목표는 초기값 ±1.6rad 안에서 찾는다. 원본 XML의 관절 제한은 변경하지 않는다.

±1.0rad 범위로는 수직 경로 일부가 충분히 도달하지 못했기 때문에 ±1.6rad 범위에서 다시 검증했다. 최종 경로의 위치 오차는 약 0.006mm 이하, nominal 방향 오차는 수치 정밀도 수준이었다. 이는 모델의 3축 elbow 구조를 사용한 기하학 경로이며 실제 사람의 해부학적 elbow 운동을 검증한 것은 아니다.

시간 계획:

| 구간 | 동작 |
|---|---|
| 0–1s | 기존 정적 Holding 목표 |
| 1–5s | minimum-jerk 보간으로 0.25m 상승 |
| 5–6.5s | 상승 끝 자세 유지 |

보간 함수는 s(u)=10u³−15u⁴+6u⁵이며 양 끝의 속도와 가속도가 0이다. IK 경로 사이 목표각은 선형 보간한다. 손가락 45개 목표각은 `fk_fit_2` 그대로 유지한다. 다른 body 목표는 source 첫 자세를 유지한다.

- B: 상승 시작 시의 손목 XYZ 목표각을 그대로 고정한다. 어깨·팔꿈치 변화가 world 손목 방향을 바꾸도록 둔다. 방향을 낮게 유지하도록 별도 보정하지 않는다.
- C: 실제 forearm(parent) 회전 R_parent를 FK로 읽어 R_local=R_parentᵀR_reference를 구한다. XYZ Euler의 동등한 두 해 중 현재 각도에 가까운 것을 손목 목표로 준다. 어깨·팔꿈치 경로는 B와 동일하다.

이 계산은 **목표 생성 adapter**이며 기존 PD gains, 토크 제한, COM 보정 로직을 수정하지 않는다. Root는 freejoint 그대로이고, 상승 중 Humanoid나 공의 위치를 강제로 이동하지 않는다.

## 5. 물리 평가

공 위치·속도는 초기 정렬 이후 `mj_step`으로만 갱신된다. 매 스텝 controller 호출 전후 공 qpos/qvel이 같음을 검사하며 `neq==0`, `qfrc_applied==0`, `xfrc_applied==0`을 확인한다. 성공 C의 저장된 궤적에서도 공과 손 이외 물체의 접촉은 없었다.

손 접촉력은 `mj_contactForce`를 world 좌표로 변환하여 공에 작용하는 방향으로 합산한다. Slip은 contact point의 공–손 상대 선속도에서 접촉 법선 성분을 제거하여 계산한다. 유효 접촉은 normal force >0.01N이다.

상승량은 상승 직전 0.9–1.0s 평균과 최종 6.0–6.5s 평균의 차이다. 따라서 공이 잠시 튀어 올라간 최고 높이만으로 성공을 판단하지 않는다. C의 마지막 정지 구간에도 공이 손 위에 남아 있었다.

Lifting 성공 조건:

- 전체 시험 중 50ms 이상 지속 접촉 이탈 없음.
- 실제 손목 상승량이 목표에서 40mm 이내.
- 실제 공 상승량 ≥ 목표의 80%, 손목–공 상승량 차이 <40mm.
- 손목 기준 상대 이동 <50mm.
- 양발 접촉률 ≥99%, COM margin ≥0, 낙상과 물리 경고 없음.
- 전체 시험 길이 완료.

| 지표 | A: 정적 Holding | B: 손목 보정 없음 | C: 손목 방향 유지 |
|---|---:|---:|---:|
| 시험 길이 | 3.5s | 6.5s | 6.5s |
| 결과 | Holding 성공 | Lifting 실패 | Lifting 성공 |
| 손목 상승량 | 약 −0.7mm | 253.60mm | 244.99mm |
| 공 상승량 | 약 −1.1mm | −932.16mm | 245.31mm |
| 최대 상대 이동 | 8.40mm | 2217.52mm | 8.38mm |
| 첫 지속 접촉 이탈 | 없음 | 3.251s | 없음 |
| 활성 접촉률 | 87.47% | 35.26% | 87.48% |
| 최대 접촉 공백 | 2.22ms | 3251.11ms | 2.22ms |
| 평균 수직 접촉력 | 5.8486N | 2.3209N | 5.8352N |
| Slip RMS, 공백을 0으로 포함 | 0.01023m/s | 0.01601m/s | 0.01153m/s |
| Slip RMS, 활성 접촉만 | 이전 Holding 정의 참조 | 0.02696m/s | 0.01233m/s |
| 오른팔 9축 tracking RMSE | 0.00749rad | 0.01981rad | 0.02449rad |
| 최대 손목 world 방향 오차 | 1.21° | 69.57° | 4.67° |
| 양발 접촉률 | 100% | 100% | 100% |
| 최소 human COM support margin | 0.1223m | 0.1214m | 0.1223m |
| Foot Slip RMS | 기존 정적 결과 참조 | 0.001025m/s | 0.001030m/s |
| 낙상 / MuJoCo 경고 | 0 / 0 | 0 / 0 | 0 / 0 |

A의 작은 높이 변화는 최초 0.5초 평균과 마지막 0.5초 평균의 차이다. A의 통계는 기존 재현 시험의 구간/정의이며 B/C의 5.5초 상승·끝 자세 구간과 같지 않다. A의 arm RMSE와 방향 오차는 저장된 약 30Hz 상태에서 초기 0.2s 제외 후 복원했다. B/C aggregate arm RMSE는 450Hz에서 상승 시작 이후 측정한다. B/C per-joint RMSE·토크는 30Hz 샘플이다. 각 비교 범위를 구분해야 한다.

B의 평균 힘과 Slip에는 공을 잃은 이후 구간이 포함된다. 이를 안정적인 Holding 힘이나 낮은 미끄러짐으로 해석하지 않는다. C는 palm normal·world up 내적의 최솟값 0.99699를 유지했다. 유지 구간 평균 손가락 수직력은 4.4964N이며 나머지를 주로 손목 capsule이 지지했다. 공 무게는 5.886N이다.

C의 관절별 RMSE/최대 토크:

| 관절 | RMSE (rad) | 최대 토크 (N·m) |
|---|---:|---:|
| R_Shoulder_x | 0.01123 | 1.4701 |
| R_Shoulder_y | 0.00572 | 0.5414 |
| R_Shoulder_z | 0.01134 | 0.3339 |
| R_Elbow_x | 0.00813 | 0.6276 |
| R_Elbow_y | 0.05363 | 4.9351 |
| R_Elbow_z | 0.02769 | 2.2920 |
| R_Wrist_x | 0.03211 | 0.3703 |
| R_Wrist_y | 0.00978 | 0.2411 |
| R_Wrist_z | 0.01665 | 1.1208 |

손가락을 포함한 54개 축의 세부 결과는 각 조건의 `per_joint_rmse.csv`에 저장했다.

![상승·접촉 비교](outputs/m4_2d_2_lifting/lifting_comparison.png)

![관절 추종 비교](outputs/m4_2d_2_lifting/arm_tracking_rmse.png)

## 6. 실패 원인과 제한

B는 팔 목표 추종 자체는 유지했으나, 손목 joint 각도 고정 때문에 forearm 회전이 손바닥 방향으로 전달됐다. 손바닥 world up 내적은 최소 0.37382로 낮아지고 방향 오차는 최대 69.57도에 달했다. 공이 지지 capsule에서 벗어나 3.251초에 지속 접촉을 잃었다. 이후 손목 상승만 계속되고 공은 바닥으로 떨어져 실패로 판정했다. 이 실패 대조 조건 다음에 C를 비교하는 것은 허용하지만 실제 자유투·Release로는 넘어가지 않는다.

C는 약 4.67도의 실제 방향 오차가 있어도 작은 상대 이동과 공 상승량을 유지했다. 어깨·팔꿈치 경로와 초기 상태가 같으므로 이번 조건에서는 world 손목 방향 보정이 접촉 Lifting에 도움이 됐다는 근거가 된다.

남은 제한:

- C는 87.5%의 활성 접촉률과 강성 접촉에 따른 한 스텝 micro-bounce가 있다.
- 고정 Grasp, 단일 체형, 느린 단일 수직 경로만 검증했다.
- 목표각과 실제 자세의 차이, 특히 Elbow_y 추종 오차가 남아 있다.
- 현재 elbow의 XYZ 구조에 대한 생리학적 운동 범위 검증은 아직 하지 않았다.
- 속도를 높이거나 손바닥을 기울이는 실제 슈팅에서도 공이 유지된다는 보장은 없다.

## 7. 실행 명령어와 GUI/GIF

실행 위치는 팀 저장소 `2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim`이다. MuJoCo 상위 폴더에서:

```bash
cd 2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim
```

A 재현 → Gate → FK/IK → B/C 비교:

```bash
PYTHONDONTWRITEBYTECODE=1 ../../../basketball-r1/.venv/bin/python \
  examples/m4_contact_lifting.py --condition all
```

C GUI(실제 macOS 실행 완료, 6.5초 후 종료):

```bash
PYTHONDONTWRITEBYTECODE=1 ../../../basketball-r1/.venv/bin/mjpython \
  examples/m4_contact_lifting.py --condition C --gui
```

B 실패 대조 GUI:

```bash
PYTHONDONTWRITEBYTECODE=1 ../../../basketball-r1/.venv/bin/mjpython \
  examples/m4_contact_lifting.py --condition B --gui
```

정적 A만 GUI로 재현:

```bash
PYTHONDONTWRITEBYTECODE=1 ../../../basketball-r1/.venv/bin/mjpython \
  examples/m4_contact_lifting.py --condition A --gui
```

`--height .25 --ramp 4 --start 1 --end-hold 1.5`가 기본이다. 높이는 0.2–0.3m, ramp는 2초 이상을 허용하며 변경한 조건은 별도 검증 결과로 봐야 한다. 여기서는 기본값만 검증했다. Gate 결과가 없는 새 출력 폴더는 먼저 `--condition all` 또는 `--condition A`를 실행한다. 동일 출력 폴더의 새 실행은 **그 폴더의 새 D2 결과**를 갱신하므로 조건 비교를 보존하려면 `--output`을 별도로 지정한다. 새 출력 폴더마다 A 재현이 필요하다.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/contact_holding_mpl \
  ../../../basketball-r1/.venv/bin/python examples/m4_contact_lifting_report.py
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/contact_holding_mpl \
  ../../../basketball-r1/.venv/bin/mjpython examples/m4_contact_lifting_report.py --gif C
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/contact_holding_mpl \
  ../../../basketball-r1/.venv/bin/mjpython examples/m4_contact_lifting_report.py --gif B
```

전신 및 손 확대 GIF는 실제 저장된 MuJoCo qpos 궤적을 렌더링한 것이다. 이 재생 과정은 시각화용이며 물리 시뮬레이션 중 공 상태를 덮어쓴 것이 아니다.

![C 성공 전신](outputs/m4_2d_2_lifting/C_full.gif)

![C 성공 손 확대](outputs/m4_2d_2_lifting/C_hand.gif)

![B 실패 전신](outputs/m4_2d_2_lifting/B_full.gif)

## 8. 테스트·보존·새 파일

기존 40개 + 새 3개 = 43개 테스트 통과. 새 테스트는 9개 arm hinge의 FK 중앙차분/실제 Jacobian 일치, IK의 상승·방향 복원 및 어깨/팔꿈치 공동 변화, 실패·누락·입력 불일치 시 Gate 거부를 검증한다.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/contact_holding_mpl \
  ../../../basketball-r1/.venv/bin/python examples/m4_smpl_contact_report.py \
  --regression --output outputs/m4_2d_2_lifting
```

원본 XML·Fitting NPZ·Controller·Weld/Kinematic·독립 실험과 기존 M4-2D-1 코드/결과를 포함한 422개 파일의 SHA256 확인 결과 변경 0개였다. 저장된 성공 C에서 손 이외의 공 지지 접촉 0개, A/B/C 초기 상태 동일 여부도 확인했다. 결과는 `preservation_audit.json`에 기록했다. Git 명령은 실행하지 않았다.

새 파일:

- `examples/m4_contact_lifting.py`
- `examples/m4_contact_lifting_report.py`
- `tests/test_contact_lifting.py`
- `M4_2D_2_CONTACT_LIFTING.md`
- `outputs/m4_2d_2_lifting/`의 CSV·JSON·NPZ·PNG·GIF·테스트 로그

주요 결과 파일:

- `condition_comparison.csv`: A/B/C 비교
- `arm_fk_perturbations.csv`, `arm_fk.json`: 실제 FK와 Jacobian
- `ik_path.csv`, `ik_path.npz`: nominal 팔 경로
- `trials/fk_fit_2/`: A 정적 재현, 원본 D1 결과와 다른 새 폴더
- `B/`, `C/`: 450Hz telemetry, 접촉 법선/힘, 관절 오차·토크, 상태 궤적, summary

## 9. 실제 자유투 연결 전 과제

1. 초기 공 위치와 손목 방향 외란, 접촉 stiffness/timestep 민감도를 확인한다.
2. 상승 속도와 수직/수평 경로를 단계적으로 바꿔 접촉 유지 범위를 정량화한다.
3. 모델의 elbow XYZ 경로가 실제 자유투의 관절 운동과 부합하는지 검토한다.
4. 기존 SPL source motion과 손목 방향 유지 목표가 충돌하는 구간을 분석하고, 원본 NPZ를 바꾸지 않는 별도 목표 adapter에서 접근한다.
5. 이후 실제 자유투 Holding Gate를 통과한 경우에만 접촉 기반 Release를 별도 구현한다.
