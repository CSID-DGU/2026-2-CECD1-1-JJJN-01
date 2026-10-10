# M4-2 확장: Weld 기반 물리적 Holding과 Release

## 결론

운동학적 부착을 유지하면서 Weld 기반 Holding을 추가했다. body_01을 먼저 검증한 뒤 네 체형에서 공 없음(A), 운동학적 부착(B), Weld 부착(C)를 비교했다. 12개 표준 실험 중 B/C는 600 g / 반지름 121 mm의 M4-1 공을 그대로 사용했으며, 낙상·MuJoCo 물리 경고가 없었다. A와 B의 Humanoid 상태 및 토크는 동일하고 C에서는 공의 하중에 따른 토크와 관절 추종 변화가 측정되었다.

이번 결과는 **물리적 하중 전달과 상태를 보존하는 릴리스 검증**이다. 실제 자유투 득점, 손가락 그립, 체형별 Retargeting이나 PPO의 구현 결과는 아니다.

## 방식 비교

| 조건 | Holding | 공 상태 쓰기 | 공의 하중 전달 | Release |
|---|---|---|---|---|
| A | 공 없음, 기존 Humanoid | 없음 | 없음 | 없음 |
| B | 기존 운동학적 부착 | 릴리스 이전 step마다 | 없음 | 실제 손 Jacobian에서 계산한 마지막 상태로 전환 |
| C | 손-공 Weld equality | 초기 정렬 1회만 | 질량·관성·제약력이 실제 solver를 통해 전달 | `data.eq_active`만 해제, 실제 공 상태 유지 |

B의 기존 실행 코드 `examples/m4_ball_release.py`는 그대로 사용할 수 있다. 공통 A/B/C 비교 코드는 `examples/m4_physical_holding.py`다. Controller는 모든 조건에서 기존 M3 `BalancedPD` + COM을 호출한다.

## Weld 설정과 초기 정렬

원본 Humanoid XML에서 기존 M4-1 `build_xml`로 별도 결과 XML을 만든 뒤 equality만 추가한다. 기존 M4-1 XML이나 원본 Humanoid XML을 덮어쓰지 않는다.

```xml
<equality>
  <weld name="hand_ball_weld"
        body1="R_Wrist" body2="basketball"
        anchor="0 0 0"
        relpose="-0.07 -0.19 0 1 0 0 0"
        active="false"
        solref=".02 1" solimp=".95 .99 .001"
        torquescale=".121" />
</equality>
```

`relpose`는 손 body 로컬 좌표에서의 ball anchor 위치와 상대 quaternion(wxyz)이다. 상대 회전은 identity로 설정했다. 기본 오른손 offset은 `[-0.07,-0.19,0]` m, 왼손은 X만 반전한다. `--offset`으로 위치를 지정할 수 있다. `solref`의 time constant 20 ms와 damping ratio 1, `torquescale` 0.121 m를 사용했다. Weld는 MuJoCo의 **soft equality constraint**로 작용한다. `torquescale`은 회전 잔차의 상대 중요도를 조절한다. [MuJoCo 공식 Weld 문서](https://mujoco.readthedocs.io/en/stable/XMLreference.html#equality-weld)

초기 절차:

1. 체형별 원본 모델에서 Fitting NPZ의 첫 자세를 로딩하고 기존 floor alignment를 적용한다.
2. 실제 초기 손의 `xpos/xmat`에서 공 중심 `p_hand + R_hand @ offset`과 방향을 계산한다.
3. 공 qpos 및 qvel을 이 상태로 **한 번** 정렬한다. 초기 속도도 손의 선속도/각속도와 일치시킨다.
4. 공-손 충돌을 끈 상태에서 Weld를 활성화하고 `mj_forward`로 equality 잔차를 확인한다.
5. 이후 qpos/qvel은 MuJoCo solver만 적분한다.

체형별 손의 world 위치와 초기 root 높이가 다르므로 각 체형의 실제 FK 결과로 초기 공 위치를 다시 계산한다. 네 체형·양손의 초기 정렬을 테스트했다. 동일 offset 적용은 체형별 Retargeting 알고리즘을 뜻하지 않는다.

## 릴리스

지정 시각 이상인 첫 물리 step에서 다음 동작만 수행한다.

```python
# qpos/qvel은 이 시각의 실제 물리 상태 그대로 둔다.
data.eq_active[weld_id] = 0
# 공-손 충돌 mask 활성화
mujoco.mj_forward(model, data)
```

MuJoCo는 실행 중 `mjData.eq_active` 변경을 지원한다. [공식 equality active 설명](https://mujoco.readthedocs.io/en/stable/XMLreference.html#equality-connect-active)

릴리스 위치는 실제 ball freejoint qpos, 선속도는 실제 world qvel, 각속도는 ball-local qvel을 world로 변환하여 기록한다. **손 Jacobian으로 다시 덮어쓰거나 목표 발사 속도를 넣지 않는다.** 요청 시각 대비 물리 step 양자화 오차는 최대 약 2.222 ms다. 기본 시점은 2.5초다. 동일 시각의 해제 전후 위치·선속도·각속도 jump와 릴리스 후 overwrite 횟수를 검사한다.

## Controller와 하중 전달

기존 M3 Controller 파일은 수정하지 않았다. 원래 Humanoid 모델에 실제 qpos/qvel을 동기화하는 bridge로 같은 PD + COM과 원래 체형별 gain_scale을 사용한다. 새 공은 별도 freejoint, 기존 actuator는 153개 그대로다. 통합 모델의 equality solver가 공과 손 사이의 힘/모멘트를 적용하고, 이후 실제 Humanoid 상태가 Controller에 피드백된다.

Controller의 내부 예측/COM 보정은 기존 Humanoid 모델을 기준으로 유지된다. 공을 포함한 새 중력 보상이나 Weld inertia를 반영하는 feedforward Controller를 만들지 않았다. 따라서 C에서 얻은 결과는 **원래 Controller가 새로운 물리적 부하에 반응한 결과**다.

Weld 힘은 equality rows만 선택한 `efc_force`에 `mj_mulJacTVec`를 적용하여 얻는다. 공 freejoint의 world translation 부분은 N 단위의 공에 작용하는 실제 Weld 힘이다. raw equality force의 회전 성분을 그대로 N·m로 해석하지 않는다.

## 충돌 형상과 설정

Wrist와 15개 손가락 body에 실제 capsule geometry가 있다. 오른손/왼손 각각 16개 충돌 geometry를 사용한다. body_01 Wrist capsule radius는 오른손 32 mm, 왼손 31.2 mm다. 이는 mesh 표면이나 정밀한 손바닥·손가락 피부 충돌 모델이 아니라 기존 SMPL+H 근사 형상이다.

- C Holding: 공-손 충돌 비활성, 공-바닥·백보드·림은 기존 M4-1 설정 그대로 활성.
- C Release: 슈팅 손 Wrist/손가락과 공 사이의 충돌을 활성.
- B: 기존 부착 방식의 충돌 처리를 유지.
- `--hand-collision both`는 양손, `none`은 손 비접촉 비교 조건.
- geometry와 broad phase body mask를 함께 갱신한다.
- 팔/몸통과 공의 충돌은 기존 M4-1처럼 제외한다.

기본 offset은 초기 손가락 겹침을 피한 설정이다. 충돌 활성만으로 실제 쥐기 동작이나 마찰 기반 파지가 구현되는 것은 아니다.

## 실험과 지표

macOS Apple Silicon, 기존 Python 3.12 환경, MuJoCo 3.15.0에서 검증했다. 같은 NPZ(60 FPS, 240프레임), 같은 모션 보간, 같은 Controller/gain 규칙, 450 Hz physics, 초기 floor alignment, free Root, 2.5초 Release를 적용했다. 각 조건 6초 실행하며 원본 모션 종료 후 마지막 목표 자세를 유지한다.

- Body Joint RMSE: 손가락 제외, 손목 포함, 원본 모션 구간의 450 Hz 목표각 대비 오차(rad).
- Root Position RMSE: floor-aligned 목표 기준(m), Root는 actuator 없이 freejoint.
- 양발 접촉: 각 발 loaded normal force > 0.1 N, 모션 구간 평가.
- Foot Slip: 기존 접촉점 Jacobian의 force-weighted tangent RMS speed를 사용한 양발/시간 RMS(m/s). 접촉점 위치의 차분을 사용하지 않는다.
- 제어 토크: `qfrc_actuator`의 153개 실제 hinge torque(N·m). full/holding 구간 RMS와 관절별 시간 데이터를 저장.
- COM: human COM과 human+ball COM을 별도 기록한다. C의 Holding 동안 combined COM, Release 이후 human COM으로 `coupled_support_margin`을 평가한다. B의 공은 부하를 전달하지 않으므로 균형 margin은 human COM을 사용한다.
- Balance fraction: 기존 tilt <25°, support margin >= -0.02 m, foot force 합 >1 N, 낙상 없음 조건. 발 geometry 투영은 보수적인 지지 영역 근사다.
- 낙상: 기존 Root 높이·골반 기울기·발 이외 바닥 접촉 조건. 공의 바닥 충돌은 낙상에서 제외한다.

CSV 최소 support margin의 음수 값은 초기 접촉 형성 과도 구간도 포함한다. 낙상 0을 모든 순간 양의 안정 여유로 해석하지 않는다. 공이 릴리스된 이후 사람과 공 전체의 COM은 독립 비행하는 두 물체의 값이므로 Humanoid 균형 판정에 사용하지 않는다.

## 정량 결과

A와 B는 모든 체형에서 동일한 상태·토크를 보였다. 30 Hz로 저장한 qpos/qvel과 450 Hz 모션 구간 torque의 최대 차이는 모두 0이다. B의 운동학적 공이 하중을 전달하지 않는다는 것을 확인했다.

| 체형 | 조건 | Body RMSE(rad) | Root RMSE(mm) | 양발 접촉(%) | Foot Slip(mm/s) | Holding torque RMS(Nm) | 낙상/경고 |
|---|---|---:|---:|---:|---:|---:|---|
| body_01 | A | 0.083503 | 87.309 | 99.610 | 16.116 | 7.8272 | 0/0 |
| body_01 | B | 0.083503 | 87.309 | 99.610 | 16.116 | 7.8272 | 0/0 |
| body_01 | C | 0.083031 | 84.245 | 99.610 | 11.654 | 8.3134 | 0/0 |
| body_02 | A | 0.075403 | 89.206 | 99.219 | 8.503 | 5.3296 | 0/0 |
| body_02 | B | 0.075403 | 89.206 | 99.219 | 8.503 | 5.3296 | 0/0 |
| body_02 | C | 0.075564 | 87.442 | 99.219 | 8.491 | 5.8418 | 0/0 |
| body_03 | A | 0.094886 | 90.120 | 97.546 | 27.765 | 11.4472 | 0/0 |
| body_03 | B | 0.094886 | 90.120 | 97.546 | 27.765 | 11.4472 | 0/0 |
| body_03 | C | 0.094993 | 86.900 | 95.761 | 32.039 | 11.9103 | 0/0 |
| body_04 | A | 0.075407 | 93.348 | 99.331 | 10.367 | 6.0285 | 0/0 |
| body_04 | B | 0.075407 | 93.348 | 99.331 | 10.367 | 6.0285 | 0/0 |
| body_04 | C | 0.075582 | 91.232 | 99.331 | 10.357 | 6.5710 | 0/0 |

### 실제 공 하중에 따른 변화

| 체형 | C Holding 토크 RMS 증가율 vs A | C-A Holding 토크 차이 RMS(Nm) | 평균 Weld 수직 힘(N) | C 최대 Holding 위치 오차(mm) | C 최대 상대 회전 오차(deg) |
|---|---:|---:|---:|---:|---:|
| body_01 | 6.212% | 1.0402 | 5.9143 | 2.112 | 6.470 |
| body_02 | 9.611% | 0.8984 | 5.8945 | 1.939 | 6.801 |
| body_03 | 4.045% | 1.2416 | 5.9446 | 2.282 | 6.184 |
| body_04 | 8.999% | 1.0569 | 5.9061 | 2.140 | 7.053 |

600 g의 정적 무게는 약 5.886 N이다. 동작 중 관측한 평균 수직 Weld 힘은 약 5.89~5.94 N이며, 손에 반작용과 모멘트가 전달된다. 전체 Holding torque RMS는 4.05~9.61% 증가했다. 단일 관절마다 반드시 증가하는 것은 아니며, 자세·균형 보정으로 하중이 분배된다. body_01의 Torso_x Holding torque RMS는 A 34.95 → C 38.67 Nm, Spine_x는 38.96 → 42.27 Nm로 증가했다. 모든 관절의 비교는 체형별 `joint_torque_comparison.csv`에 있다.

공의 질량 때문인지 확인하기 위해 **별도 진단 fixture**에서 body_01 C의 공을 300 g으로 바꿨다. 원본 M4-1 모델이나 600 g 표준 결과를 수정하지 않았다. 같은 Weld 설정에서 300 g은 평균 수직 힘 2.963 N, Holding torque RMS 8.064 Nm이고 600 g은 5.914 N, 8.313 Nm다. A의 torque RMS는 7.827 Nm이다. 같은 구조에서 질량에 따라 힘과 제어 토크가 변했으므로 단순한 표시용 부착이 아님을 확인했다. 별도 단위 테스트에서도 질량을 두 배로 늘리면 Holding 수직 힘이 약 두 배가 되는 것을 검증했다.

Root RMSE는 이번 C 조건에서 모두 약 1.76~3.22 mm 감소했다. Body RMSE는 body_01에서 작게 감소하고 나머지에서 작게 증가했다. 추가 하중이 모든 지표를 일방적으로 악화시키는 것은 아니다. **body_03은 양발 접촉률이 97.546% → 95.761%, Slip이 27.765 → 32.039 mm/s로 악화**되어 균형에 더 민감했다. 기존 Controller를 공의 질량까지 고려하도록 바꾸지 않았으므로, 이러한 차이는 다음 단계의 개선 과제다.

### COM 및 균형

| 체형 | A/B balance(%) | C balance(%) | A/B 최소 human margin(mm) | C 최소 coupled margin(mm) |
|---|---:|---:|---:|---:|
| body_01 | 99.610 | 99.610 | -131.683 | -130.909 |
| body_02 | 99.219 | 99.219 | -119.220 | -118.183 |
| body_03 | 97.546 | 95.761 | -35.017 | -43.432 |
| body_04 | 99.275 | 99.275 | -138.933 | -137.761 |

### 체형별 초기 정렬과 릴리스

| 체형 | C 초기 공 world xyz(m) | C 릴리스 world xyz(m) | C 릴리스 world v(m/s) | 초기 equality 잔차 |
|---|---|---|---|---:|
| body_01 | 0.2816, 0.0660, 0.9711 | 0.4751, 0.2406, 1.9007 | 0.7676, -0.2545, 0.1000 | 1.3e-18 |
| body_02 | 0.2652, 0.0732, 0.8902 | 0.4944, 0.0439, 1.7217 | 0.6567, -0.2496, -0.0054 | 1.3e-18 |
| body_03 | 0.2980, 0.0588, 1.0561 | 0.4603, 0.3296, 2.0683 | 0.8552, -0.3679, 0.2133 | 1.3e-18 |
| body_04 | 0.2991, 0.0887, 0.9811 | 0.4862, 0.0665, 1.9404 | 0.7993, -0.2485, 0.0486 | 1.3e-18 |

모든 C 조건에서 초기 상태 쓰기 1회, Holding/Release 이후 쓰기 0회, 위치·선속도·각속도 handoff jump 0이었다. 초기 equality force는 수치 정밀도 수준(약 1e-14)으로 큰 초기 정렬 충격이 없었다. 기본 C 릴리스 이후 첫 접촉 전 이산 탄도 위치/속도 잔차는 약 1e-13 수준이다. 연속 탄도식과 물리 적분 간 timestep 오차를 구분해 평가했다.

### 손 접촉 재활성화 검증

기본 C의 네 체형에서는 릴리스 이후 손 접촉이 발생하지 않았다. 따라서 별도 body_01 실험에서 offset `[-0.07,-0.17,0]`으로 경미한 손가락 겹침을 만들었다. Weld Holding 동안 손 충돌이 없고, 해제 시 충돌이 활성화되어 4개의 physics 프레임에서 손과 접촉했다. 최대 손 접촉 normal force 37.38 N, 최대 침투 4.273 mm였으며 낙상·NaN·물리 경고·state overwrite는 없었다.

이 실험은 자연스러운 파지를 증명하지 않는다. 접촉 활성화 시 초기 겹침으로 충격이 생길 수 있다는 별도 진단이다. 첫 순간부터 접촉하므로 Release 초기 상태를 기준으로 하는 무접촉 탄도 검사는 적용하지 않았다. 완료 여부·연속성·무덮어쓰기·유한 상태·낙상/경고·침투 <10 mm·접촉력 <200 N 조건으로 검사했다. 더 깊은 겹침은 기존 운동학적 stress 실험에서 약 986 N의 큰 튕김을 만들었으므로 기본 offset으로 사용하지 않는다.

추가 C 2.1초 릴리스는 정상 자유비행 검사를 통과했고, 기본 2.5초와 속도/궤적이 달랐다. 이 조건에서는 손에 재접촉하지 않았다. 기존 B의 2.1초 릴리스에서 관찰한 자연스러운 재접촉 결과는 `M4_2_BALL_RELEASE.md`에 별도로 남겨 두었다.

![Weld Holding / 릴리스 / 자유비행](outputs/m4_physical_holding/weld_montage.png)

![네 체형 A/B/C 비교](outputs/m4_physical_holding/holding_comparison.png)

![공 비행과 관절 토크 비교](outputs/m4_physical_holding/flight_and_torque.png)

[Weld GUI GIF](outputs/m4_physical_holding/visualization/body_01/C/holding.gif) · [기존 운동학적 GUI GIF](outputs/m4_ball_release/gui/release.gif)

## 실행 명령어

팀 프로젝트 디렉터리에서 실행한다.

```bash
cd '/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim'
```

body_01 Weld GUI 및 GIF:

```bash
PYTHONDONTWRITEBYTECODE=1 \
../../../basketball-r1/.venv/bin/mjpython examples/m4_physical_holding.py \
  --body body_01 --mode C --release-time 2.5 --gui --video --check \
  --output outputs/m4_physical_holding/visualization
```

Weld를 기본으로 사용하며, `--mode B`는 운동학적, `--mode A`는 공 없는 기존 동작이다. `--body body_02`처럼 체형을 선택한다. `--hand left`로 왼손 Weld를 설정할 수 있다. 화면에는 Holding 방식과 RELEASED 상태, 시간, 노란 공 궤적을 표시한다. 6초 실행 후 창이 닫힌다.

전체 A/B/C 물리 검증과 저장 결과 비교:

```bash
PYTHONDONTWRITEBYTECODE=1 \
../../../basketball-r1/.venv/bin/python examples/m4_physical_holding.py \
  --all --mode all --check

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/m3_mpl_cache \
../../../basketball-r1/.venv/bin/python examples/m4_holding_analysis.py
```

질량 효과 분리 진단(표준 600 g 결과와 별도 저장):

```bash
PYTHONDONTWRITEBYTECODE=1 \
../../../basketball-r1/.venv/bin/python examples/m4_physical_holding.py \
  --body body_01 --mode C --test-ball-mass 0.3 --check \
  --output outputs/m4_physical_holding/mass_ablation
```

손 접촉 활성화 진단(정상 초기 offset이 아니라 경미한 겹침 조건):

```bash
PYTHONDONTWRITEBYTECODE=1 \
../../../basketball-r1/.venv/bin/python examples/m4_physical_holding.py \
  --body body_01 --mode C --offset -0.07 -0.17 0 \
  --output outputs/m4_physical_holding/contact_stress

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/m3_mpl_cache \
../../../basketball-r1/.venv/bin/python examples/m4_holding_analysis.py
```

이 초기 접촉 진단에는 `--check`의 무접촉 초기 탄도 조건이 적용되지 않는다. 저장된 접촉 진단이 존재하면 분석 스크립트가 완료·연속 상태·낙상/경고·침투·접촉력 조건을 검사한다. 정상 offset의 표준 실험은 `--check`를 사용한다.

기존 운동학적 실행과 파라미터 검증은 그대로 유지한다.

```bash
PYTHONDONTWRITEBYTECODE=1 \
../../../basketball-r1/.venv/bin/mjpython examples/m4_ball_release.py --gui --video --check
```

## 테스트, 보존 및 출력 파일

- 전체 unittest **32개 통과, skip 없음**: 기존 22개 + 운동학적 릴리스 5개 + Weld 물리 테스트 5개.
- 네 체형 × A/B/C = 12개 표준 시뮬레이션 `--check` 통과.
- 체형별 A/B qpos/qvel/torque 차이 0 확인 및 C 실제 Holding torque 차이 검증 통과.
- body_01 300 g 질량 진단, 2.1초 Release, 손 접촉 진단, 즉시 Release(0초) 실행 통과.
- C의 네 체형·양손 초기 정렬, 초기 제약 잔차, 하중 전달, 릴리스 무덮어쓰기, 충돌 mask 유지 단위 테스트 통과.
- 기존 M4-1의 6개 공 물리 검증 통과.
- 기존 합성 자유투 네 체형 회귀 테스트 통과.
- 기존 M3 Controller, Fitting NPZ, M4-1 모델, 원본 개발 환경과 SPL 데이터 변경 없음.
- macOS `mjpython` GUI 실행 완료, GIF 181프레임 저장 및 렌더링 이미지 확인.

전체 테스트 명령은 `M4_2_BALL_RELEASE.md`의 unittest 실행 블록을 사용한다. 기존 motion 테스트의 모델 경로만 실행 시 원본 로컬 모델 디렉터리로 지정하고 테스트 파일은 수정하지 않는다. Python import 과정의 기존 SyntaxWarning/FutureWarning은 MuJoCo 물리 경고와 구분했다.

추가 구현 파일:

| 파일 | 역할 |
|---|---|
| `smpl_sim/physics/physical_holding.py` | 별도 XML Weld 구성, 단 한 번 정렬, 실제 ball state 유지 Release |
| `examples/m4_physical_holding.py` | 네 체형 A/B/C 물리 실행, 토크/COM/Slip/접촉/탄도 및 GUI/GIF |
| `examples/m4_holding_analysis.py` | 상태·토크 parity, C 하중 효과, 조건 비교 CSV/그래프 |
| `tests/test_physical_holding.py` | 5개 실제 물리 검증 |
| `M4_2_PHYSICAL_HOLDING.md` | 구현·정량 비교·문제점·실행 방법 |

운동학적 관련 5개 신규 파일은 `M4_2_BALL_RELEASE.md`에 정리했다. M1~M4-1 소스 파일은 변경하지 않았다. Git Branch/Commit/Push/PR/Issue 작업은 수행하지 않았다.

결과 루트 `outputs/m4_physical_holding/`:

- `comparison.csv`, `paired_comparison.csv`: 전체 지표 및 A/C 차이.
- `body_0X/A|B|C/trajectory.csv`: 450 Hz COM, 발, Slip, 공, Weld 힘, 낙상.
- `body_0X/A|B|C/joint_torques.csv`: 관절별 추종/토크 RMS와 최대 토크.
- `body_0X/A|B|C/torques.npz`: 모션 구간 관절별 실제 토크 시계열.
- `body_0X/A|B|C/states.npz`: 30 Hz qpos/qvel/time, Fitting 파일이 아니다.
- `body_0X/A|B|C/summary.json`, `contacts.csv`: 정량 결과·발생한 공 접촉.
- `body_0X/B|C/holding.xml`: 새 환경 사본. 기존 M4-1 XML을 수정하지 않는다.
- `body_0X/joint_torque_comparison.csv`: A/C 관절별 Holding 토크 차이.
- `holding_comparison.png`, `flight_and_torque.png`, `weld_montage.png`: 그래프 및 시각화.
- `visualization/body_01/C/holding.gif`: Weld 실제 GUI 실행 GIF.
- `mass_ablation/`, `contact_stress/`, `recontact/`: 표준 실험과 분리한 진단 결과.

보존 및 테스트 로그는 `outputs/m4_ball_release/`에 저장했다. 원본 모델·데이터·`.venv`는 기존 로컬 경로를 참조하고 복사하지 않았다.

## 물리적 문제와 한계

Weld는 soft constraint이므로 완전히 고정된 그립이 아니다. 모션 중 최대 위치 오차는 약 2 mm, 상대 회전 오차는 약 6~7°다. 형상이 작은 sphere여서 회전 오차는 공 중심 오차만큼 시각적으로 드러나지 않지만 실제 측정값을 남겼다. 추후 solver 설정과 고속 손 운동의 constraint 오차를 더 검토해야 한다.

초기 공-손 겹침은 Release 시 물리적 충격을 만든다. 기본 geometry와 offset으로 충돌을 확인했지만 모든 손 pose나 체형에서 안전한 파지를 보장하지 않는다. 현재 손 형상은 capsule 근사이며, 손가락 pose는 실제 자유투 손가락 관측을 Fitting한 결과가 아니다. 양손 파지, 마찰 기반 그립, 손가락별 접촉 제어는 아직 없다.

현재 Controller는 공을 포함한 COM/관성에 맞춰 설계하지 않았다. body_03에서 발 접촉과 미끄러짐 악화가 보였으므로 추가 부하에 대한 균형 여유를 다음 단계에서 다뤄야 한다. Controller를 수정해 문제를 숨기지 않았다.

기본 실제 모션의 Release 속도는 골대에 도달하기에 부족하다. Weld로 하중을 전달해도 올바른 슈팅 릴리스가 자동으로 복원되지는 않는다. 임의의 발사 속도, 원본 NPZ 수정, PPO, Retargeting으로 결과를 보정하지 않았다.

GitHub Issue 완료 확인 항목:

- [x] 기존 운동학적 Holding 유지
- [x] Weld 기반 600 g 공 하중 전달
- [x] 체형별 초기 정렬과 무덮어쓰기 Holding
- [x] 제약 해제만으로 실제 상태를 유지하는 Release
- [x] A/B/C 및 관절별 토크·COM·Slip 비교
- [x] 네 체형 검증 및 충돌 활성화 진단
- [x] GUI/GIF/CSV/문서 저장
- [x] 기존 테스트 회귀 검증 및 원본 보존

보존 audit: 작업 이전 팀 파일 134개, SPL 원본 594개, 원본 개발 환경 snapshot 379개에서 변경을 발견하지 않았다. 상세 해시와 신규 10개 파일 목록은 `outputs/m4_ball_release/preservation_audit.json`에 저장했다.
