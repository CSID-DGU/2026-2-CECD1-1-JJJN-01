# M3-1B-4: 실제 자유투 Motion의 MuJoCo 물리 추종 검증

검증일: 2026-10-10. 대상은 SPL 2025 P0002/T0049의 기존 Fitting NPZ와 `body_01` 하나이다. macOS Apple Silicon arm64, Python 3.12.13, MuJoCo 3.15.0에서 실행했다.

**초기 바닥 높이를 맞춘 조건에서 약 4초 동안 관절 동작을 추종하고 낙상 없이 유지했다. 원본 root 경로의 정확한 물리 재현까지 달성한 것은 아니다.** 바닥 정렬 후 root 위치 RMSE 87.31mm, 회전 RMSE 10.65°가 남는다. 손목·손가락 회전은 기존 body-only Fitting에서 관측되지 않았으므로 실제 릴리스 동작 검증에 해당하지 않는다.

## 1. 입력과 변경 범위

- 입력: [/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/outputs/m3_spl_fitting/spl_P0002_T0049.npz](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/outputs/m3_spl_fitting/spl_P0002_T0049.npz)
- 입력 SHA-256: `af4a970bbac834955ea6222f36a2ef3a320b520a40e9fbf17b85b0a32910200d`. 실행 전후 동일하다.
- 60 FPS, 240프레임, 양 끝 프레임 포함 239/60 = 3.983333초. 입력의 Fitting MPJPE 17.31mm는 14개 body 관측점에 대한 위치 오차이고, 이번 관절각 추종 RMSE와 다른 지표다.
- 기존 원본 환경의 body_01.xml, .venv 및 모델을 참조했다. 런타임·모델·원본 SPL 데이터는 팀 저장소에 복사하지 않았다.
- 기존 M1~M3-1A 및 Fitting 코드는 변경하지 않았다. 체형별 Retargeting, 농구공·골대·PPO는 구현하지 않았다.
- Git Branch 생성·Commit·Push·Pull Request·Issue 수정은 수행하지 않았다.

새 파일:

- [/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/examples/m3_spl_physics.py](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/examples/m3_spl_physics.py): 물리 실행, 450Hz/60Hz 기록, 원본 root/바닥 정렬 비교, 기준 skeleton GUI/GIF, 그래프.
- [/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/smpl_sim/motion/physics_metrics.py](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/smpl_sim/motion/physics_metrics.py): root quaternion 거리, 힘이 실린 발 접촉점의 접선 속도.
- [/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/tests/test_physics_metrics.py](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/tests/test_physics_metrics.py): quaternion 부호 동치, 접선/법선 속도 분리, 공중 접촉 제외 테스트.
- [/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/M3_1_SPL_PHYSICS_VERIFICATION.md](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/M3_1_SPL_PHYSICS_VERIFICATION.md): 본 문서.

## 2. 실행 명령어

터미널 작업 위치:

```bash
cd "/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim"
```

화면 없이 기본 검증:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/m3_mpl_cache \
../../../basketball-r1/.venv/bin/python examples/m3_spl_physics.py \
  --root-mode aligned --output outputs/m3_spl_physics/aligned
```

원본 root 위치로 비교 실행:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/m3_mpl_cache \
../../../basketball-r1/.venv/bin/python examples/m3_spl_physics.py \
  --root-mode raw --output outputs/m3_spl_physics/raw
```

GUI에서 기준/물리 동작 비교 및 GIF 저장:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/m3_mpl_cache \
../../../basketball-r1/.venv/bin/mjpython examples/m3_spl_physics.py \
  --gui --video --root-mode aligned --output outputs/m3_spl_physics/gui
```

녹색 선은 동일 XML에서 FK로 표시한 기준 skeleton이고 갈색 충돌 형상은 실제 물리 상태다. 기준 skeleton은 viewer/render scene에만 추가하며 힘이나 충돌을 만들지 않는다. 시뮬레이션 종료 시 창이 닫힌다. `--video`를 빼면 GUI만 실행한다. GUI 실행 도중 창을 닫으면 `incomplete_reason`에 기록되며 전체 검증으로 간주하지 않는다.

기본 입력/모델 경로가 이미 위 파일들을 가리킨다. 다른 위치를 지정할 때는 `--source`, `--model`을 사용한다. 이 단계에서 실제 검증한 XML은 body_01 하나뿐이다.

## 3. Loader·좌표계·Root 처리

- 기존 `SMPLHMotion.load()` → `bind()` → `BoundMotion.sample()`을 그대로 사용한다. 추가로 저장된 root_orientation/body_pose를 별도 경로로 조립하지 않고, 기존 인터페이스의 `poses[:, :3]`와 `poses[:, 3:66]`에서 같은 회전을 읽는다.
- `root_translation`은 SMPL 라이브러리 transl이고 `root_position`은 체형 의존 pelvis 위치를 포함한 실제 root 관절 위치다. 입력은 `translation_kind=root_position`이므로 MuJoCo free joint 위치에는 root_position을 사용한다. transl을 다시 더하면 중복 변환이 된다.
- 원본 pose는 SMPL Y-up의 52×3 axis-angle(rad). 현재 코드의 C는 `(x,y,z)_SMPL → (z,x,y)_MuJoCo`이고 root 회전은 `C * R_root`이다. XML의 중력은 world -Z이다. 지역 관절 회전은 회전행렬 → intrinsic XYZ → 시간 방향 unwrap → 이름에 대응하는 x/y/z hinge로 변환한다.
- MuJoCo qpos는 root xyz 3개 + root wxyz quaternion 4개 + 153 hinge이다. nq=160, nv=159, nu=153. Root actuator는 없다.
- Root 위치와 hinge는 기존 선형 보간, root quaternion은 SLERP를 사용한다. 관절별 axis-angle를 그대로 hinge 각도로 복사하지 않는다. 지역 hinge 보간은 SO(3) geodesic 보간과 다르다는 기존 인터페이스의 한계가 있다.
- 450Hz 물리에서 매 스텝 목표를 보간한다. 60Hz 프레임 시각을 다음 물리 스텝에서 측정하므로 스텝 간격이 7/8번씩 번갈아 나타나고 최대 시간 오차는 1.111ms이다. 1793스텝, 1794개 450Hz 표본(초기 상태 포함), 240개 60Hz 표본을 기록했다. 마지막 1.111ms에는 마지막 목표를 유지한다.
- 원본 root 초기 높이는 1.239146m이고 바닥 정렬 후 0.977831m이다. 정렬은 기존 `ground()`가 충돌 형상의 최저 높이를 +5mm로 놓도록 계산한 **world Z 상수 -0.261315m**만 실행 중 reference 복사본에 적용한다. root 회전·XY·관절각·시간별 상대 root 이동과 NPZ는 그대로다.
- 원본 root의 foot collision 최저 높이는 초기 +266.315mm다. 같은 관절 회전이라도 source의 추정 beta와 body_01의 beta=0 형상이 다르다. source beta norm은 7.59397이고 자세/발 회전 추정 및 접촉 제약 부재도 영향을 줄 수 있다. 26cm 차이를 체형 하나의 원인만으로 단정하지 않는다.
- qpos는 초기화할 때만 설정한다. 이후 root를 원하는 경로로 덮어쓰거나 외력을 넣지 않고 자유 관절의 물리 결과를 측정한다. Root 위치/회전 목표는 진단용이며 servo하지 않는다. 정렬된 root와 원본 source root 양쪽 오차를 별도로 기록한다.

## 4. Controller

기존 M3-0 `BalancedPD`와 `StablePDController`를 import해 사용하며 게인, torque limit, COM 보정 공식을 변경하지 않았다.

| 대상 | 기본 kp | 기본 kd | torque limit (Nm) |
|---|---:|---:|---:|
| Hip/Knee/Ankle/Toe/Torso/Spine/Chest | 1000 | 100 | 500 |
| 나머지 몸 관절 | 300 | 30 | 150 |
| 손가락 | 5 | 0.5 | 2 |

모든 값에 기존 질량·다리길이 규칙의 body_01 gain_scale=0.977857을 곱한다. 목표 관절 속도는 기존 구현처럼 별도 feed-forward 없이 PD 감쇠를 사용한다.

COM XY 오차는 양쪽 ankle 중심에서 [0.02, 0]m 앞쪽을 목표로 한다. COM 위치/속도에서 `clip(8*error + 2*velocity, -0.6, 0.6)` 보정을 구하고 양쪽 ankle_x, ankle_z, hip_z 목표에 적용한다. 이번 실행의 보정 최대 절댓값은 X 0.339rad, Y 0.492rad로 포화 0.6rad에 도달하지 않았다.

기준 목표 자체와의 RMSE, COM 보정 후 effective target과의 RMSE를 둘 다 저장한다. COM 목표는 fitted root 이동/발 contact phase를 직접 최적화하지 않으므로 균형과 reference 전신 위치의 정확성이 서로 다를 수 있다.

## 5. 전체 구간 정량 결과

워밍업을 제외하지 않고 초기 상태부터 마지막 물리 스텝까지 450Hz 표본 전체를 집계한다. 몸 RMSE는 63개 body hinge이며 90개 finger hinge와 free root를 제외한다. 접촉률과 균형률은 초기 상태를 포함한 시간 표본 비율이다.

| 평가 항목 | 초기 바닥 정렬 | 원본 root |
|---|---:|---:|
| 몸 관절 RMSE (rad) | 0.08349 | 0.09020 |
| COM 보정 목표 RMSE (rad) | 0.08949 | 0.09344 |
| Root 위치 RMSE: 실행 reference (mm) | 87.31 | 310.17 |
| Root 위치 RMSE: 원본 source (mm) | 313.07 | 310.17 |
| Root 회전 RMSE (°) | 10.65 | 13.52 |
| 균형 표본 (%) | 99.61 | 94.37 |
| 양발 접촉 (%) | 99.61 | 94.48 |
| 발 접촉 없음 (%) | 0.17 | 5.41 |
| 최대 pelvis tilt (°) | 19.27 | 20.13 |
| 최소 접촉 support margin (mm) | -131.68 | -111.84 |
| 낙상 발생 수: 한 rollout 중 첫 발생 latch | 0 | 0 |
| MuJoCo 물리 경고 | 0 | 0 |
| 유한 상태/전체 실행 완료 | 예 | 예 |

바닥 정렬 실행의 root 축별 위치 RMSE는 X 63.79mm, Y 29.37mm, Z 51.87mm이고 최대 169.92mm가 2.548889초에 발생했다. Root 회전 오차는 wxyz quaternion의 부호 동치를 처리한 shortest geodesic angle이다. 원본 root를 기준으로 평가하면 바닥 정렬 실행의 오차는 313.07mm이므로 87.31mm를 원본 사람 위치 재현 정확도로 해석하면 안 된다.

균형 판정은 기존과 동일한 proxy: 낙상 없음, pelvis tilt <25°, 적어도 한 발 normal force >1N, projected support margin ≥-20mm. 낙상은 root 높이 <초기값의 65%, tilt >35°, 또는 힘 >0.1N인 비발 부위 바닥 접촉이다. fall_count는 최초 발생을 latch해 rollout당 0/1로 세며 반복 낙상 실험 수를 의미하지 않는다.

정렬 실행에서 첫 발 접촉은 6.667ms, 양발 접촉은 15.556ms에 시작했다. 최저 support margin -131.68mm는 6.667ms의 초기 한 발 착지 표본이며, 100ms 이후 최저값은 +99.35mm다. 접촉이 있는 표본의 COM이 support 안에 있는 비율은 99.78%다. 원본 root 실행은 약 215.56ms 동안 공중에 있다가 첫 접촉, 220ms에 양발 접촉이 생겼다.

지지 영역은 힘이 실린 발 box 충돌 형상의 전체 모서리를 바닥에 투영한 convex hull이다. 실제 contact point의 좁은 영역보다 넓은 보수적 footprint이며 CoP/ZMP·동적 안정성 보증이 아니다. 접촉이 없는 표본의 margin=-1은 missing sentinel이라 최소 support 집계에서 제외한다.

## 6. 팔 관절별 추종 오차

좌/우 어깨·팔꿈치·손목 9개 축씩의 RMSE: 좌 0.11336rad, 우 0.13292rad. 슈팅 손을 별도 정답 label로 확정하지 않고 양쪽 결과를 제공한다.

| 관절 | 좌 x RMSE | 좌 y RMSE | 좌 z RMSE | 우 x RMSE | 우 y RMSE | 우 z RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Shoulder | 0.09696 | 0.08664 | 0.17809 | 0.08533 | 0.08516 | 0.14034 |
| Elbow | 0.13968 | 0.07235 | 0.20556 | 0.17968 | 0.10051 | 0.28691 |
| Wrist | 0.00238 | 0.00393 | 0.00342 | 0.00204 | 0.00629 | 0.00561 |

단위는 rad. 가장 큰 몸 hinge 오차는 R_Elbow_z=0.28691rad(약 16.44°), 이어 R_Ankle_z=0.21057rad, L_Elbow_z=0.20556rad이다. 전체 평균이 작아도 모든 팔 축의 정확도가 같은 것은 아니다. Wrist의 작은 오차는 손목 위치에서 손목 회전이 복원되지 않아 기존 body-only Fitting의 거의 정적인 reference를 따라간 결과다. 실제 손목 snap·손가락 release 복원 성공의 근거가 아니다.

153개 모든 hinge의 RMSE/최대 오차는 joint_summary.csv, reference/실제/effective 목표와 torque는 joint_tracking_60hz.csv에 저장했다.

## 7. 양발 접촉·미끄러짐·이동

발 접촉은 L/R Ankle 또는 Toe와 floor 사이의 force >1N으로 정의한다. 미끄러짐은 force >0.1N contact마다 발 body의 contact.pos에서 `mj_jac @ qvel`을 계산한 뒤 법선 성분을 제거한다. 바닥은 정적이다. contact.pos는 매 프레임 바뀌는 충돌점이므로 그 좌표를 차분해서 미끄러짐으로 계산하지 않는다.

한 시각의 여러 점은 normal force로 가중한 접선 속도 RMS로 집계한다. 전체 slip RMS는 해당 발 force >1N인 표본들의 RMS다. 적분 slip path는 매 물리 스텝 force 가중 평균 접선 속도를 적분한 값이며 동일한 재료 점이 이동한 거리 또는 순변위와 같지 않다.

| 항목 | 정렬 좌 | 정렬 우 | 원본 좌 | 원본 우 |
|---|---:|---:|---:|---:|
| 발 접촉 (%) | 99.83 | 99.61 | 94.59 | 94.48 |
| 접촉 slip RMS (mm/s) | 9.26 | 20.86 | 13.70 | 19.87 |
| 최대 점 slip (mm/s) | 190.61 | 265.22 | 238.78 | 189.73 |
| 접촉 중 slip >10mm/s 비율 (%) | 8.21 | 27.59 | 14.97 | 56.76 |
| 적분 slip path (mm) | 20.56 | 40.79 | 25.49 | 47.75 |

초기 발 body 위치 대비 최대 이동:

| Body | 정렬 XY (mm) | 정렬 3D (mm) | 원본 XY (mm) | 원본 3D (mm) |
|---|---:|---:|---:|---:|
| L_Ankle | 16.87 | 25.95 | 46.43 | 292.36 |
| L_Toe | 18.43 | 25.33 | 40.19 | 281.68 |
| R_Ankle | 25.13 | 26.28 | 42.63 | 284.45 |
| R_Toe | 47.40 | 49.10 | 54.62 | 284.10 |

원본 root 조건의 약 28~29cm 3D 발 이동에는 초기 공중 상태에서의 수직 낙하가 포함된다. 이를 전부 바닥 미끄러짐이라고 해석하지 않는다. 정렬 조건도 오른쪽 toe의 최대 XY 이동 47.40mm와 접촉 slip RMS 20.86mm/s가 있으므로 완전한 발 고정은 달성하지 않았다.

## 8. 실패·한계와 원인

1. **원본 root 기준 발 공중 상태:** 초기 최저 collision 높이 +266.315mm 때문에 원본 root 실행은 착지 과도를 만든다. 같은 source 회전을 다른 rest 형상에 적용한 상태이며 체형 최적화나 발 IK는 이번 범위에서 수행하지 않았다.
2. **바닥 정렬 후에도 reference 발 contact 불일치:** kinematic reference의 가장 낮은 발 box 높이가 전체 구간 -20.20~+79.87mm다. 상수 높이 정렬만으로 모든 프레임의 발이 바닥에 붙지는 않는다. 물리 상태는 접촉 제약과 COM 보정으로 다른 root/관절을 택한다. 이를 fitting 결과의 수정으로 숨기지 않았다.
3. **Root 경로 오차:** free root에 위치/회전 actuator가 없고 COM 제어는 stance 안정성만 보조한다. reference root를 강제로 덮어쓰지 않았으므로 안정적 서기와 정확한 root trajectory 추종이 다르다. Root tracking error를 별도로 노출했다.
4. **팔 일부 축의 큰 오차:** R_Elbow_z 등의 오차가 전체 평균보다 크다. 현재 PD 게인/질량·관성/시간별 reference 속도와 균형 보정의 영향을 받는 것으로 추정되지만 원인 분리 실험을 수행하지 않았으므로 단일 원인으로 확정하지 않는다.
5. **손목/손가락 정답 부족:** XML의 손가락 물리 관절은 존재하지만 기존 Fitting이 손가락을 최적화하지 않았다. 실제 자유투 손가락/손목 각속도나 공 release는 이번 검증에 포함되지 않는다.
6. **MP4 환경 실패:** GUI+MP4 첫 시도에서 Homebrew FFmpeg가 libx265.216.dylib를 찾지 못했고 인코더 pipe가 종료됐다. 이 시도는 검증 완료로 세지 않았다. 시스템 설치를 변경하지 않고 PIL GIF 저장으로 바꾼 뒤 GUI 전체 실행/저장을 재검증했다. mjpython wrapper가 내부 예외에도 종료 코드 0을 반환할 수 있어 로그와 summary 생성도 확인했다.
7. **경고 구분:** 7종 MuJoCo warning 카운터는 전부 0이다. 기존 소스의 Python invalid-escape SyntaxWarning, Torch JIT FutureWarning, SMPL+H 16 shape-coefficient 안내는 런타임 로그에 남아 있으며 MuJoCo 물리 경고와 구분했다.

## 9. 시각화 결과

![기준 녹색 skeleton과 실제 물리 형상 비교](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/outputs/m3_spl_physics/gui/comparison_preview.png)

- [반복 재생 비교 GIF: 640×480, 120프레임, 4.00초](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/outputs/m3_spl_physics/gui/reference_vs_physics.gif)
- [추종·root·COM·접촉·slip 그래프](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/outputs/m3_spl_physics/aligned/physics_diagnostics.png)
- GUI launch/sync와 전체 실행 완료를 확인했고 저장된 render 이미지를 검사했다. foreground GUI 화면 자체를 별도 캡처해 확인한 것은 아니다.
- GIF 녹색 reference는 body_01 형상 위의 같은 fitted angular/root reference다. 원래 SPL 관측점/source beta 사람 skeleton과의 직접 비교 영상은 아니다.
- GUI/render를 켠 실행의 summary 전체가 headless aligned 실행과 동일해 시각화가 물리 상태에 영향을 주지 않았음을 확인했다.

## 10. 회귀 테스트

기존 합성 자유투 실행:

```bash
PYTHONDONTWRITEBYTECODE=1 ../../../basketball-r1/.venv/bin/python \
  examples/m3_free_throw.py --body body_01 --check \
  --models ../../../basketball-r1/SMPLSim/outputs/smplh_physics \
  --output outputs/m3_spl_physics/synthetic_regression
```

- 3.000초, body RMSE 0.03243177rad, task RMSE 0.07317579rad.
- 균형/양발 접촉률 100%, 낙상 0, MuJoCo 경고 0, 최대 ankle XY 이동 2.209mm. 기존 M3-1A body_01 결과와 동일하고 `--check` 통과했다.
- 단위 테스트 16/16 통과: 기존 motion 5개, 기존 synthetic 3개, 기존 SPL fitting 5개, 신규 physics metric 3개. 기존 mapping 테스트의 4개 XML 참조는 FK 회귀 확인이며 실제 자유투 물리 실험은 body_01만 수행했다.

팀 저장소에 XML을 복사하지 않기 위해 기존 테스트 모듈의 ROOT만 테스트 프로세스 내에서 원본 환경으로 연결했다. 파일 자체는 변경하지 않았다. 재현 명령:

```bash
PYTHONDONTWRITEBYTECODE=1 ../../../basketball-r1/.venv/bin/python - <<'PYTEST'
from pathlib import Path
import sys, unittest
team = Path.cwd()
sys.path.insert(0, str(team / "tests"))
import test_smplh_motion
test_smplh_motion.ROOT = team.parents[2] / "basketball-r1/SMPLSim"
suite = unittest.defaultTestLoader.discover(str(team / "tests"))
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
PYTEST
```

## 11. 결과 파일과 보존 검증

출력 디렉터리: [/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/outputs/m3_spl_physics](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/outputs/m3_spl_physics)

| 파일 | 내용 |
|---|---|
| comparison_summary.csv | aligned/raw 집계 비교 |
| aligned/, raw/, gui/summary.json 및 summary.csv | 전체 지표, root 정렬, 시간, fall, MuJoCo warnings(JSON), source hash |
| timeseries_450hz.csv / timeseries_60hz.csv | 몸 오차, root XY/Z 및 orientation, COM XYZ/margin, 접촉/force, slip, 발 이동, 낙상 조건 |
| joint_tracking_60hz.csv / joint_summary.csv | 153 hinge별 target/actual/effective/torque 및 전체 RMSE |
| foot_contact_points_60hz.csv | 힘이 실린 contact point 좌표/법선 힘/접선 속도 |
| reference_kinematics_60hz.csv | reference root/COM/발 위치와 가장 낮은 foot collision 높이 |
| physical_states.npz | 60Hz 물리 qpos/qvel/time, source/model SHA-256, 정렬 offset; fitting motion 파일과 별도 |
| physics_diagnostics.png | 전체 시간 변화 그래프 |
| gui/reference_vs_physics.gif, comparison_preview.png | 기준/물리 비교 동작 및 정지 이미지 |
| tests.log, */run.log | 단위 테스트와 실행 로그 |
| synthetic_regression/ | 기존 합성 모션 회귀 CSV·report·log |
| preservation_audit.json | 기존 파일 hash 및 GUI/headless 일치 확인 |

보존 hash 감사: 기존 팀 Python 116개, 원본 환경 392개(.DS_Store 외부 metadata 제외), SPL 데이터셋 594개 모두 내용 변화 없음. 입력 Fitting NPZ hash도 동일하다. outputs는 기존 Git ignore 대상이며 출력 데이터는 Git에 추가하지 않았다. `git diff --stat`는 비어 있고 이번 단계 신규 소스/문서만 untracked 상태로 남겼다.

## 12. 다음 개선 작업과 Issue 체크 항목

권장 순서는 (1) source beta/rest skeleton과 body_01 기준의 발 위치 차이 및 contact phase를 시각적으로 진단, (2) source motion의 발/바닥 제약과 root 경로 정합성 평가, (3) root/팔 error budget을 분리한 제어 실험, (4) 손목·손가락 관측을 포함한 별도 Fitting 검증이다. 체형별 retargeting을 진행할 때만 target 체형의 foot IK/contact 보정과 root 위치 정합을 설계한다. 기존 결과를 덮어쓰지 않고 비교 실험으로 진행해야 한다.

직접 Issue에서 체크할 수 있는 완료 항목:

- [x] 기존 NPZ Loader와 SMPL+H → hinge 변환/60FPS 보간 연동
- [x] body_01에서 기존 PD + COM 물리 추종 실행
- [x] 원본 root / 초기 바닥 정렬 조건의 분리 비교
- [x] 관절별/root/COM/접촉/접촉 slip/발 이동/fall/warning CSV 기록
- [x] GUI reference skeleton 비교 및 반복 재생 GIF 저장
- [x] 기존 합성 자유투 회귀 및 16개 단위 테스트 통과
- [x] 원본 NPZ·코드·환경·데이터 보존 확인
- [ ] 원본 source root 경로를 높은 정확도로 물리 재현
- [ ] 완전한 발 고정 및 실제 손목·손가락 release 검증
- [ ] source/target 체형 분리 retargeting: 이후 단계

위 체크 항목은 보고용이며 GitHub Issue를 직접 변경하지 않았다.

## 13. 한 GUI 창에서 반복 미리보기

추가 파일 `examples/m3_spl_loop.py`는 기존 PD + COM을 사용해 실제 물리 동작을 하나의 GUI 창에서 반복한다. NPZ와 기존 제어·평가 코드는 변경하지 않는다. 매 회차 시작 때 qpos/qvel/시간을 초기 상태로 리셋하므로 마지막 자세에서 첫 자세로 전환된다. 반복 경계를 부드럽게 연결한 새 모션은 아니다. 이 미리보기는 CSV를 저장하지 않으며 정량 평가는 위 검증 명령을 사용한다.

앞의 팀 SMPLSim 작업 위치에서 실행:

```bash
PYTHONDONTWRITEBYTECODE=1 \
../../../basketball-r1/.venv/bin/mjpython examples/m3_spl_loop.py --repeats 0
```

`--repeats 0`은 창을 닫을 때까지 계속 실행한다. `--repeats 5`는 같은 창에서 5회 실행한 뒤 종료한다. 녹색 기준 skeleton을 숨기려면 `--no-reference`를 추가한다. 창 닫기 또는 터미널 Ctrl+C로 중단할 수 있다.
