# M3-2A: 실제 자유투 Motion의 4개 체형 Baseline 검증

검증일: 2026-10-10. macOS Apple Silicon arm64, Python 3.12.13, MuJoCo 3.15.0. 동일한 SPL P0002/T0049 Fitting Motion을 기존 네 XML에 직접 적용한 baseline이다. 새로운 retargeting/IK/체형 최적화는 구현하지 않았다.

**네 체형 모두 전체 실행 동안 낙상과 MuJoCo 물리 경고 없이 동작했다. 그러나 몸 관절 오차·root·접촉 slip·균형은 체형에 따라 달랐다.** 바닥 정렬 root 오차는 body_04, 원본 source 기준 오차는 body_02, 발 slip과 몸 관절 오차는 body_03이 가장 크다. 무낙상이 동일한 전신 경로/발 고정의 재현을 의미하지 않는다.

## 1. 실험 조건과 재사용

- 입력: [spl_P0002_T0049.npz](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/outputs/m3_spl_fitting/spl_P0002_T0049.npz). 기존 14 body 관측점 Fitting MPJPE 17.31mm와 이번 물리 관절각 추종 오차는 별개다.
- 60FPS, 240프레임, 끝 프레임 포함 길이 239/60=3.983333초. 원본 NPZ를 변경하지 않았다.
- 공통 입력 SHA-256: `af4a970bbac834955ea6222f36a2ef3a320b520a40e9fbf17b85b0a32910200d`.
- 모델: 기존 원본 환경의 outputs/smplh_physics/body_01.xml ~ body_04.xml을 읽기 전용으로 참조. 원본 환경·SMPL+H 모델·데이터셋을 팀 저장소에 복사하지 않았다.
- 기존 M3-1B-4 `m3_spl_physics.run()`을 네 번 호출한다. 물리 step/PD/COM/측정 코드는 동일하다. 공통 구조의 StablePD + COM과 기존 체형별 gain_scale=m/80*(mean_leg_length/0.82)^2를 유지했다.
- 물리 450Hz, 전체 구간 1793스텝/1794개 450Hz 표본, 관절 상세/저장 상태는 60Hz 240개 표본. 실제 종료 시간은 3.984444초이고 60Hz 시각과 최대 1.111ms 차이가 난다.
- 초기화 때만 qpos를 설정한다. 이후 root는 freejoint이며 root actuator·외부 지지력·중간 상태 reset이 없다. root 목표는 diagnostic이고 강제로 servo하지 않는다.
- 모든 체형에서 초기 충돌 형상 최저 높이를 바닥 +5mm에 놓는 상수 world Z offset만 적용한다. 관절 pose/회전·root XY/시간별 상대 이동·FPS를 수정하지 않는다. 기존 COM 보정에 의한 유효 목표 변화는 그대로 측정한다.
- `SMPLHMotion` 및 기존 mapping: SMPL axis-angle → 회전행렬 → intrinsic XYZ hinge, root는 C*R_root, C는 (x,y,z)_SMPL→(z,x,y)_MuJoCo. root 위치/hinge 선형 보간 및 root SLERP도 기존 그대로다.
- 신장·질량·팔·다리 길이와 gain_scale이 함께 달라진다. 따라서 결과는 이 전체 실험 조건에 대한 비교이며 신장 한 요소의 인과 효과를 분리한 실험은 아니다. 단일 기준 모션·단일 결정적 실행이고 통계적 일반화/외란 견고성 시험을 수행하지 않았다.

## 2. 체형 치수와 초기 바닥 정렬

신장은 male SMPL+H 모델에 각 beta를 적용한 zero body pose mesh의 SMPL Y축 max-min이다. 팔 길이는 XML shoulder→elbow→wrist, 다리 길이는 hip→knee→ankle 관절 사이 분절 길이 합의 좌우 평균이다. 손/발 길이는 팔/다리 값에 포함하지 않는다. 체중은 실제 사람의 체중을 측정한 값이 아니라 XML body_mass의 합으로 정의한 시뮬레이션 질량이다.

| 체형 | beta 설정 (16개) | 신장 cm | 질량 kg | 팔 cm | 다리 cm | Z offset mm | gain_scale |
|---|---|---:|---:|---:|---:|---:|---:|
| body_01 | 전부 0 | 179.11 | 79.60 | 52.73 | 81.29 | -261.31 | 0.977857 |
| body_02 | beta[0]=+2, 나머지 0 | 163.85 | 61.42 | 47.83 | 73.59 | -349.53 | 0.618358 |
| body_03 | beta[0]=-2, 나머지 0 | 194.70 | 101.33 | 57.63 | 88.99 | -168.81 | 1.491647 |
| body_04 | beta[1]=+2, 나머지 0 | 181.24 | 63.76 | 54.04 | 85.23 | -223.58 | 0.861053 |

원본 source root 초기 Z=1.239146m는 동일하다. 체형마다 뼈 길이와 충돌 형상이 달라져 바닥 정렬 offset이 다르다. body_02의 큰 음수 offset은 원본 source 기준 root 오차에 직접 포함된다. source beta를 각 target에 새로 fitting하거나 target 모션을 보정하지 않았다.

## 3. 전체 관절 및 Root 평가

Body RMSE는 root와 90 finger hinge를 제외한 63 body hinge의 평균 제곱 오차를 전체 450Hz 시간 표본에서 집계한 값이다. 초기 착지 구간을 포함하며 warmup을 버리지 않았다. Root 위치 RMSE는 Euclidean xyz 오차의 RMS, 회전 RMSE는 quaternion의 부호 동치를 고려한 shortest geodesic angle의 RMS다.

| 체형 | Body RMSE rad | Root 정렬 기준 mm | Root 원본 기준 mm | Root 회전 ° | 좌 팔 RMSE rad | 우 팔 RMSE rad |
|---|---:|---:|---:|---:|---:|---:|
| body_01 | 0.08349 | 87.31 | 313.07 | 10.65 | 0.11336 | 0.13292 |
| body_02 | 0.07539 | 89.20 | 400.36 | 7.29 | 0.11356 | 0.13299 |
| body_03 | 0.09487 | 90.12 | 229.05 | 13.93 | 0.11325 | 0.13288 |
| body_04 | 0.07539 | 93.34 | 279.28 | 6.34 | 0.11315 | 0.13388 |

- **초기 바닥 정렬 기준 root 오차 최대:** body_04, 93.34mm. 실제 root와 그 체형에 맞춘 상수 Z offset reference 사이의 차이다.
- **원본 source 기준 root 오차 최대:** body_02, 400.36mm. 바닥 정렬 offset을 제거하지 않은 원본 root와의 차이다. 이를 body_02 제어기가 단독으로 가장 나쁘다는 의미로 읽으면 안 된다.
- Root 회전 오차는 body_03 13.93°가 최대이고 body_04 6.34°가 최소다. Root 위치는 모든 체형에 87~93mm 오차가 남는다.
- Body RMSE는 body_03 0.09487rad가 최대다. body_02와 body_04는 약 0.07539rad로 매우 비슷하다.

주요 관절별 RMSE (rad):

| Joint | body_01 | body_02 | body_03 | body_04 |
|---|---:|---:|---:|---:|
| L_Hip_z | 0.15504 | 0.07881 | 0.21049 | 0.07488 |
| R_Hip_z | 0.13739 | 0.05419 | 0.19283 | 0.06316 |
| L_Knee_x | 0.08312 | 0.06393 | 0.09357 | 0.06186 |
| R_Knee_x | 0.05109 | 0.05494 | 0.05220 | 0.04969 |
| L_Ankle_z | 0.09009 | 0.06826 | 0.10455 | 0.07722 |
| R_Ankle_z | 0.21057 | 0.05741 | 0.35427 | 0.08071 |
| L_Shoulder_z | 0.17809 | 0.17875 | 0.17759 | 0.17800 |
| R_Shoulder_z | 0.14034 | 0.14078 | 0.13991 | 0.14030 |
| L_Elbow_z | 0.20556 | 0.20574 | 0.20536 | 0.20446 |
| R_Elbow_z | 0.28691 | 0.28696 | 0.28688 | 0.28780 |
| R_Elbow_y | 0.10051 | 0.10049 | 0.10046 | 0.10036 |
| R_Wrist_x | 0.00204 | 0.00162 | 0.00252 | 0.00343 |

body_03의 R_Ankle_z=0.35427rad가 큰 오차다. R_Elbow_z는 네 체형 모두 약 0.287rad로 비슷하며, 좌우 팔 aggregate도 체형 간 큰 차이가 없다. 체형 차이에 따른 전체 Body RMSE 증가가 팔 정확도 악화만을 뜻하지 않는다. 모든 153 hinge의 RMSE/최대 오차는 joint_comparison.csv에 저장했다.

Wrist의 낮은 오차는 기존 body-only Fitting에서 손목 회전과 손가락이 충분히 복원되지 않은 거의 정적인 목표에 대한 결과다. 실제 손목 snap이나 손가락 release 정답을 추종한 증거는 아니다.

## 4. 균형·지지 영역·접촉

| 체형 | 균형률 % | 양발 접촉 % | 접촉 중 COM 내부 % | 최소 margin 전체 mm | 최소 margin 100ms 이후 mm | 최대 pelvis tilt ° | 낙상 | 경고 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| body_01 | 99.61 | 99.61 | 99.78 | -131.68 | 99.35 | 19.27 | 0 | 0 |
| body_02 | 99.22 | 99.22 | 99.44 | -119.22 | 91.70 | 13.69 | 0 | 0 |
| body_03 | 97.55 | 97.55 | 97.87 | -35.02 | -35.02 | 21.74 | 0 | 0 |
| body_04 | 99.28 | 99.33 | 99.44 | -138.93 | 88.93 | 14.08 | 0 | 0 |

균형 기준은 기존 그대로: 낙상 없음, pelvis tilt<25°, 한 발 이상 normal force>1N, support margin≥-20mm. 낙상은 root 높이가 초기의 65% 미만, tilt>35° 또는 힘이 실린 비발 바닥 접촉 중 하나라도 발생할 때다. fall_count는 한 rollout 중 최초 발생을 latch한 0/1이다.

지지 영역은 force>0.1N이 실린 발 box geom 전체 모서리의 XY convex hull이다. Margin은 COM XY에서 hull 경계까지의 signed distance이고 양수가 내부다. 실제 접촉점/CoP/ZMP 영역과 같지 않은 footprint proxy다. 발 force>1N이 없는 시간의 sentinel -1m는 margin 최소/내부 비율 집계에서 제외한다.

초기 상수 정렬의 5mm clearance 때문에 시작 시 짧은 공중/한 발 착지 과도가 있다. 전체 최소 margin은 이를 포함한다. 전체 지표는 그대로 두고, 착지 과도와 동작 중 문제를 구분하도록 100ms 이후 최소 margin을 보조로 제시했다. body_01/02/04는 100ms 이후 균형 판정을 벗어나지 않았고 양의 margin을 유지했다.

**body_03은 2.453333–2.573333초 사이 일부 표본에서 COM이 footprint 밖으로 나가며 오른발 접촉을 잃는다.** 100ms 이후 음수 margin/불균형 표본은 38개(450Hz 약 84.4ms 누적)다. 최소 margin -35.02mm 시각 2.573333초에는 왼발 normal force 879.39N, 오른발 0N, pelvis tilt 14.85°였다. 낙상 판정에는 도달하지 않았지만 단순 무낙상 여부보다 약한 안정성을 보여 준다.

## 5. 발 미끄러짐과 이동

Foot slip은 force>0.1N 접촉점마다 `mj_jac(contact.pos, foot_body) @ qvel`에서 접촉 법선 속도를 제거한 접선 속도다. 바닥은 정적이다. 한 시각의 여러 contact는 normal force로 RMS 가중하고, 전체 RMS는 그 발 normal force>1N 표본에 대해서만 집계한다. 바뀌는 contact.pos의 차분으로 slip을 계산하지 않는다.

좌우 통합 값은 sqrt((L_RMS²+R_RMS²)/2)인 equal-foot RMS다. 좌우 접촉 시간·힘을 모두 풀링한 별도 통계가 아니다.

| 체형 | 좌 slip RMS mm/s | 우 slip RMS mm/s | 좌우 통합 mm/s | 좌 ankle XY 이동 mm | 우 ankle XY 이동 mm | 우 toe XY 이동 mm |
|---|---:|---:|---:|---:|---:|---:|
| body_01 | 9.26 | 20.86 | 16.14 | 16.87 | 25.13 | 47.40 |
| body_02 | 7.73 | 9.25 | 8.53 | 14.87 | 8.42 | 22.23 |
| body_03 | 14.07 | 37.11 | 28.07 | 22.99 | 53.16 | 86.05 |
| body_04 | 9.14 | 11.50 | 10.39 | 17.62 | 15.71 | 43.40 |

**body_03의 slip이 최대**다. 우 발 37.11mm/s, 좌우 통합 28.07mm/s, 우 toe 최대 XY 이동 86.05mm다. body_02의 통합 slip은 8.53mm/s로 최소다. Foot body 초기 위치 대비 최대 이동은 접촉점 미끄러짐과 다른 지표이며 roll/기구학적 움직임도 포함한다. 모든 체형에 접촉 slip이 남아 완전한 발 고정은 달성하지 않았다.

## 6. 동일 Motion 직접 적용의 한계와 Retargeting 필요성

1. **동일 관절각은 동일 공간 경로를 보장하지 않는다.** 길이가 다른 팔/다리에서는 wrist/foot의 FK 위치가 달라진다. 한 번의 root Z 이동은 초기 최저 geom만 맞출 뿐 이후 foot/contact/wrist 경로를 정합하지 않는다.
2. **source root와 target 바닥 제약이 다르다.** Source beta로 복원한 root 경로를 target XML에 직접 사용하면 큰 상수 높이 차이가 생긴다. 이번 실험은 각 체형 초기 바닥 정렬로 이를 분리해 드러낸 baseline이다.
3. **PD gain scaling만으로 물리적 동일성을 만들 수 없다.** 질량/관성/분절 비율/지지 영역/COM과 scaled gains가 함께 변하며 body_03은 ankle 오차·slip·한 발 지지 구간이 증가했다. 모든 차이를 키나 체중 하나의 원인으로 단정하지 않는다.
4. **무낙상과 높은 접촉률이 root/릴리스 정확도의 보증은 아니다.** Root는 servo하지 않으므로 균형을 유지하면서도 경로와 회전이 어긋난다. 손목/손가락 모션 정답 부재도 해결되지 않았다.
5. **지원 영역은 proxy이고 실험 범위가 작다.** 단일 실제 모션·단일 결정적 실행만 검증했다. 공/골대/shot success가 없으며 외란·다른 trial·contact quality에 대한 일반화는 평가하지 않았다.

다음 단계는 source/target을 구분한 contact-aware retargeting 설계가 필요하다. 우선 body_03의 오른발 접촉 상실 구간과 reference foot 경로를 분석한 뒤 target rest 분절 길이를 고려한 발 IK/root 정합, support/COM 제약, 슈팅 손의 공간 경로 보존을 비교 실험으로 평가할 수 있다. 손목/손가락은 기존 Fitting의 별도 관측/최적화 검증이 선행되어야 한다. 이번 단계에서는 이 알고리즘을 구현하거나 NPZ를 보정하지 않았다.

## 7. 시각화

![3초 동일 시점의 네 체형 물리 상태](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/outputs/m3_body_baseline/pose_comparison_3s.png)

- [네 체형 비교 GIF](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/outputs/m3_body_baseline/four_body_comparison.gif): 2×2, 1280×960, 120프레임, 30FPS 평균 시간 간격, 4.00초 반복 재생.
- [Body/Root/Slip 비교 그래프](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/outputs/m3_body_baseline/tracking_comparison.png)
- [시간별 오차·지지 margin](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/outputs/m3_body_baseline/timeseries_comparison.png)
- [63 body hinge RMSE heatmap](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/outputs/m3_body_baseline/joint_rmse_heatmap.png)
- 0/1/2/3초의 동일 물리 시점 비교 PNG를 저장했다. 네 체형은 공통 카메라 위치·거리·투영을 사용하며 body_03 손끝이 잘리지 않도록 범위를 확인했다. snapshot은 물리 상태만 보여준다.
- 각 체형 GIF와 4체형 GIF의 녹색 선은 동일 target XML에서 FK로 계산한 reference이고 갈색 형상은 물리 결과다. 원래 source 사람/관측점 mesh와 비교한 영상은 아니다.
- 렌더링은 저장된 qpos/qvel/time을 별도 MjData에 읽어 표시하며 실험 물리를 재실행하거나 결과를 변경하지 않는다. source/XML hash가 저장 상태와 일치하는지 검사했다.

## 8. 실행 명령어

팀 저장소 작업 위치:

```bash
cd "/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim"
```

네 체형 물리 실행·치수 측정·CSV/그래프:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/m3_mpl_cache \
../../../basketball-r1/.venv/bin/python examples/m3_body_baseline.py
```

저장 상태를 읽어 비교 PNG 및 네 체형 GIF 생성:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/m3_mpl_cache \
../../../basketball-r1/.venv/bin/mjpython examples/m3_body_baseline.py --render-only
```

입력/모델/출력 위치를 바꿀 때는 `--source`, `--models`, `--smpl-models`, `--output`을 지정한다. 기본값은 기존 원본 환경의 XML/모델 및 팀 저장소의 fitted NPZ와 outputs/m3_body_baseline이다.

## 9. 회귀 및 동일 조건 검증

- 기존 단위 테스트 16/16 통과. Rotation composition, root convention/SLERP, synthetic continuity, SPL fitting FK, contact slip/quaternion 테스트를 유지했다.
- 네 체형 기존 합성 자유투 `--all --check` 통과.

| 체형 | 합성 Body RMSE rad | 합성 Task RMSE rad | 균형 % | 양발 % | 낙상 | 경고 |
|---|---:|---:|---:|---:|---:|---:|
| body_01 | 0.03243 | 0.07318 | 100.00 | 100.00 | 0 | 0 |
| body_02 | 0.03204 | 0.07355 | 100.00 | 100.00 | 0 | 0 |
| body_03 | 0.03441 | 0.07286 | 100.00 | 100.00 | 0 | 0 |
| body_04 | 0.03093 | 0.07232 | 100.00 | 100.00 | 0 | 0 |

- 실제 모션 body_01 summary는 기존 M3-1B-4 aligned summary 전체와 정확히 동일하다. 기존 동작을 유지한 것을 수치로 확인했다.
- 네 체형의 60Hz 관절 기준값 차이 최대 0rad, 저장 시각 차이 최대 0초. 원본 pose를 네 체형에 동일하게 적용한 것을 확인했다. 각 체형 1793스텝/240프레임을 완료했고 source hash가 모두 같다.
- 기존 팀 Python 124개 중 변경은 평가 코드의 body label/return summary 추가 한 파일뿐이다. 나머지 기존 Python, 원본 환경 및 원본 SPL dataset hash 변경 없음.
- 기존 Python invalid-escape SyntaxWarning 및 Torch/SMPL 모델 안내는 로그에 남아 있지만 7종 MuJoCo 물리 warning과 구분한다. 물리 warning 카운터는 네 체형 모두 0이다.

합성 회귀 실행:

```bash
PYTHONDONTWRITEBYTECODE=1 ../../../basketball-r1/.venv/bin/python \
  examples/m3_free_throw.py --all --check \
  --models ../../../basketball-r1/SMPLSim/outputs/smplh_physics \
  --output outputs/m3_body_baseline/synthetic_regression
```

기존 테스트의 XML 경로를 파일 변경 없이 프로세스 내에서 원본 모델 위치로 연결해 실행:

```bash
PYTHONDONTWRITEBYTECODE=1 ../../../basketball-r1/.venv/bin/python - <<'PYTEST'
from pathlib import Path
import sys, unittest
team = Path.cwd()
sys.path.insert(0, str(team / "tests"))
import test_smplh_motion
test_smplh_motion.ROOT = team.parents[2] / "basketball-r1/SMPLSim"
result = unittest.TextTestRunner(verbosity=2).run(
    unittest.defaultTestLoader.discover(str(team / "tests")))
raise SystemExit(not result.wasSuccessful())
PYTEST
```

## 10. 변경 파일과 결과 파일

- 신규 [examples/m3_body_baseline.py](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/examples/m3_body_baseline.py): 네 체형 실험 orchestration, 치수 측정, 통합 CSV/그래프, 저장 상태 비교 렌더.
- 수정 [examples/m3_spl_physics.py](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/examples/m3_spl_physics.py): summary body를 `getattr(args, 'body', 'body_01')`로 받도록 하고 run이 summary를 반환하도록 추가. 기본 CLI와 body_01 결과는 유지한다. PD·COM·보간·물리·평가 수식은 변경하지 않았다.
- 신규 [M3_2_BODY_BASELINE.md](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/M3_2_BODY_BASELINE.md): 본 문서.
- 출력 디렉터리: [/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/outputs/m3_body_baseline](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/outputs/m3_body_baseline). 기존 Git ignore 대상이고 Git에 추가하지 않았다.

| 파일 | 내용 |
|---|---|
| comparison_summary.csv | 체형별 모든 지표, source/정렬 root, gain, warning count, 좌우 및 통합 slip |
| body_dimensions.csv | beta·mesh 신장·XML 질량·좌우/평균 팔·다리 길이·XML SHA-256 |
| joint_comparison.csv | 네 체형 153 hinge별 RMSE 및 최대 오차 |
| body_*/summary.json, summary.csv | 공통 물리 평가 코드가 생성한 각 체형 지표 |
| body_*/timeseries_450hz.csv, timeseries_60hz.csv | root/COM/접촉/force/slip/균형/낙상 시계열 |
| body_*/joint_tracking_60hz.csv | 기준/실제/effective target/torque |
| body_*/foot_contact_points_60hz.csv | 접촉점/힘/접선 속도 |
| body_*/reference_kinematics_60hz.csv, physical_states.npz | reference 진단 및 물리 상태 재현 자료 |
| body_*/reference_vs_physics.gif, four_body_comparison.gif | 개별/4체형 비교 동작 |
| pose_comparison_0s.png ~ 3s.png | 동일 시점 4체형 자세 |
| tracking_comparison.png, timeseries_comparison.png, joint_rmse_heatmap.png | 통합 비교 그래프 |
| run.log, render.log, tests.log, synthetic_regression.log | 실행/렌더/테스트 로그 |
| synthetic_regression/ | 기존 합성 모션 회귀 결과 |
| preservation_before.json, preservation_audit.json | 코드/입력/원본 보존 및 공통 목표·이전 결과 일치 검사 |

## 11. Issue에서 직접 체크할 완료 항목

- [x] 동일 fitted NPZ/60FPS/240프레임으로 네 XML baseline 실행
- [x] 기존 PD + COM, gain_scale, 초기 바닥 정렬 및 freejoint 유지
- [x] source/정렬 root 오차 구분 및 관절·COM·접촉·slip·fall·warning 저장
- [x] 체형 신장/질량/분절 길이/offset 및 관절별 오차 비교
- [x] 동일 시점 이미지·RMSE 그래프·개별 및 4체형 GIF 생성
- [x] 16개 단위 테스트 및 4체형 합성 모션 회귀 통과
- [x] 입력·원본 데이터·기존 기능 보존 확인
- [ ] Contact-aware retargeting 구현/평가: 다음 단계
- [ ] 손목/손가락 실제 회전 복원 및 릴리스 정확도 검증: 별도 단계

Git Commit/Push/Branch 생성/Pull Request/Issue 수정은 수행하지 않았다. 체크 항목은 보고용이다. 농구공·골대·PPO도 구현하지 않았다.
