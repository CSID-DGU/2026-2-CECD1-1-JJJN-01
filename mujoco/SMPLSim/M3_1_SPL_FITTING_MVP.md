# M3-1B-3 SPL 자유투 SMPL+H Fitting MVP

2026-10-10, macOS Apple Silicon arm64 / Python 3.12 / MuJoCo 3.15.0. 팀 `mujoco/SMPLSim`, 기존 `dev/mincheol`에서 작업했다. 단일 실제 시도 **2025-12-18 / P0002 / T0049**만 fitting했다. 60FPS, 240프레임, 3.983333초를 유지했다. 손가락 fitting은 제외한 body MVP다.

## 결과와 적용 범위

전처리 → named observation mapping → differentiable source-shape/body/root fitting → schema-v1 NPZ export → 기존 MuJoCo motion loader/관절 변환까지 구현하고 실행했다. 입력 69개 key 중 **14개 body 관측점**의 위치를 평가했다. 전체 69개 또는 모든 52개 관절의 정확도를 측정했다는 뜻은 아니다.

**MPJPE 249.858 → 17.310mm**, 좌/우 어깨·팔꿈치·손목 평균 12.089 / 17.878mm, hip 중점 root proxy 오차 11.926mm. 이 결과는 같은 시도에 fitting한 in-sample 관측 잔차다. 일반화 성능이나 진짜 3D ground truth 정확도가 아니며, baseline은 초기 hip 기반 root 방향 + 중립 body pose + zero beta다.

NaN/Inf 없는 출력, 기존 XML 관절 범위/gimbal 변환 통과, full SMPL+H forward와 FK 일치를 확인했다. MuJoCo source-shape XML을 새로 만들거나 실제 물리 rollout/GUI 실행을 하지 않았다. Motion Tracking RMSE·Balance·발 contact·낙상은 M3-1B-4에서 검증해야 한다. 기존 body_01 XML은 **NPZ 인터페이스/관절 범위 검사만** 수행했으며 source fitting 결과를 다른 체형에 retargeting하지 않았다.

## 구현 파일

| 신규 파일 | 역할 |
|---|---|
| smpl_sim/motion/spl_data.py | 읽기 전용 JSON loader, feet→meter, 좌표/heading 정규화, finite/outlier 마스크, 짧은 gap 보간, 관절 mapping |
| smpl_sim/motion/spl_fitting.py | 기존 SMPL+H 관절 regressor/shapedirs와 smplx FK 재사용, root/body/고정 source beta 공동 최적화 |
| examples/m3_spl_fit.py | 단일 시도 제한 CLI, 모델 참조, 실행·평가·NPZ/CSV/그림/메타데이터 저장 |
| tests/test_spl_fitting.py | gap/outlier/좌표/좌우/실제 model FK 및 gradient/root 관측 검증 5개 |
| M3_1_SPL_FITTING_MVP.md | 이 실행의 설계·정량 결과·명령어·한계 |

기존 M1~M3-1A 코드와 테스트는 수정하지 않았다. `.venv` 및 SMPL+H pkl은 기존 `basketball-r1`에서 참조했고 팀 저장소에 복사하지 않았다. 원본 JSON도 수정/복사하지 않았다. `outputs/*`는 기존 Git ignore 대상이며 NPZ/CSV/그림은 로컬 파생 산출물이다. 이전 단계 문서 2개의 untracked 상태는 이번 변경과 구별한다. branch/commit/push/PR/Issue 수정은 없다.

## 전처리와 좌표계 검증

MuJoCo가 일반적으로 Z-up을 강제한다는 가정으로 처리하지 않았다. 실제 XML을 로드해 floor quaternion의 법선이 `[0,0,1]`, gravity가 `[0,0,-9.81]`임을 검사했다. 현재 SMPL+H 모델/모션의 SMPL_TO_WORLD를 사용해 fitting 좌표와 XML world 좌표를 구별한다.

- SPL 원본: right-handed, Z-up, feet. 인체 xyz에 0.3048을 곱해 meter로 변환한다. 공은 fitting에 사용하지 않는다.
- 최초 유효 hip 중점의 XY를 고정 원점으로 빼며, court Z는 유지한다. 매 프레임 root를 0으로 만들지 않는다.
- 최초 30프레임의 좌우 hip vector 중앙값을 이용해 고정 heading을 정렬한다. 정규화 후 LEFT_HIP−RIGHT_HIP의 Y가 양수이며 rotation det=+1임을 검사했다.
- 정규화 world는 여전히 Z-up이다. **fitting 단계에만 C⁻¹을 적용해 SMPL Y-up**으로 바꾼다. 기존 loader는 C를 통해 실제 root를 현재 XML world로 돌려놓는다.
- 원본 nominal `frame/fps`와 정수 ms timestamp 차이가 1.01ms 이내인지 검사한다. 원본 ms timestamp는 loader가 보존하며 fitting은 검증된 등간격 60FPS를 사용한다.
- coordinate inverse 오차는 1.11e-16m, SMPL↔world round trip 오차는 0m이다. 변환의 invertibility/chirality 검증이며, forward 방향의 절대 의미를 센서 calibration으로 증명한 것은 아니다.

속도 10m/s 초과 인접 point는 양 끝을 후보 이상치로 마스크한다. 좌우 상완/전완/대퇴/하퇴의 길이가 median에서 max(3cm, 6×1.4826×MAD) 넘게 벗어난 프레임도 마스크한다. 이는 conservative heuristic이며 실제 빠른 동작과 추적 이상치를 완벽히 구별하지 못한다.

양쪽 finite endpoint가 있는 **3프레임 이하(60FPS에서 50ms 이하)** gap만 선형 보간하고 가중치 0.25를 준다. 시작/끝 및 긴 gap을 임의 extrapolate하지 않는다. MID_HIP/NECK가 missing이고 두 hip/shoulder가 유효하면 평균으로 대체하고 가중치 0.5를 준다. 남은 결측 point는 weight=0이며 NaN을 loss에 직접 넣지 않는다. 긴 pelvis gap이 남으면 fitting을 중단해 clip 선정을 요구한다.

이번 입력에서는 모든 69개 key의 원본 유효율 **100%**, 이상치 후보 **0**, 보간 **0**이었다. 따라서 보간 기능은 실제 입력에서 작동할 필요가 없었으며, synthetic gap/spike test로 별도 검증했다. 가중치는 센서에서 제공한 confidence가 아닌 **추정 품질 가중치**다.

## Mapping 및 root 관측의 의미

body fitting 항목은 MID_HIP, NECK, 좌우 Hip/Knee/Ankle/Shoulder/Elbow/Wrist다. SPL shoulder는 SMPL L/R_Shoulder와 대응하며 Thorax와 혼동하지 않는다. Torso/Spine/Chest/Thorax의 직접 관측은 없어 body pose 변수로 최적화하되 중립/시간 prior로 제한한다. 얼굴·발끝·heel·fingertip 등 나머지 landmark는 이번 위치 loss에 사용하지 않는다. mapping.csv는 69개 source key 전부를 body candidate / finger candidate / unfitted landmark로 분류한다.

SPL MID_HIP는 hip 중심 관측이므로 **fitting loss에서 mean(SMPL L_Hip, R_Hip)**와 비교한다. SMPL regressed Pelvis의 위치 자체와 동일시하지 않는다. NPZ root_position은 관측 proxy가 아니라 **실제 SMPL Pelvis joint 위치**로 내보낸다. NECK는 SMPL Neck의 근사 관측이며 marker/regressor 차이는 남는다.

따라서 11.926mm의 Root Position Error는 **hip-midpoint proxy**의 관측 오차다. SPL에는 SMPL Pelvis의 진짜 중심 위치 라벨이 없어 실제 pelvis ground truth 오차를 직접 측정하지 못한다. root_translation 필드는 SMPL library transl이고 root_position과 다르다.

손은 기본 실행에서 local pose=0이며 모두 flat_hand_mean=True인 source 모델에서 FK를 계산한다. `--include-fingers`는 상세 30개 finger joint-center mapping을 사용하는 실험 옵션으로 분리했다. 이번에는 실행하거나 검증하지 않았으며 fingertip mesh loss는 없다. hand mean/rest와 현재 MuJoCo XML의 손 기준은 아직 정합 검증되지 않았다. 이 옵션의 결과를 물리 손가락 재현이 확인된 모션이라고 취급하지 않는다.

## 최적화와 Loss Function

기존 SMPLH_Parser를 male / num_betas=16 / use_pca=False / flat_hand_mean=True / create_transl=False로 생성했다. 참가자의 성별은 데이터에 없어 남성으로 추정했다는 뜻이 아니라 기존 모델을 사용한 것이다.

JointFK는 기존 `J_regressor @ (v_template + shapedirs·beta)` 관절과 smplx `batch_rodrigues`/`batch_rigid_transform`을 재사용한다. 매 iteration에 full 6,890-vertex mesh skinning을 수행하지 않고 같은 kinematic-joint path를 계산한다. 최종 5개 sample frame 및 무작위 nonzero pose/beta 회귀 테스트에서 full parser forward의 첫 52 joints와 비교했다. 최종 최대 좌표 성분 차이는 **7.15e-7m**, 허용치는 2e-6m다. pose/beta/transl의 finite nonzero gradient도 확인했다.

변수는 각 frame의 global root orientation 3, non-root body pose 63, library translation 3, 시퀀스 전체에서 공유하는 beta 16이다. 기본 hand 90성분은 고정 0이다. source beta는 box [-3,3] 및 L2 prior를 적용했으며 target beta는 최적화하지 않았다. Root rotation은 hip lateral + hip→neck up vector를 정규직교화하여 초기화한다. translation은 hip 관측과 초기 FK hip 중점을 맞춰 초기화한다.

CPU 1 thread, seed=0, Adam **800 iteration**. learning rate 0.025→0.0005 cosine 감소, gradient norm clip=1. 각 관측 좌표의 Huber(delta=0.05m)를 xyz 합산 후 quality weight로 평균한다. root/neck 근사 중심 관측에는 추가로 0.5를 곱한다. 목적함수는 다음과 같다.

```text
L = weighted joint Huber
  + 1e-6 × mean((Δ rotation_matrix × FPS)^2)
  + 1e-9 × mean((Δ² rotation_matrix × FPS²)^2)
  + 1e-6 × mean((Δ² translation × FPS²)^2)
  + 2e-5 × mean(body rotvec^2)
  + 1e-5 × mean(source beta^2)
  + 1e-4 × dominant knee/elbow flexion proxy penalty
```

시간 prior는 rotvec 값의 ±π 분기를 차분하지 않고 rotation matrix 차분을 사용한다. 이는 chordal rotation smoothness이며 learned motion prior나 정확한 SO(3) 가속도 모델이 아니다. body rotvec L2는 **중립 pose prior**이며 VPoser/학습형 prior를 사용하지 않았다. knee/elbow dominant rotvec 성분에 soft sign/상한 penalty를 적용하지만 Euler angle에 대한 정확한 생리학적 제약은 아니다. XML 범위 검사와 함께 쓰는 MVP 수준의 방어 항이다.

beta가 시간에 따라 고정이므로 bone lengths는 FK가 결정한다. 별도 bone loss는 추가하지 않았다. shape beta 3개가 ±3 경계에 도달했으므로 source 체형을 정확히 식별했다고 주장하지 않는다. sparse 위치와 모델/marker 정의 차이를 흡수한 effective shape이며 이후 shape 안정성 평가가 필요하다.

## 정량 평가

MPJPE는 유효한 **14개 body 관측점×240프레임**의 유클리드 거리 평균(mm), metric은 학습 weight 없이 평균한다. 이번 입력에서는 raw finite error와 전처리 관측 error가 같다.

| SPL 관측 → 모델 후보 | 평균 mm | P95 mm | 최대 mm |
|---|---:|---:|---:|
| MID_HIP → hip midpoint proxy | 11.926 | 22.889 | 23.826 |
| NECK → Neck | 17.780 | 36.682 | 41.086 |
| LEFT_HIP → L_Hip | 34.246 | 36.776 | 37.191 |
| LEFT_KNEE → L_Knee | 11.762 | 20.095 | 21.379 |
| LEFT_ANKLE → L_Ankle | 9.668 | 15.210 | 16.123 |
| LEFT_SHOULDER → L_Shoulder | 9.238 | 17.319 | 20.186 |
| LEFT_ELBOW → L_Elbow | 15.736 | 25.437 | 30.571 |
| LEFT_WRIST → L_Wrist | 11.292 | 22.632 | 23.726 |
| RIGHT_HIP → R_Hip | 34.832 | 37.484 | 38.421 |
| RIGHT_KNEE → R_Knee | 16.391 | 27.543 | 29.886 |
| RIGHT_ANKLE → R_Ankle | 15.836 | 27.277 | 30.276 |
| RIGHT_SHOULDER → R_Shoulder | 14.638 | 27.584 | 36.542 |
| RIGHT_ELBOW → R_Elbow | 16.869 | 37.338 | 68.752 |
| RIGHT_WRIST → R_Wrist | 22.128 | 33.670 | 42.018 |

| 연속성/검사 | 결과 |
|---|---:|
| 최대 root frame step | 7.041mm |
| 최대 root 차분 속도 | 0.422m/s |
| root 가속도 성분 RMS | 0.406m/s² |
| 최대 local/global 관절 상대 회전 frame step | 0.191885rad |
| 최대 관절 차분 회전속도 | 11.513rad/s |
| rotation-matrix 2차 차분 성분 RMS | 5.480s⁻² |
| 최대 rotvec norm | 1.758rad |
| NaN/Inf 출력 | 없음 |
| 기존 XML joint range 위반 / gimbal 변환 실패 | 0 / 0 |

관절 frame step은 R_tᵀR_t+1의 geodesic angle이고 velocity는 이에 60FPS를 곱한 값이다. root acceleration RMS는 xyz 성분을 함께 평균한 값이다. rotation acceleration metric은 matrix 성분 2차 차분이므로 rad/s²가 아니다. 전환 peak에서 큰 각속도가 나타날 수 있으며 finite/range pass가 동작 자연스러움의 증명은 아니다. 팔별 평균은 LEFT 12.089mm, RIGHT 17.878mm다. shooting hand 라벨이 없어 오른팔을 슈팅 팔로 단정하지 않고 양팔을 보고했다.

## 시각화

검정은 SPL 관측점/선, 위의 주황은 초기 pose, 아래의 파랑은 최종 복원이다. 시작·2초·오른손 손목 높이 최대인 2.95초를 비교한다. pelvis 선은 같은 MID_HIP 관측 proxy로 비교한다. 이 그림은 fitted skeleton이며 MuJoCo 물리 결과가 아니다. 이미지 파일을 직접 열어 관절/축/패널 표시를 확인했다.

![Fitting 전후 skeleton](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/outputs/m3_spl_fitting/skeleton_before_after.png)

![양팔 관절별 위치 오차](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/outputs/m3_spl_fitting/arm_position_errors.png)

## NPZ와 저장 결과

`spl_P0002_T0049.npz`의 기존 loader 필드:

```text
schema_version = 1
poses = (240,156), radians axis-angle in SMPLH_BONE_ORDER_NAMES order
root_position = (240,3), actual pelvis position, meters, SMPL Y-up
fps = 60
coordinate_system = smpl
translation_kind = root_position
```

추가 설명 필드: root_translation(240,3)=library transl, root_orientation(240,3)=poses[:,:3], body_pose(240,63)=poses[:,3:66], betas_source(16). 기존 SMPLHMotion.load는 추가 필드를 무시하고 정상 로드한다. SMPLHMapping.bind는 기존 body_01 XML의 240프레임을 변환해 주소·XYZ 분해·범위·root quaternion 형식을 확인했다. 이것은 서로 다른 source/target body의 공간 관절 정합 검증이 아니다.

`outputs/m3_spl_fitting/` 산출물:

- spl_P0002_T0049.npz: source fitted motion
- summary.csv / summary.json: MPJPE·양팔·root proxy·연속성·FK/좌표/loader 검증
- joint_errors.csv: 14관측점의 평균/P95/최대/초기/원본 finite 위치 잔차
- frame_errors.csv: 240프레임의 MPJPE·root step·rotation step
- validity.csv: 69개 key의 원본 유효율/outlier/보간/최종 유효율
- mapping.csv: 69개 key 분류 및 대응 후보
- optimization.csv: 50 iteration 간격 loss history와 마지막 결과
- fitting_diagnostics.npz: meter 변환/정규화된 관측, initial/fitted observation 및 품질 가중치
- skeleton_before_after.png / arm_position_errors.png: 시각화
- metadata.json: source SHA-256/단위/좌표/원점/heading/beta/solver/loss/hand 기준/출처·사용 조건
- tests.log / run.log: 실제 테스트 및 fitting 실행 로그

파생 motion은 SPL CC BY-NC-SA 4.0 및 저장소 사용 조건을 따른다. metadata에 MLSE Sport Performance Lab 출처, 원본 경로/hash, 가공 사실과 라이선스 링크를 기록했다. 기존 분석 문서의 추가 사용 조건도 유지한다. SMPL+H 모델 파일을 파생 motion과 함께 재배포하지 않는다.

## 실행 명령

팀 SMPLSim에서 실행한다. 기존 로컬 런타임과 모델/XML을 참조하므로 새 설치나 데이터 복사가 필요 없다.

```bash
cd "/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/m3_mpl_cache
LOCAL_M3_ENV="../../../basketball-r1"

"$LOCAL_M3_ENV/.venv/bin/python" examples/m3_spl_fit.py \
  --source ../../../SPL-Open-Data/basketball/freethrow/data/2025-12-18/P0002/BB_FT_P0002_T0049.json \
  --model-dir "$LOCAL_M3_ENV/SMPLSim/assets/body_models/smplh" \
  --xml "$LOCAL_M3_ENV/SMPLSim/outputs/smplh_physics/body_01.xml" \
  --iterations 800 --output outputs/m3_spl_fitting
```

`--fixed-betas`는 zero beta 비교용, `--max-gap`은 보간 상한이다. `--include-fingers`는 미검증 실험 옵션이며 기본 실행에 넣지 않는다. MVP CLI는 다른 session/participant/trial의 fitting을 거부한다.

기존 회귀 테스트가 XML을 팀 outputs에 고정해서 읽으므로 실행 중 테스트 module의 파일 lookup만 원본 XML 위치로 지정한다. 테스트 소스나 원본 파일을 바꾸지 않는다.

```bash
"$LOCAL_M3_ENV/.venv/bin/python" - <<'PYTEST'
from pathlib import Path
import sys, unittest
team=Path.cwd()
sys.path.insert(0,str(team/'tests'))
import test_smplh_motion
test_smplh_motion.ROOT=team.parents[2]/'basketball-r1/SMPLSim'
suite=unittest.defaultTestLoader.discover(str(team/'tests'))
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(not result.wasSuccessful())
PYTEST
```

**13/13 통과**: 기존 회귀 8개 + 새 검사 5개(짧은/긴/경계 gap, spike 제거/보간 가중치, 실제 시도 좌표/좌우/mask/mapping, hip 중점 관측 규약, 실제 full SMPL FK 및 finite gradient). 실제 fitting 실행에서도 손실 유한성, gradient 유한성, MPJPE 초기 대비 개선, full-FK parity, source hash 불변 및 loader/range 검사에 통과했다. 입력 모델의 16-beta 안내 warning 및 기존 docstring/torch warnings는 오류와 구별했다. 처음 matplotlib font cache 경고가 있어 재실행부터 임시 MPLCONFIGDIR를 사용했다.

## 남은 문제와 다음 단계

1. **Wrist orientation/손가락 미복원.** 손목 위치만으로 wrist local rotation을 결정할 수 없으며 기본 fit에서는 wrist 회전이 중립 prior에 의해 결정된다. 손목 flick, fingertip, 그립을 복원한 결과가 아니다. palm/hand 관측과 hand rest calibration을 추가해야 한다.
2. **관절 회전의 비유일성.** body twist/spine 분배는 sparse keypoint에 비해 변수가 많다. L2/시간/soft flexion prior는 학습형 또는 생리학적으로 엄밀한 prior를 대체하지 않는다. body/root MPJPE가 낮아도 pose naturalness를 추가 검수해야 한다.
3. **Source beta 불확실성.** ±3 경계에 3개 coefficient가 도달했으며 hip 오차가 약 34mm로 남는다. marker/regressor 차이, prior 및 제한된 관측의 영향이 있다. zero beta 비교, shape 안정성과 body proportions를 후속 평가해야 한다.
4. **발과 contact 미평가.** ankle rotation은 toe/heel 관측 없이 잘 결정되지 않는다. 바닥 침투, 발 미끄러짐 및 동적 균형은 평가하지 않았다. 물리 적용 전 foot landmark/contact 제약을 준비해야 한다.
5. **NPZ 변환과 물리 추종은 별개.** 현재 PD + COM은 floating root를 직접 servo하지 않는다. source fitted root trajectory가 물리에서 유지될지 별도 검증해야 한다. generic m3_control --motion-file는 쓸 수 있으나 imported --check는 지원하지 않으며 summary가 처음 1초를 제외하는 점도 고려해야 한다. 기존 합성 m3_free_throw의 3초/91-frame 제약을 실데이터에 적용하지 않는다.
6. **목표 체형 Retargeting/PPO/공·골대는 이번 작업에서 제외.** source geometry의 한 프레임 FK/rest 및 foot/palm 관측 검증 뒤 M3-1B-4 실제 모션 물리 재현으로 넘어간다. 그 후에 체형별 retargeting을 분리해 진행한다.

Issue 완료 항목(실제 Issue 수정 없음):

- [x] P0002/T0049 한 시도 loader 및 60FPS/좌표 정규화
- [x] 결측·이상치 탐지, short-gap 보간, 관절별 유효율
- [x] body/finger 후보 mapping 분리 및 hip/좌우/축 검증
- [x] source root/body/고정 beta의 robust 위치 + 시간 + 중립 prior fitting
- [x] 기존 loader 호환 NPZ, 관절별/프레임별 위치 잔차·연속성 저장
- [x] 3D fitting 전후/양팔 잔차 시각화 및 테스트 13개 통과
- [ ] hand/wrist 방향 및 source-to-XML 손 rest 정합
- [ ] 실제 모션 MuJoCo 물리 추종/균형/발 접촉 검증
- [ ] 체형별 retargeting

최종 보존 검사: 원본 데이터셋 594개 파일의 내용 해시 및 원본 개발 환경의 Python/모델 내용은 이전 해시와 동일했다. 팀 기존 Python 116개도 불변을 확인했다.
