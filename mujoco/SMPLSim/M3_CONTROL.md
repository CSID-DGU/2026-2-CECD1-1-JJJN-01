# M3: SMPL+H 관절 매핑, PD 기본 동작, 모션 입력

검증일 2026-10-10. macOS Apple Silicon / Python 3.12 / MuJoCo 3.15.0. CPU NumPy/SciPy 기반이며 PPO, CUDA, Isaac Gym을 사용하지 않는다. M2 XML은 변경하지 않았다.

## 완료 결과

네 체형 모두 서기 6초, 팔 올리기 8초, 무릎 굽히기 8초를 실행했다. 총 12개 테스트, 88초의 시뮬레이션, 39,600 physics steps. 최종 결과는 outputs/m3_control/summary.csv.

| 체형 | 서기 RMSE rad | 팔 RMSE rad | 무릎 RMSE rad | 팔 동작 손목 상승 m | 무릎 최대 rad | 전체 최대 골반 기울기 ° |
|---|---:|---:|---:|---:|---:|---:|
| body_01 | 0.01548 | 0.01806 | 0.02201 | 0.280 | 0.621 | 4.45 |
| body_02 | 0.01428 | 0.01724 | 0.02131 | 0.248 | 0.629 | 3.16 |
| body_03 | 0.01917 | 0.02133 | 0.02503 | 0.311 | 0.614 | 6.36 |
| body_04 | 0.01223 | 0.01601 | 0.01711 | 0.297 | 0.603 | 2.65 |

표의 RMSE는 손가락/root를 제외한 63개 body hinge의 공통 기준 목표각 대비 오차이다. 손가락 포함 153개 RMSE는 summary.csv의 joint_rmse_rad. 손가락 정지 관절이 평균을 낮추므로 body 지표를 대표로 사용했다.

전 체형/동작에서 측정 구간 balance_fraction=1, both_feet_contact_fraction=1, fallen_any=0, MuJoCo 경고 0. 최대 발 수평 이동은 약 3.88mm. 이 결과는 평평한 바닥과 정해진 짧은 동작에서의 검증이며, 외란/장시간 안정성이나 자유투 성공률을 의미하지 않는다.

실제 움직이는 관절만 보면 팔 동작 Shoulder_x 오차 약 0.055~0.056rad (3.2°), 무릎 동작 Knee_x 오차 약 0.074~0.091rad (4.3~5.2°)이다. 무릎 목표 최대는 0.5rad이지만 실제 최대는 0.603~0.629rad로 오차가 남아 있다. 기준 동작을 정확하게 복제한 것은 아니다.

## 실행

SMPLSim 디렉터리에서:

```sh
# 네 체형 × 서기/팔/무릎 전체 headless 검증과 CSV 생성
../.venv/bin/python examples/m3_control.py --all --check

# 개별 GUI: 모션 종료 또는 창 닫힘으로 종료
../.venv/bin/mjpython examples/m3_control.py --body body_01 --motion stand --gui --output outputs/m3_gui
../.venv/bin/mjpython examples/m3_control.py --body body_01 --motion arms --gui --output outputs/m3_gui
../.venv/bin/mjpython examples/m3_control.py --body body_01 --motion knees --gui --output outputs/m3_gui

# 네 체형에 팔 올리기 순차 GUI 실행
../.venv/bin/mjpython examples/m3_control.py --all --motion arms --gui --output outputs/m3_gui

# 저장된 동일 모션을 다른 체형에 적용: 초기 root 바닥 높이 정렬
../.venv/bin/python examples/m3_control.py --body body_02   --motion-file outputs/m3_control/body_01_arms_reference.npz   --align-floor --output outputs/m3_imported

# 균형 피드백 제거 비교 (실패도 CSV에 기록)
../.venv/bin/python examples/m3_control.py --body body_01 --motion stand   --no-balance --output outputs/m3_baseline

# 회전/좌표계/동일 목표/NPZ 인터페이스 회귀 테스트
../.venv/bin/python -m unittest discover -s tests -p 'test_smplh_motion.py' -v
```

--frames로 실행을 줄일 수 있다. --check는 전체 길이의 headless 내장 동작에서만 사용한다. --models/--output은 다른 경로를 지정할 수 있다. 기본 경로는 스크립트 위치 기준 절대 경로이다. 같은 output 디렉터리의 동일 체형/모션 결과는 갱신된다. GUI 결과는 별도 outputs/m3_gui를 권장한다. macOS 연속 viewer 실행은 별도 mjpython 프로세스를 사용한다.

## 관절 구조와 qpos

모든 M2 모델에서 human body 52개 + world. njnt=154 = root freejoint 1 + XYZ hinge 153. nu=153, nq=160, nv=159. root는 actuator가 없고 지면 접촉과 관절 토크에 의해 움직인다.

- qpos[0:3]: MuJoCo world 기준 pelvis 위치 (m).
- qpos[3:7]: root quaternion, **w,x,y,z** 순서.
- qpos[7:160]: 관절 hinge 각도 (rad), XML depth-first 순서.
- qvel[0:3]: root 선속도, qvel[3:6]: root 각속도, 나머지 153개 hinge 속도.

SMPL+H 순서는 Pelvis, L_Hip, R_Hip, Torso, L_Knee, R_Knee, ... 이며 XML 순서와 다르다. 156개 pose 배열을 qpos 뒤에 단순 복사하면 잘못 매핑된다. SMPLH_BONE_ORDER_NAMES의 이름별 3차원 axis-angle 회전을 Rotation.from_rotvec으로 만들고 intrinsic XYZ Euler 각으로 분해하여 `{body}_x`, `{body}_y`, `{body}_z`의 실제 qpos 주소에 대입한다.

기본 body 이름 22개:

Pelvis, L_Hip, R_Hip, Torso, L_Knee, R_Knee, Spine, L_Ankle, R_Ankle, Chest, L_Toe, R_Toe, Neck, L_Thorax, R_Thorax, Head, L_Shoulder, R_Shoulder, L_Elbow, R_Elbow, L_Wrist, R_Wrist.

손가락 body 30개: 좌우 각각 Index/Middle/Pinky/Ring/Thumb의 1/2/3 분절. root를 제외한 각 body마다 x/y/z hinge가 있고 각각 이름이 같은 motor가 1개씩 연결된다. 전체 이름, SMPL index, qpos/dof 주소, 축, actuator ID/이름을 `{body}_joints.csv`에 저장했다.

M2 생성 모델의 local hinge 축은 X,Y,Z이고 회전 곱은 Rx·Ry·Rz이다. SciPy의 대문자 'XYZ'가 이 합성과 일치한다. 복합 회전을 포함한 MuJoCo forward kinematics 테스트가 네 모델에서 통과했다. 기존 smpl_mujoco_new.smpl_to_qpose의 기본값 ZYX 및 암묵적 root offset을 피하기 위해 M3에는 명시적인 name/address 기반 매핑을 추가했다.

SMPL Y-up → MuJoCo Z-up 변환 C는 (x,y,z) → (z,x,y), quaternion wxyz=(.5,.5,.5,.5). root orientation은 C·R_global, root_position은 C·position. **local 관절 회전에는 C를 다시 곱하지 않는다.** scipy quaternion 출력 xyzw는 저장할 때 wxyz로 바꾼다.

## 공통 동작과 PD 제어

공통 기준의 관절 pose 배열은 네 체형에서 완전히 동일함을 회귀 테스트로 확인했다. 다리가 다른 체형은 초기 root 높이만 collision geometry로 계산한다. 뼈 길이/shape는 M2 모델 자체에 반영되어 있다.

- 기본 서기: Shoulder_z 좌 -1.35, 우 +1.35rad로 팔을 내림. Hip_x=-0.05, Knee_x=0.1, Ankle_x=-0.05rad.
- 팔 올리기: 2~4초에 양쪽 Shoulder_x를 0 → -1.3rad로 변경, 4~5초 유지, 5~7초 복귀. 같은 팔 각도여도 체형별 손목 이동 거리는 다르다.
- 무릎 굽히기: 같은 시간표로 Hip_x=-0.25, Knee_x=0.5, Ankle_x=-0.25rad까지 변경 후 복귀.
- cubic smoothstep으로 목표를 부드럽게 변경한다. 단일 물리 스텝으로 pose를 덮어쓰는 animation이 아니다.

StablePDController를 physics timestep 1/450초마다 호출하고 실제 motor torque를 적용한다. 초기 자세 설정 이후 root 고정, mocap, equality constraint, 외부 지지력, xfrc_applied를 사용하지 않는다. 예측한 관절 위치/속도에 PD 토크를 주는 기존 SPD를 활용했고 reference velocity feedforward는 추가하지 않았다.

정적인 관절 목표만으로는 부유 root의 균형이 보장되지 않았다. COM과 양쪽 ankle 중심의 수평 오차 및 COM Jacobian으로 얻은 속도를 사용해 다음 목표 보정을 적용한다:

```text
c = clip(8 * (COM_xy - ankle_center_xy - [0.02,0]) + 2 * COM_velocity_xy, -0.6,0.6)
Ankle_x += c_x
Ankle_z -= c_y
Hip_z   += 0.5*c_y
```

세 체형에서 발목만으로 유지되었지만 body_03에서는 좌우 이동이 누적되어 hip 보정이 필요했다. 몸통은 팔 동작 중 중력에 굽는 현상을 줄이도록 높은 PD gain을 사용했다. 최종 모든 체형에 동일한 제어 공식을 적용한다.

기준 gain: Hip/Knee/Ankle/Toe/Torso/Spine/Chest kp=1000, kd=100, 토크 제한 500Nm. 나머지 body kp=300, kd=30, 제한 150Nm. finger kp=5, kd=.5, 제한 2Nm. 각 값을 공통 규칙 `s = total_mass/80 * (mean_leg_length/0.82)^2`로 스케일한다. leg_length는 hip→knee→ankle 길이이고 scale은 약 0.978, 0.618, 1.492, 0.861이다. 목표 동작은 동일하지만 체형별 물리량에 맞춰 gain/limit는 이 규칙으로 달라진다. 학습된 체형별 정책은 아니다.

reference 추종 오차에는 균형 보정에 따른 변화도 포함된다. effective_target_rmse와 joint CSV의 effective_target_rad는 보정한 목표 자체의 추종 오차를 구분하기 위해 함께 저장한다. StablePD가 hinge qpos/dof와 actuator 순서가 같은 모델을 요구하므로 이를 검사하고 순서가 다르면 명시적으로 실패한다.

## 측정 지표와 출력

- summary.csv: 12개 테스트의 RMS 오차/균형 비율/양발 접촉 비율/골반 tilt/최저 root 높이/발 drift/손목 상승/최대 무릎각/gain scale/넘어짐/경고.
- `{body}_{motion}_timeseries.csv`: 30Hz 시간별 지표. 최초 1초 settling은 summary 측정에서 제외하며 CSV에는 남긴다.
- `{body}_{motion}_joint_tracking.csv`: 10Hz 모든 153개 joint의 원래 목표/보정 목표/실제 각도/오차/토크.
- `{body}_joints.csv`: 관절·actuator 구조.
- `{body}_{motion}_reference.npz`: 모션 입력 인터페이스 형식으로 저장한 기준 동작.
- `{body}_{motion}_summary.csv`: 개별 실행 결과. summary.csv는 한 명령에 실행된 테스트를 합친 결과이다.

floor contact에서 mj_contactForce로 normal force를 얻는다. foot=Ankle/Toe body이고 각 발의 force 합이 1N을 넘으면 접촉으로 판정한다. 힘이 실리는 발의 collision box 모서리를 평면에 투영한 convex hull로 support_margin을 계산한다. 이것은 지지영역의 근사 지표이며 실제 CoP/ZMP나 contact wrench feasibility를 계산한 결과는 아니다.

넘어짐: root 높이가 초기의 65% 미만, 골반 tilt>35°, 또는 힘이 있는 nonfoot-floor 접촉 중 하나. 균형: 넘어짐 없음, tilt<25°, 한 발 이상 접촉, COM projection이 근사 지지영역 밖 2cm 이내. tilt는 pelvis local Y축과 world Z축의 각도이므로 상체 전체 기울기와 구분한다.

--check 기준: balance/양발 접촉 ≥95%, 넘어짐 없음, body RMSE≤.08rad. 팔: 평균 손목 상승≥.15m 및 Shoulder_x RMSE≤.1rad. 무릎: 최대 각도≥.4rad 및 Knee_x RMSE≤.15rad. 이를 임상/스포츠 동작의 정확성 기준으로 해석하지 않는다.

## 모션 NPZ 인터페이스

schema_version=1, allow_pickle=False:

| 필드 | 형태 | 의미 |
|---|---|---|
| poses | (N≥2,156) float | radians axis-angle: global 3 + body 63 + left hand 45 + right hand 45 |
| root_position | (N,3) float | m 단위 root 위치 또는 명시적으로 선택한 SMPL transl |
| fps | scalar >0 | 기준 모션 샘플링 속도 |
| coordinate_system | scalar string | smpl 또는 mujoco (root pose/position 좌표계; local 관절은 rig 좌표) |
| translation_kind | scalar string | root_position 또는 smpl_transl |
| schema_version | scalar int | 1 |

root_position은 pelvis 위치를 직접 의미하므로 모델의 root offset을 자동으로 더하지 않는다. smpl_transl은 SMPL 좌표에서의 mesh translation으로 간주하고 모델의 shaped pelvis rest offset(body_pos[Pelvis])을 더한 뒤 C로 변환한다. SMPL transl convention은 외부 데이터셋 adapter에서 반드시 확인해야 한다. --align-floor는 첫 pose를 floor 위 5mm에 배치하도록 전체 root trajectory에 일정한 위치 offset만 더한다. 시간별 자동 높이 보정이나 발 IK는 구현하지 않았다.

```python
import numpy as np
from smpl_sim.motion import SMPLHMotion

motion = SMPLHMotion(
    poses=pose_axis_angle,       # (N,156)
    root_position=pelvis_xyz,    # (N,3)
    fps=30,
    coordinate_system='smpl',
    translation_kind='root_position',
)
motion.save('reference.npz')
bound = motion.bind(mj_model)
qpos_target = bound.sample(1.25)  # (160,)
```

root orientation은 SLERP, 위치와 unwrapped hinge 각은 선형 보간한다. 관절 Euler 보간은 full SO(3) motion interpolation과 다를 수 있다. gimbal lock, joint range 초과, 비유한값, 형식 오류를 명시적으로 거부한다. 현재 PD는 관절 목표만 제어하고 root trajectory는 초기화/오차 측정에 사용한다. 동작 전체 root 경로를 추종하려면 다음 단계에서 발 IK/contact-aware retargeting과 전신 제어가 필요하다.

M2 rig는 SMPL+H 손 mean pose로 형상/offset을 만들지만 local joint frame을 단순화한 모델이다. 이 매핑은 생성 rig의 local 회전 convention을 검증한 것이며 원본 SMPL mesh와 모든 손가락 FK가 정확히 일치함을 검증한 것은 아니다. 외부 hand motion에서는 flat_hand_mean/hand mean, 원본 rest frame, 관절 limit를 adapter로 정렬해야 한다.

## 변경 파일과 검증

- smpl_sim/motion/__init__.py: 모션 API export.
- smpl_sim/motion/smplh_motion.py: name/address 매핑, root convention, XYZ 분해, NPZ 입출력, target sampling.
- examples/m3_control.py: 기준 동작 생성, PD+balance feedback, 물리 실행, GUI, 접촉/오차 측정, CSV 출력, imported motion 및 바닥 정렬.
- tests/test_smplh_motion.py: compound FK와 root orientation/translation, 모든 체형의 공통 reference, NPZ interpolation, 입력 오류 회귀 테스트.
- M3_CONTROL.md: 이 문서.

검증:

1. `--all --check`: 12개 최종 headless 테스트 통과, 모든 MuJoCo warning 카운터 0.
2. unittest: 5개 테스트 통과. compound XYZ 회전의 실제 local FK와 기대 회전 오차 약 1.7e-16.
3. body_01의 arms NPZ → body_02에 --align-floor로 입력: 8초 실행, 균형/양발 접촉 100%, 손목 상승 0.248m, 경고 0. 자체 생성 body_02 reference와 동일한 결과.
4. body_03의 세 동작 native GUI 전체 실행: 6/8/8초, headless와 동일한 수치, exit 0. 화면 캡처/시각 판독은 수행하지 않았다.
5. `--no-balance` body_01 서기 비교: 이 실험 당시의 gain 설정에서 root 높이 약 .069m로 떨어지고 넘어짐을 검출했다. 이 비교 이후 최종 몸통 gain을 보완했으므로 최종 PD의 엄밀한 ablation 결과로 사용하지 않는다. 이전 tuning 출력은 최종 summary에 포함하지 않았다.
6. compileall과 git diff --check 통과.

다음 자유투 단계는 실제 기준 pose sequence 확보/좌표계와 hand rest frame 확인, 발 고정/접촉을 고려한 root·하지 retargeting, 속도 목표 및 tracking reward 구성이다. 현재 완료 범위는 기본 동작과 입력 인터페이스이며 자유투 데이터 확보나 PPO 학습은 수행하지 않았다.
