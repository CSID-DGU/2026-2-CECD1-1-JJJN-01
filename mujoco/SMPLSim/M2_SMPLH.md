# M2: SMPL+H 체형별 물리 Humanoid

2026-10-10 / macOS Apple Silicon / Python 3.12 / MuJoCo 3.15.0 / CPU.

## 결과

기존 OBJ와 동일한 16개 beta 벡터로 남성 물리 모델 4개를 생성했다. 결과는 outputs/smplh_physics/body_01.xml ~ body_04.xml이다. capsule/box/sphere 충돌 형상으로 구성된 독립 MJCF이며 외부 OBJ/임시 mesh 파일이 필요 없다. mesh=False 경로를 사용했다.

각 모델: 52개 human body + world, 154개 joint (root freejoint 1 + hinge 153), nq=160, nv=159, actuator=153. 손가락은 30개 body, 90개 hinge, 90개 motor이다.

| 체형 | beta 앞 2개 | 질량 kg | 왼쪽 허벅지 m | 왼쪽 정강이 m | 최대 분절 길이 차이 m |
|---|---|---:|---:|---:|---:|
| body_01 | [0.0, 0.0] | 79.598 | 0.3890 | 0.4288 | 0.00000 |
| body_02 | [2.0, 0.0] | 61.417 | 0.3593 | 0.3836 | 0.04609 |
| body_03 | [-2.0, 0.0] | 101.331 | 0.4187 | 0.4739 | 0.04618 |
| body_04 | [0.0, 2.0] | 63.758 | 0.4152 | 0.4450 | 0.02622 |

허벅지 길이는 L_Hip→L_Knee, 정강이는 L_Knee→L_Ankle 관절 간 거리이다. 길이 차이는 body_01 대비 전체 parent offset 길이의 최대 절대 차이이다. CSV의 Pelvis는 부모가 world인 root offset이므로 해부학적 분절 길이로 해석하지 않는다.

질량과 관성은 체형 geometry의 부피 및 SMPLSim 밀도 보정으로 생성한 추정치이다. 실측 인체 질량/관성은 아니다. beta 축은 학습된 shape PCA이므로 beta1 양수가 키/질량 증가라는 의미를 보장하지 않는다.

## 실행 명령

SMPLSim 디렉터리에서:

```sh
# XML 4개 생성 + 체형 비교 + 물리/손가락/Gym 검증
../.venv/bin/python examples/smplh_bodies.py --generate

# 각각 보기: body_01을 body_02, body_03, body_04로 바꾼다
../.venv/bin/mjpython examples/smplh_bodies.py --body body_01

# 4개 순차 보기 (창을 닫으면 다음 체형)
../.venv/bin/mjpython examples/smplh_bodies.py --all

# 4개 GUI를 각 30 control frames만 실행하고 종료
../.venv/bin/mjpython examples/smplh_bodies.py --all --steps 30

# 저장된 XML 4개를 GUI 없이 각각 1350 physics steps 검증
../.venv/bin/python examples/smplh_bodies.py --all --headless --steps 1350

# 기존 Gym HumanoidEnv에서 SMPL+H 사용
../.venv/bin/python examples/env_humanoid_test.py robot=smplh_humanoid headless=True +test_steps=90
../.venv/bin/mjpython examples/env_humanoid_test.py robot=smplh_humanoid
```

--generate는 outputs/smplh_physics의 생성 결과를 갱신한다. 기존 outputs/body_*.obj는 사용하거나 변경하지 않는다. 스크립트 기본 경로는 __file__ 기준 절대 경로라 다른 작업 디렉터리에서도 동작한다. 일반 Hydra 예제의 설정 내 모델 경로는 SMPLSim 작업 디렉터리 기준이다. --model-dir/--output으로 별도 경로를 지정할 수 있다.

GUI 데모는 450Hz physics를 프레임당 15스텝 실행하여 30Hz로 sync한다. 기본 체형에 맞는 초기 높이를 충돌 geometry 최저점으로 계산하고 floor 위 1cm로 배치한다. 아직 균형 학습을 하지 않았으므로 300프레임마다 초기화한다. 이 데모의 손가락 gain은 kp=5, kd=0.5, torque limit=2Nm이며 기존 HumanoidEnv에는 generator의 finger gain을 연결했다. 균형/농구공 제어를 위해서는 추가 튜닝과 학습이 필요하다.

## 모델 연결 및 코드 분석

1. 기존 경고는 robot=smpl_humanoid가 SMPL용 data/smpl을 찾기 때문이었다. SMPL+H 설정 robot=smplh_humanoid는 실제 assets/body_models/smplh를 사용한다. SMPLH_MALE.pkl 및 FEMALE.pkl만 존재하고 NEUTRAL.pkl은 없다.
2. smpl_local_robot.py: SMPL+H는 선택한 성별만 lazy load하도록 수정했다. num_betas=16을 명시하고 cfg.betas를 최초 load_from_skeleton까지 전달한다. 이후 load_from_skeleton(betas=...)도 설정된 성별을 기본 사용한다. rebuild 후 joint_names를 갱신한다. 최초 beta 개수/유한값 검사를 추가했다.
3. smpl_parser.py: SMPLH get_joints_verts의 left_hand_pose/right_hand_pose 인수를 수정했다. 이 함수의 156차원 pose는 global 3 + body 63 + left hand 45 + right hand 45이다. beta를 포함한 forward 결과 vertices/joints로 joint_offsets와 피부 가중치별 convex hull을 만든다.
4. skeleton_local.py (기존 코드 활용): 관절 offset으로 body 계층을 만들고 hull 부피와 길이로 sphere/capsule/box 크기를 만든다. real_weight 및 비율 보정 옵션이 density를 지정하고 MuJoCo 컴파일러가 body_mass/body_inertia를 계산한다. finger geometry와 motor 정의는 이미 있었다.
5. smpl_mujoco_new.py: beta 생성기가 아니라 SMPL pose 축/순서를 MJCF joint 순서에 맞춰 qpos로 변환하는 도구이다. 이번 모델에 model='smplh', euler_order='XYZ', pose=(1,156)를 넣어 결과 (1,160)을 검증했다. 생성된 hinge 축 순서는 XYZ이므로 converter 기본 ZYX 대신 XYZ를 명시해야 한다. 이 파일은 수정하지 않았다.
6. humanoid_env.py: gender/betas 설정을 robot에 전달하고 finger PD gain을 generator의 GAINS_MJ에서 찾도록 연결했다. smplh 경로가 없으면 기본 SMPL fallback 대신 FileNotFoundError를 발생시켜 체형 미적용을 숨기지 않는다.
7. smplh_humanoid.yaml: 16 betas, male, 실제 모델 경로, freeze_hand=False 설정 추가.
8. examples/smplh_bodies.py: 생성/검증/CSV/JSON 출력 및 저장 XML을 읽는 GUI 데모 추가. macOS 연속 launch_passive 종료 경합을 피하기 위해 --all GUI는 별도 프로세스로 실행한다.

## 수치 검증

outputs/smplh_physics/validation.json에 전체 결과, body_properties.csv에 관절/분절/질량/관성, collision_properties.csv에 충돌 종류/크기/위치/contact masks를 저장했다. 관성 단위는 kg·m², 좌표/길이는 m이다. body_world_pos는 각 체형을 바닥 위에 배치한 초기 pose 기준으로 root 높이 차이도 포함한다. 생성된 joint local anchor가 모두 0이므로 body world origin이 관절 위치와 일치한다.

- 모든 비기본 체형에서 local 관절 위치, 분절 길이, geom_size, mass, inertia 변화가 0보다 큼을 assert.
- 모든 동적 body의 질량 및 주관성 양수/유한, MuJoCo 모델 컴파일 성공.
- 모델당 1350 physics steps = 3초: qpos/qvel 유한, MuJoCo 경고 카운터 0.
- 모델당 기존 HumanoidEnv 30 control steps: 관측 유한, 경고 카운터 0. 별도 SMPL+H 기본 모델에서 90 control steps 통과.
- 손가락 30개 body의 90개 hinge 각각에 motor 1개 존재. 중력을 끄고 공중에서 모든 finger motor에 0.002Nm를 동시에 주었을 때 무토크 baseline 대비 모든 finger qpos가 1e-8rad 이상 변화. 관절별 반응 최소 약 0.000204rad. 이는 구동/연결 검증이며 독립 관절 제어 품질/잡기 성공률 검증은 아니다.
- 손 pose 0.3rad perturbation이 vertices 최대 약 0.01477m 변화: 수정한 hand 인수가 실제 적용됨.
- female 파서 로딩 성공, 16번째 beta만 변경했을 때 vertices 차이 약 0.00469m. male robot 재생성에서 16번째 beta에 따른 body_pos 차이 최대 0.003m. 단순히 앞 10 beta만 사용하는 모델이 아님을 검증.
- 15개 beta 전달 시 ValueError 확인. 기존 M1 SMPL fallback 1스텝 회귀 확인.
- 4개 XML 각각 native GUI launch/sync/물리 실행/close 30프레임 완료, warning 카운터 0, 전체 프로세스 exit 0. 화면 캡처/시각 판독은 수행하지 않았다.

SMPL+H는 손가락 각 분절에 3축 hinge를 만든다. 해부학적으로 정교한 손 모델(손가락별 제한된 자유도, 관절 범위, 마찰 및 tendon)은 아니다. 현재 광범위한 joint range와 capsule 충돌은 M2 생성 단계의 시작점이다. 다음 단계에서 공과 손의 충돌/파지, 손가락 limit, 동작 기준 pose와 controller를 검증해야 한다.

## 직접 API 사용

```python
import torch
import mujoco
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot

robot = SMPL_Robot({
    'model': 'smplh', 'gender': 'male', 'betas': [0.] * 16,
    'mesh': False, 'upright_start': False, 'freeze_hand': False,
    'real_weight': True, 'real_weight_porpotion_capsules': True,
    'real_weight_porpotion_boxes': True, 'box_body': True,
}, data_dir='assets/body_models/smplh')

betas = torch.zeros(1, 16)
betas[0, 0] = 2
robot.load_from_skeleton(betas=betas)  # uses configured male gender
model = mujoco.MjModel.from_xml_string(robot.export_xml_string().decode())
```

전체 HumanoidEnv를 체형마다 새로 생성하는 경로가 controller/관측/공간 크기를 일관되게 갱신하므로 권장한다. 기존 env의 mj_model만 교체하지 않는다.
