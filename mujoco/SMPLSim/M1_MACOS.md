# M1: macOS / MuJoCo 3.15.0 실행 및 SMPL+H 연결 분석

검증일: 2026-10-10. Python 3.12, Apple Silicon, 프로젝트 .venv.

## 실행

SMPLSim 디렉터리에서 실행한다.

```sh
# GUI: 창을 닫으면 종료. Space 일시정지, R 재설정, F 카메라 추적.
../.venv/bin/mjpython examples/env_humanoid_test.py

# GUI 유한 길이 검증
../.venv/bin/mjpython examples/env_humanoid_test.py headless=False +test_steps=90

# 화면 없이 물리 검증
../.venv/bin/python examples/env_humanoid_test.py headless=True +test_steps=90

# RGB 영상 (렌더링에는 headless=False 필요)
../.venv/bin/python examples/env_humanoid_test.py headless=False env.render_mode=rgb_array +test_steps=90
```

RGB 명령은 제공했지만 이번 작업에서는 검증하지 않았다. 영상 경로는 실행 작업 디렉터리의 video.mp4이다.

## 원인 및 변경

- config.yaml 기본 headless=True가 human 렌더링을 건너뛰었다. 기존 예제는 무한 루프여서 계속 실행되는 것이 정상 동작이었다.
- examples/env_humanoid_test.py: 예제에서만 GUI를 기본 활성화한다. 명시한 headless 설정은 유지한다. test_steps, 단계별 로그, 유한 상태 검사, 종료/재설정 처리, finally 자원 정리를 추가했다. human 모드에서 step이 이미 렌더링하므로 중복 render 호출과 None 프레임 누적을 제거했다. eval과 불필요한 imports를 제거했다.
- controllers.py: mj_mulM 단위벡터 반복 대신 mj_fullM을 사용한다. 설치된 3.15.0 API는 mj_fullM(model, data, dst). qM이 있는 구버전은 기존 API를 사용한다. 구버전 분기는 이번 환경에서 실행하지 않았다.
- base_env.py: mujoco.viewer 명시적 import, 창 닫힘 확인, 카메라 변경 lock, viewer/renderer 자원 정리를 추가했다. render_mode=None일 때 렌더러를 만들지 않는다.
- humanoid_env.py: fallback 경고에 실제 찾는 경로와 사용자 체형 미적용 사실을 명시했다.
- 학습용 전역 headless 기본값은 유지한다. CUDA나 Isaac Gym을 추가하지 않았다.

## 검증 결과

- headless 90 control steps: exit 0, time=3.0초, qpos/qvel 모두 유한.
- mjpython human 90 steps: launch_passive, reset, sync, close 완료, exit 0.
- 추가 GUI 600스텝: exit 0, 20초 누적 실행 완료. 에피소드 제한에 따라 중간 재설정 포함.
- nv=75, nu=69, gravity=[0,0,-9.81], 초기 대비 qpos 변화 L2=1.5911886074.
- 접촉 개수 16 → 8, 제어 norm 39.281 → 58.636. 90스텝 후 MuJoCo warning 카운터 모두 0.
- mj_fullM과 mj_mulM 열 복원 행렬의 최대 오차 0.0, 최소 고유값 0.0100539304 (bundled 모델 초기 상태).
- GUI 라이프사이클은 실제 실행으로 검증했다. 화면 캡처는 OS 접근성/화면 기록 권한 대기로 완료하지 못했으므로 외형을 시각적으로 확인했다고 주장하지 않는다.
- 이것은 기본 물리 및 PD 제어 smoke test이며, 자유투나 장시간 균형 안정성을 검증한 결과는 아니다. 0 action도 PD 모드에서는 0 torque가 아니다.

## SMPL+H 모델 경로와 장애물

현재 robot=smpl_humanoid는 humanoid_type=smpl, smpl_data_dir=data/smpl이다. 이 경로가 없어 bundled smpl_humanoid.xml을 사용한다. OBJ는 이 과정에 사용되지 않는다.

실제 로컬 모델은 assets/body_models/smplh/SMPLH_MALE.pkl 및 SMPLH_FEMALE.pkl. SMPLH_NEUTRAL.pkl은 없다. Extended npz는 별도 디렉터리에 있다.

smplx.create와 달리 SMPLSim의 SMPLH_Parser는 모델 클래스에 경로를 직접 전달한다. 다음은 실제 로딩 및 forward 검증에 성공했다:

```python
from smpl_sim.smpllib.smpl_parser import SMPLH_Parser
p = SMPLH_Parser(model_path="assets/body_models/smplh", gender="male",
                num_betas=16, use_pca=False, create_transl=False)
# vertices (1,6890,3), joints (1,73,3)
```

assets/body_models를 직접 전달하면 SMPLH_MALE.pkl 경로 AssertionError가 발생한다.

경로 변경만으로 HumanoidEnv가 준비되지는 않는다:

1. SMPL_Robot.__init__은 neutral/male/female을 모두 로딩하고 기본 neutral 체형을 생성한다. 사용할 성별만 로딩하도록 변경하고 초기 load_from_skeleton에도 그 성별을 전달해야 한다. 남성 파일을 neutral로 복사하는 것은 의미가 달라 권장하지 않는다.
2. SMPLH 파서는 기본 num_betas가 16임을 가정하면 안 된다. 로봇 생성에서 num_betas=16을 명시해야 한다.
3. SMPLH_Parser.get_joints_verts가 L_hand_pose/R_hand_pose를 전달한다. 설치된 smplx에서 쓰는 인수는 left_hand_pose/right_hand_pose이다. 기본 자세 forward는 성공하지만 손가락 동작 전달은 별도 수정/검증이 필요하다.
4. HumanoidEnv의 GAINS는 SMPL 24-body 이름 기준이다. SMPL+H의 개별 손가락 joint 이름에 대한 PD gain/torque limit과 관측/초기화 배열을 점검해야 한다. 손가락은 생성 가능성만으로 현재 환경 제어가 지원된다고 볼 수 없다.

## 체형별 물리 모델 생성 방법 (M2 설계)

기존 SMPL_Robot.load_from_skeleton(betas=..., gender=[1]) 경로를 활용할 수 있다. gender는 0 neutral, 1 male, 2 female. 위 로딩 문제를 먼저 해결한 뒤 CPU torch.float32 (1,16) betas를 전달하고 export_xml_string으로 MJCF를 만들고 MuJoCo 모델을 다시 컴파일한다.

```python
# 성별 선택 로딩과 num_betas 수정 이후 적용할 흐름
robot.load_from_skeleton(betas=betas, gender=[1])
model = mujoco.MjModel.from_xml_string(robot.export_xml_string().decode("utf-8"))
```

load_from_skeleton은 beta로 vertices/joints를 다시 계산하고 joint_offsets, 각 분절 convex hull/volume을 재생성한다. capsule/box 또는 mesh 생성기가 이 정보를 이용해 충돌 형상을 만든다. 밀도/부피로 질량이 설정되고 MuJoCo 컴파일 시 관성이 생성된다. 이 질량 모델은 형상 기반 근사이며 측정된 개인별 질량/관성이 아니다. real_weight 옵션도 이 근사 밀도/부피 보정이다.

각 체형에서 body_pos, geom_size, body_mass, body_inertia의 변화와 양수/유한 여부를 비교해야 한다. 이후 접촉, 관절 limit, 안정성을 검증한다. 환경 모델만 교체하면 controller와 observation/action space가 이전 모델을 참조할 수 있으므로 전체 환경을 재생성하거나 모든 관련 설정을 재구성해야 한다.

M1은 bundled SMPL 모델로 실행 검증했다. 사용자 SMPL+H beta 물리 모델 4개 생성은 아직 구현하지 않았다.

MuJoCo 공식 macOS viewer 안내: https://github.com/google-deepmind/mujoco/blob/main/doc/python.rst
