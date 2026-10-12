# Standalone reproduction: body_01 / 200g Weld free throw

현재 성공한 `followthrough_forward_200g` 조건을 공유하는 작은 패키지입니다.
실행 진입점은 **run.py**입니다. 이전 실험 코드, 로컬 `basketball-r1` 경로,
SPL JSON 원본, 원본 fitting NPZ를 읽지 않습니다.

## 필요한 것

- Python 3.12; 아래 requirements의 MuJoCo 3.15.0 / NumPy / SciPy
- 동일한 **body_01.xml**: 남성 SMPL+H, 16개 beta 모두 0, 기존 M2 생성 설정
- macOS GUI는 `python` 대신 **mjpython** 실행

MuJoCo와 SMPL+H를 설치한 것만으로는 동일 조건이 완성되지 않습니다.
팀원이 자신의 라이선스로 확보한 SMPL+H 모델로 동일한 humanoid XML을 생성해야 합니다.
원본 모델/생성된 인체 XML은 이 패키지에 포함하지 않습니다.
컴파일된 모델의 관절·질량·관성·충돌·actuator fingerprint를 검사하여 다른 체형은 거부합니다.

기존 팀 코드에서 XML 생성:

```bash
# 팀 저장소 mujoco/SMPLSim에서 실행; M2 생성에 필요한 기존 프로젝트 의존성도 필요
python examples/smplh_bodies.py --generate \
  --model-dir /absolute/path/to/licensed/smplh \
  --output /absolute/path/to/local_models
```

이미 생성한 `body_01.xml`이 있다면 그대로 사용합니다. 이 파일은 각자 로컬에 보관합니다.

## 설치와 실행

```bash
# 팀 저장소 루트에서
cd mujoco/SMPLSim/repro/free_throw_200g
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

저장된 실제 시뮬레이션을 무한 반복 재생:

```bash
mjpython run.py replay --humanoid /absolute/path/to/body_01.xml \
  --view court --repeats 0
```

- `--repeats 3`: 세 번 재생
- `--view side`: 사람 측면 보기
- `--speed 2`: 화면 재생 2배속. 물리 timestep이나 동작 자체를 변경하지 않습니다.
- GUI 창을 닫으면 종료합니다.
- replay는 저장된 qpos 재생입니다. 새로운 물리 실험으로 간주하지 않습니다.

**물리를 처음부터 다시 계산하고 동일 결과인지 검증:**

```bash
python run.py simulate --humanoid /absolute/path/to/body_01.xml --verify
```

물리 실행을 GUI로 보기:

```bash
mjpython run.py simulate --humanoid /absolute/path/to/body_01.xml --gui --view court
```

새 결과는 현재 실행 디렉터리의 `outputs/free_throw_repro/`에 저장됩니다.
`--output /path/to/result`로 바꿀 수 있습니다.
`summary.json`, `states.csv`, `ball_flight.csv`를 저장합니다.
`--verify`는 득점·낙상·경고뿐 아니라 기존 릴리스 위치/속도와 전체 replay qpos 오차를 확인합니다.
다른 CPU/수치 라이브러리에서 작은 차이가 발생하면 검증 실패 결과를 보고 조사해야 합니다.

## 재현 검증 (2026-10-12, Apple Silicon)

- 전체 저장 qpos 최대 차이: **0.0**
- 릴리스 위치 및 속도 차이: **0.0**
- 릴리스: 1.6511111111110868초
- 공 속도: **7.6977036252m/s**; 수직 속도 6.3560874536m/s
- 첫 비행에서 공 전체가 아래 방향으로 림 통과: **성공**
- 낙상 0, MuJoCo 경고 0
- 공 초기 정렬 1회; 이후 qpos/qvel 강제 변경 0회
- 원래 실험과 전체 궤적이 동일하며 기존 소스/모델/결과를 수정하지 않았습니다.

## 조건과 한계

200g 공, 반지름 121mm, Weld 기반 holding입니다. 릴리스 시 Weld만 비활성화하고
실제 공 위치·속도를 유지합니다. 외부 지지력이나 정답 발사 속도를 넣지 않습니다.
이번 조건은 손-공 충돌을 비활성화한 기존 성공 실험을 그대로 재현합니다.
왼손의 실제 접촉 기반 지지, 표준 600g 농구공, PPO 성능을 검증한 패키지가 아닙니다.
공기 저항은 없습니다. 결과는 릴리스 타이밍에 민감하며 강건한 최적 동작으로 해석하지 않습니다.
기존 평가에서 최소 COM 지지 여유는 -72.4mm로 순간적으로 음수였습니다.
골 통과 성공과 자연스러운 자유투/완전한 균형 달성은 별도입니다.

## 포함 파일과 라이선스

- `run.py`: replay / simulate CLI
- `engine.py`: 독립 PD+COM, motion interpolation, Weld release, score detector
- `environment.xml`: 바닥·골대·공·Weld 템플릿; 인체 구조는 실행 시 외부 XML에서 결합
- `settings.toml`: 모든 실험 설정, 기준 결과, 모델/모션 checksum
- `data/reference.csv`: 목표 궤적
- `data/replay.csv`: 실제 계산한 상태 궤적
- `requirements.txt`: 검증한 버전

Controller는 SMPLSim 코드에서 가져온 것으로 BSD-3-Clause 고지를
`licenses/CODE_LICENSE.txt`에 보존합니다.
모션 데이터의 원천은 [SPL Open Data](https://github.com/Sport-Performance-Lab/SPL-Open-Data)의
2025 / P0002 / T0049입니다. 수정·fitting·새 shooting trajectory 생성 및 물리 계산을 거쳤습니다.
원천 데이터의 **CC BY-NC-SA 4.0 및 추가 사용 제한**을
`licenses/DATA_LICENSE.txt`에 포함했습니다. 파생 모션 데이터도 이 조건을 따르는 것으로 배포합니다.
공유할 때 라이선스 파일을 함께 유지하세요.
[SMPL 모델 라이선스](https://smpl.is.tue.mpg.de/modellicense.html)에 따라 원본 인체 자산은 각자 확보합니다.

GitHub 공유 범위는 이 폴더입니다. `.venv`, `local_models`, `outputs`는 제외됩니다.
원본 SMPL+H pkl/npz, 개인 경로의 XML, 전체 SPL 데이터셋을 추가하지 마세요.
