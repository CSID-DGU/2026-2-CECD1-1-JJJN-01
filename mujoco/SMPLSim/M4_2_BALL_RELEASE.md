# M4-2: SMPL+H 손-공 연결 및 실제 모션 릴리스

## 구현 범위

`body_01`, SPL `P0002/T0049` 실제 자유투 Fitting NPZ(60 FPS, 240프레임)를 사용했다. 원본 M3 PD + COM Controller, Fitting NPZ, Humanoid XML, M4-1 농구공·골대 XML은 수정하지 않았다. 물리 시간 간격은 1/450초다. 입력 모션은 기존 Loader의 root SLERP/관절 보간을 그대로 사용하며, 약 3.983초 이후에는 마지막 목표 자세를 유지한다. 검증은 공의 비행·바닥 반동까지 관찰하기 위해 총 6초 진행했다.

운동학적 부착과 실제 손 상태에서 계산하는 릴리스가 구현되었다. 득점 가능한 슈팅 속도를 만들어 주는 구현은 포함하지 않는다. 기본 조건의 공은 골대에 도달하지 않는다.

## 파일 구성

추가 파일만 작성했다.

| 파일 | 역할 |
|---|---|
| `smpl_sim/physics/ball_release.py` | 손 offset FK/Jacobian, 운동학적 부착, 단방향 릴리스 상태 전환, 임시 손목 테스트 목표 |
| `examples/m4_ball_release.py` | 실제 모션 물리 실행, GUI/GIF, 궤적·관절·균형·접촉 기록 |
| `examples/m4_release_validation.py` | 릴리스 시점/손목/왼손/재충돌 실험과 비교 그래프 |
| `tests/test_ball_release.py` | 독립 유한차분 속도 검증, freejoint 각속도 좌표계, 물리 자유비행, 부착/충돌/불변성 검증 |
| `M4_2_BALL_RELEASE.md` | 실행 및 평가 문서 |

결과는 Git에서 제외되는 `outputs/m4_ball_release/`에 저장했다. `.venv`, 원본 모델, 원본 데이터셋을 팀 저장소로 복사하지 않았다.

## 손 위치·방향과 공 중심

오른손은 `R_Wrist`, 왼손은 `L_Wrist` MuJoCo body를 사용한다. 계산 대상은 **물리적으로 움직이는 실제 손**이며, 목표 모션의 FK 위치로 공을 발사하지 않는다.

- 손 원점: `data.xpos[hand_body]`, world 미터.
- 손 방향: `data.xmat[hand_body]`, hand-local → world 회전행렬 R.
- 공 중심: `p_ball = p_hand + R @ offset`.
- 기본 오른손 offset: `[-0.07, -0.19, 0]` m.
- 기본 왼손 offset: `[+0.07, -0.19, 0]` m.
- 공 방향: 손의 world 회전 R을 wxyz quaternion으로 저장.

offset은 손목 body 축 기준이며, world Z나 SMPL anatomical 방향을 뜻하지 않는다. 같은 이름의 body와 로컬 축을 사용하므로 다른 체형에도 계산 방식은 적용 가능하지만, 이번 물리 검증은 body_01만 수행했다. offset은 관측된 실제 그립 위치의 복원값이 아니라 MVP의 설정값이다. 기본값은 body_01 손가락 geometry와 초기 겹침을 피하도록 확인했다. 다른 체형/손가락 pose에서는 별도 충돌 검사가 필요하다.

## 부착과 충돌 처리

릴리스 전에는 매 물리 step 직후 공의 qpos/qvel을 손에서 계산한 상태로 맞춘다. 공 geometry와 body의 collision mask를 모두 0으로 설정하여 강제 부착 중 접촉력이 Humanoid에 전달되지 않게 한다. MuJoCo의 broad phase에서 사용하는 body mask도 함께 갱신한다.

**부착 동안 0.6 kg 공의 무게와 관성이 손에 전달되지 않는다.** 공은 freejoint를 갖지만, 부착 구간의 상태를 강제로 맞추는 운동학적 그립이다. 물리적인 두 손 파지나 손가락 마찰 그립은 구현하지 않았다.

릴리스 후에는 M4-1의 공-바닥·백보드·림 충돌을 복구한다. 기본 `--hand-collision shooting`은 슈팅 손의 Wrist와 손가락 geometry에 별도 bit 32를 추가한다. `both`는 양손, `none`은 M4-1과 같은 손 비접촉 조건이다. 팔·몸통과 공의 충돌은 계속 제외한다. 모든 mask 변경은 로드한 M4-2 모델의 메모리에서만 수행하며 M4-1 XML은 다시 저장하지 않는다.

기존 `HumanoidControllerBridge`로 실제 Humanoid qpos/qvel을 원래 모델에 동기화하여 원래 `BalancedPD`를 호출한다. 농구공은 독립 freejoint이므로 Humanoid 질량 행렬 블록/actuator 배열은 그대로다. 접촉 이후 바뀐 실제 Humanoid 상태는 다음 제어 step에 전달된다. 기존 PD에는 손-공 접촉을 미리 보상하는 새로운 힘 제어나 그립 제어를 추가하지 않았다.

## 릴리스 속도와 시점

선속도는 손 원점 속도만 복사하지 않고 offset 회전에 따른 속도를 포함한다.

```text
r_world = R @ offset
v_ball_world = v_hand_world + omega_hand_world × r_world
```

실제 구현은 공 중심에서 `mj_jac`를 계산하여 `J_position @ qvel`, `J_rotation @ qvel`을 사용한다. 손목·팔·Root의 실제 움직임이 모두 포함된다. 유한차분 FK로 이 계산을 독립 검증했다.

공 freejoint의 선속도는 world 좌표계에 저장하고, 각속도는 공 local 좌표계에 저장한다.

```text
qvel_ball_translation = v_ball_world
qvel_ball_rotation = R.T @ omega_hand_world
```

이 좌표계 처리는 `mj_objectVelocity(..., flg_local=0)`의 world 각속도와 비교해 검증했다. 공의 초기 선속도·회전속도는 손 움직임에서 계산하며, 골대에 맞추는 고정 발사 속도를 넣지 않는다. M4-1의 테스트용 발사 함수를 사용하지 않는다.

`--release-time`은 모션 시작 이후 초 단위다. 실제 릴리스는 요청 시점 이상인 최초 물리 step에서 발생하므로 시간 오차는 최대 한 step(약 2.222 ms)이다. 기본 요청은 2.5초다. hand-derived 상태를 마지막으로 설정한 후 collision mask만 바꾸며, 위치·quaternion·속도를 재설정하지 않는다. 이후 `HandBallRelease.update()`는 즉시 반환하고 공 상태를 쓰지 않는다. 실제 실행에서도 update 전후 ball state를 비교하여 덮어쓰기 0회를 확인했다.

연속성 지표는 **동일한 릴리스 시각의 운동학적 상태 → 물리 상태 handoff** 차이다. 다음 물리 step에서 중력이나 손 접촉으로 속도가 바뀌는 것은 실제 가속/접촉 응답이며 연속성 오류로 분류하지 않는다.

## 실행 명령어

모든 명령은 팀 프로젝트 폴더에서 실행한다.

```bash
cd '/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim'
```

기본 GUI와 GIF 생성:

```bash
PYTHONDONTWRITEBYTECODE=1 \
../../../basketball-r1/.venv/bin/mjpython examples/m4_ball_release.py \
  --release-time 2.5 --gui --video --check \
  --output outputs/m4_ball_release/gui
```

6초 실행 후 GUI가 닫힌다. GIF는 `outputs/m4_ball_release/gui/release.gif`이며 반복 재생한다. 노란 점은 공 중심 궤적, 화면 표시는 HELD/RELEASED 상태와 실제 시각이다.

headless 기본 검증 및 비교 실험:

```bash
PYTHONDONTWRITEBYTECODE=1 \
../../../basketball-r1/.venv/bin/python examples/m4_ball_release.py --check

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/m3_mpl_cache \
../../../basketball-r1/.venv/bin/python examples/m4_release_validation.py
```

왼손·릴리스 시점·손목 변경:

```bash
PYTHONDONTWRITEBYTECODE=1 \
../../../basketball-r1/.venv/bin/mjpython examples/m4_ball_release.py \
  --hand left --release-time 2.5 --gui --output outputs/m4_ball_release/left_gui

PYTHONDONTWRITEBYTECODE=1 \
../../../basketball-r1/.venv/bin/mjpython examples/m4_ball_release.py \
  --release-time 2.1 --gui --output outputs/m4_ball_release/early_gui

PYTHONDONTWRITEBYTECODE=1 \
../../../basketball-r1/.venv/bin/mjpython examples/m4_ball_release.py \
  --test-wrist-amplitude 0.35 --gui --output outputs/m4_ball_release/wrist_gui
```

손목 옵션은 선택한 손의 `Wrist_x` 목표에 1.5~2.5초 동안 quintic ramp를 더하는 **별도 검증 조건**이다. 기존 목표 샘플의 복사본만 변경하고 Fitting NPZ/Controller는 변경하지 않는다. 임의의 공 속도 입력 옵션은 없다.

사용자 지정 offset은 예를 들어 `--offset -0.07 -0.19 0`으로 지정한다. 이 값의 단위는 m이다. 손 안쪽으로 공을 겹치게 배치하면 릴리스 때 큰 충격이 생길 수 있다.

## 출력과 평가 정의

각 실험 디렉터리:

- `trajectory.csv`: 450 Hz의 공 위치/선속도, 릴리스 상태, body RMSE, Root 위치·회전 오차, COM XY, 발 힘, support margin, 낙상, 손/환경 접촉.
- `ball_contacts.csv`: 공 접촉 geometry, normal force, penetration depth. 접촉이 없으면 헤더만 저장.
- `joint_tracking.csv`: 153개 hinge 각각의 목표 추종 RMSE.
- `physical_states.npz`: 30 Hz 시각화용 물리 qpos/time. 새 Fitting 결과가 아니다.
- `summary.json`: 부착/연속성/탄도/관절/균형/낙상/물리 경고/기준 상태 비교.

공통 결과: `comparison.csv`, `release_comparison.png`, `release_montage.png`, GUI GIF, 테스트 로그, `preservation_audit.json`.

Body/Root RMSE는 원본 모션 구간에서 450 Hz로 계산한다. Body는 손가락 hinge를 제외하며 손목을 포함한다. Root 오차는 각 체형의 초기 floor alignment가 반영된 목표 기준이다. 양발 접촉률은 전체 6초의 normal force > 0.1 N 조건이다. COM support는 기존 발 geometry 투영의 보수적 지지 영역이며, 공 질량을 포함한 전체 사람+공 COM 지표가 아니다. 낙상은 기존 root 높이/골반 tilt/발 이외 바닥 접촉 조건을 사용한다.

## 실행・정량 결과

macOS Apple Silicon / 기존 Python 3.12 `.venv` / MuJoCo 3.15.0에서 headless 및 `mjpython` GUI+GIF 실행을 완료했다. GIF는 181프레임(30 Hz, 마지막 endpoint 포함)이다. GUI 실행 로그와 렌더링 이미지를 확인했다.

| 기본 오른손, 2.5초 릴리스 | 결과 |
|---|---|
| 공 중심 위치 (world m) | [0.460207, 0.230168, 1.905041] |
| 공 중심 선속도 (world m/s) | [0.71326, -0.501499, 0.180308] |
| 공 각속도 (world rad/s) | [0.290801, 2.551068, -2.985197] |
| 최대 부착 위치 오차 | 0 m |
| 릴리스 위치 / 선속도 / 각속도 jump | 0 m / 0 m/s / 0 rad/s |
| 릴리스 후 state 덮어쓰기 | 0회 |
| 첫 접촉 전 연속 탄도식 위치 오차 | 6.758 mm |
| 첫 접촉 전 속도 오차 | 2.35e-13 m/s |
| 동일 timestep 이산 탄도식 위치 잔차 | 1.56e-13 m |
| Body tracking RMSE | 0.083503 rad |
| Root aligned position RMSE | 0.087309 m |
| Root orientation RMSE | 10.6511 deg |
| 6초 양발 접촉률 | 99.7408% |
| 최소 COM support margin | -0.131683 m |
| 낙상 / MuJoCo 경고 | 0 / 0 |
| M3 기준 240개 시점 최대 Humanoid qpos 차이 | 0 |
| 골대 통과 | 없음 |

연속 탄도식의 6.758 mm 오차는 450 Hz semi-implicit Euler의 위치 적분 차이다. `p_continuous - [0,0,0.5*g*dt*elapsed]`를 사용하는 이산 탄도식과의 잔차는 약 1.6e-13 m이고, 속도는 중력 가속도와 일치한다. 이 검증은 릴리스부터 첫 공 접촉까지의 280개 샘플에 한정한다. 바닥이나 손 충돌 후에는 해당 초기 탄도식을 적용하지 않는다.

COM 최소 margin은 음수이며 초기 접촉 형성 등 과도 상태를 포함한다. 양발 접촉률이나 낙상 0만으로 전 구간 안정 여유가 양수였다고 해석하지 않는다. M3 기준 상태와의 동일성은 유지됐으며, 60 Hz 원래 보고 RMSE(0.0834856 rad)와 이번 450 Hz RMSE의 작은 차이는 평가 sampling 차이다.

### 파라미터 비교

| 조건 | 릴리스 (s) | v_world (m/s) | Body RMSE (rad) | Root RMSE (m) | 손 접촉 프레임 | 낙상 / 경고 |
|---|---:|---|---:|---:|---:|---|
| default | 2.5 | 0.7133, -0.5015, 0.1803 | 0.083503 | 0.087309 | 0 | 0 / 0 |
| early | 2.1 | -0.8535, 0.6285, 2.8148 | 0.083539 | 0.087943 | 7 | 0 / 0 |
| late | 2.9 | -0.1032, -0.8038, -0.0854 | 0.083503 | 0.087309 | 0 | 0 / 0 |
| wrist_test | 2.5 | 0.6325, -0.5078, 0.1929 | 0.083531 | 0.087303 | 0 | 0 / 0 |
| left | 2.5 | -0.3752, 0.5438, 0.1537 | 0.083503 | 0.087309 | 0 | 0 / 0 |
| contact_stress | 2.5 | 0.7565, -0.3004, 0.3564 | 0.083845 | 0.089439 | 3 | 0 / 0 |

`early`는 2.1초에 릴리스한 뒤 2.39778초에 손과 자연스럽게 다시 접촉했다. 손 접촉은 7 physics 프레임이며 최대 normal force 59.53 N, 손 접촉 최대 침투 3.775 mm였다. 실제 접촉력으로 궤적과 Humanoid 상태가 바뀌었지만 state overwrite, NaN, 낙상, 물리 경고는 없었다. 이 경우 최대 M3 qpos 차이는 0.06277로, 접촉을 배제한 기존 실험과 동일할 필요는 없다.

`wrist_test`는 원본 Fitting 파일을 건드리지 않고 선택된 손목 목표에 최대 0.35 rad를 더했다. 기본 대비 릴리스 속도 차이는 약 0.082 m/s이며 궤적이 달라졌다. `early`/`late`도 손 운동의 서로 다른 시점을 사용하여 속도 방향과 높이가 달라졌다. 임의의 정답 발사 속도는 넣지 않았다.

`contact_stress`는 의도적으로 더 안쪽 offset `[-0.07,-0.12,0]`을 써서 릴리스 순간 겹침을 만든 실험이다. 손 접촉 최대 침투 44.13 mm, 최대 normal force 986.47 N으로 강한 튕김이 생겼다. 수치 발산/낙상은 없었지만 **자연스러운 릴리스 조건이 아니다.** 초기 겹침은 큰 접촉 충격과 왜곡된 궤적을 만든다는 실패 사례다. 이 조건은 정상 그립으로 사용하지 않는다. 최초 시점부터 접촉하므로 초기 탄도 검증은 적용하지 않고, 접촉 발생·연속 handoff·무덮어쓰기·유한 상태·낙상/경고 여부를 검사했다. 자연스러운 재충돌 검증은 `early` 조건을 기준으로 한다.

![부착 / 릴리스 / 자유비행](outputs/m4_ball_release/release_montage.png)

![조건별 공 궤적과 속도](outputs/m4_ball_release/release_comparison.png)

[기본 GUI GIF](outputs/m4_ball_release/gui/release.gif) · [실험 비교 CSV](outputs/m4_ball_release/comparison.csv)

## 테스트와 보존 확인

- 운동학적 구현 검증은 기존 22개 + 새 릴리스 5개 = 27개 unittest 통과, skip 없음. Weld 확장 이후 전체 32개도 통과했으며 `M4_2_PHYSICAL_HOLDING.md`에 정리했다.
- M4-1의 자유낙하/바닥/백보드/림/통과/테스트 발사 6개 검증 통과.
- 기존 합성 자유투 네 체형 `--all --check` 통과, 모두 낙상·물리 경고 0.
- M3 기준 실제 모션 240개 시점의 body_01 qpos와 기본 부착 실행의 Humanoid qpos 최대 차이 0.
- 작업 이전 팀 파일 134개, SPL 원본 594개, 원본 개발 환경 snapshot 379개 SHA256 비교: 변경 없음.
- Fitting NPZ SHA256: `af4a970bbac834955ea6222f36a2ef3a320b520a40e9fbf17b85b0a32910200d`.
- M4-1 XML SHA256: `5d1296b19841aa8b4d80e23dffe47a9b1f20a30a5134a190c7b859e1f74205e1`.

전체 테스트는 기존 motion 테스트가 참조하는 로컬 모델 경로만 실행 시 지정했다. 테스트 파일을 수정하거나 원본 모델을 복사하지 않았다.

```bash
PYTHONDONTWRITEBYTECODE=1 ../../../basketball-r1/.venv/bin/python - <<'PY'
from pathlib import Path
import sys, unittest
root = Path.cwd()
sys.path.insert(0, str(root/'tests'))
import test_smplh_motion
test_smplh_motion.ROOT = root.parents[2]/'basketball-r1/SMPLSim'
result = unittest.TextTestRunner(verbosity=2).run(
    unittest.defaultTestLoader.discover(str(root/'tests')))
raise SystemExit(not result.wasSuccessful())
PY
```

## 알려진 한계와 다음 단계

기본 릴리스 속도의 크기는 약 0.89 m/s이며 상승 속도는 약 0.18 m/s다. 골대 높이/거리로 공을 보내기에 부족하므로 자유투 성공 구현으로 취급하지 않는다. 위치 Fitting과 기존 물리 추종만으로 실제 슈팅의 릴리스 속도·손목·손가락 동작·그립이 복원되었다고 볼 수 없다.

운동학적 그립은 공의 중량을 전달하지 않고, 손가락을 공에 맞춰 감싸지 않는다. 현재 손가락 pose는 실제 관측을 Fitting한 결과가 아니다. 공 회전은 손 전체 회전을 복사하며 별도 마찰 기반 backspin을 생성하지 않는다. 양손 지지와 팔/몸통 접촉, 공기 저항은 구현하지 않았다. 릴리스 이후 자연스러운 손 접촉은 안정적이었지만 초기 공-손 겹침은 큰 인공 충격을 만들 수 있다.

다음으로는 실제 질량을 전달하는 물리적 Holding과 운동학적 방식의 차이를 비교하고, 관측된 그립 위치·손목/손가락 pose·릴리스 시점 검증을 진행할 수 있다. 기존 데이터에 임의의 발사 속도를 넣는 보정은 하지 않았다.

GitHub Issue에서 확인할 완료 항목:

- [x] 실제 오른손/왼손 FK와 로컬 offset 적용
- [x] 운동학적 부착 및 한계 명시
- [x] 손 움직임에서 유도하는 릴리스 선속도/각속도
- [x] 릴리스 후 state overwrite 없음
- [x] 시점·손목 변경과 자연스러운 재충돌 검증
- [x] GUI/GIF/CSV/Markdown 결과 저장
- [x] 기존 테스트 및 원본 보존 확인
