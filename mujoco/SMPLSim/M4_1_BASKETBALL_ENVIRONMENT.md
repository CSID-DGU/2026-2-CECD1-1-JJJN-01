# M4-1: MuJoCo 농구공 및 골대 물리 환경 구축

검증일 2026-10-10. macOS Apple Silicon arm64, Python 3.12.13, MuJoCo 3.15.0. 대상은 기존 body_01과 실제 자유투 fitted NPZ다. 농구공은 독립 freejoint이고 손에 연결되지 않는다. 발사는 물리 검증용 초기 속도이며 실제 사람의 릴리스 모션과 동기화한 shot가 아니다. PPO·RL·체형 retargeting·네트는 구현하지 않았다.

**자유낙하, 바닥 반발, 백보드/림 충돌, 림 중앙 통과, 초기 위치/속도 변경을 통과했다.** 통합 모델에서 공이 바닥에 튀는 동안 Humanoid의 4초 관절/root 상태는 기존 M3 결과와 정확히 같았다. GUI 실행과 여섯 테스트 GIF 및 4초 GUI GIF를 저장했다.

## 1. 적용 기준·단위·배치

공과 주요 골대 치수는 **FIBA 2024 Basketball Equipment**를 명시적으로 선택했다. 최신 규정이라고 주장하지 않으며 인증 장비를 완전히 구현한 것도 아니다. Size 7 범위는 둘레 750–770mm, 질량 580–620g이고, 1.8m(공 아래쪽) 낙하 시 반발 아래쪽 높이 범위는 1.035–1.085m다. [FIBA 2024 Equipment §1–2](https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2024-official-basketball-rules-and-basketball-equipment.pdf)

| 항목 | 구현값 | 단위/의미 |
|---|---:|---|
| 공 반지름 | 0.121 | m, 둘레 0.760265m |
| 공 질량 | 0.600 | kg |
| 공 관성 각 축 | 0.0058564 | kg·m², 균일 얇은 구형 껍질 근사 I=2mr²/3 |
| 중력 | (0,0,-9.81) | m/s², world Z-up |
| 림 nominal 내경 | 0.454 | m |
| 림 금속 단면 지름 | 0.020 | m |
| 림 centerline 반경 | 0.237 | m |
| 림 위쪽 높이 | 3.050 | m |
| 림 centerline 높이 | 3.040 | m, 튜브 반지름만큼 낮게 배치 |
| 백보드 | 1.800 × 1.050 × 0.040 | m, Y 폭 × Z 높이 × X 두께 |
| 백보드 아래/위 | 2.900 / 3.950 | m |
| 림 내부 뒤쪽→백보드 면 | 0.151 | m |
| 자유투 선 먼쪽 X | 0.350 | m, 원본 인체 stance를 선 뒤에 두는 scene 위치 |
| 백보드 앞면 X | 4.950 | m |
| 림 중심 | (4.572,0,3.040) | m, 림 centerline |

Equipment 기준은 림 내경 450–459mm, 금속 지름 16–20mm, 윗면 높이 3050mm 및 뒤쪽 내부점과 보드 면 간격 151mm를 제시한다. 구현값은 이 치수 범위에서 선택했다. 백보드 두께 40mm는 장면용 가정이며 공식 장비 인증값이 아니다. [FIBA 2024 Equipment §1](https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2024-official-basketball-rules-and-basketball-equipment.pdf)

공격 방향은 world +X이고 Y는 좌우다. 자유투 선 먼쪽 가장자리→백보드 앞면은 4.600m, 선→림 중심은 4.222m다. 규정의 자유투 선 먼쪽 5.8m 및 보드 면 1.2m의 endline 기준 위치로부터 4.6m를 사용했다. FT 선은 폭 50mm, 길이 3.6m의 비충돌 장식 geom이다. [FIBA 2024 Rules §2.5.2–2.5.3](https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2024-v10a.pdf)

위치 0.350m는 원본 humanoid root/발을 이동하지 않고 선 뒤에 유지하기 위한 scene 원점 선택이다. 실제 자유투 line violation 판정은 구현하지 않았다. 백보드 지지대는 시각 장식이며 non-colliding이다. 압력해제 림, 네트, 보드 변형·진동·패딩 인증은 구현 범위 밖이다.

## 2. XML과 충돌 구조

생성 XML: [body_01_basketball.xml](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/outputs/m4_basketball/body_01_basketball.xml). 원본 XML을 읽고 파생 사본에만 추가한다. 기존 SMPL+H pkl/XML/NPZ는 수정하지 않았다.

- 공: sphere 하나, 독립 body basketball + freejoint basketball_free. explicit inertial로 질량과 shell 근사 관성을 설정했다. ball actuator는 없다.
- 골대: worldbody에 고정 box 백보드와 64개 capsule로 연결한 림. 원형 구멍을 막는 cylinder/solid disk는 사용하지 않는다.
- 64각 chord 근사에서 최소 내부 반경은 `0.237*cos(pi/64)-0.010=0.226715m`(유효 내경 약 453.429mm)다. 끝 capsule의 중첩으로 림을 닫고 구멍은 열려 있다.
- 원본 human freejoint 1+hinge 153은 그대로며 공 freejoint 하나가 뒤에 추가된다. 통합 nq=167, nv=165, njnt=155, nu=153. joint name/axis/range/address, body mass/inertia/offset, actuator transmission/gear/range 보존을 검사했다.
- Humanoid-공, Humanoid-골대 충돌은 M4-1 단계에서 의도적으로 비활성화한다. 공을 손목 위치에 겹쳐도 인체 충돌이 발생하지 않는 것을 검사했다. 공-바닥/림/백보드 및 기존 인체-바닥/인체-인체는 유지한다.
- 원본 collision filtering의 OR 규칙을 유지하면서 새 bits를 사용한다: human 원본 masks, 공 contype=8/conaffinity=16, hoop 16/8, floor 23/9(원본 7/1에 ball/environment bits 추가). FT 선·지지대는 0/0이다.
- 새로운 ball/hoop 접촉만 condim=6으로 sliding/torsional/rolling 마찰을 사용한다. 기존 human contact geom의 solver/friction은 바꾸지 않는다. [MuJoCo contact parameters](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters)

## 3. 마찰·반발 파라미터와 한계

| 대상 | priority | friction: sliding/torsion/rolling | solref | solimp |
|---|---:|---|---|---|
| 공 (바닥 접촉을 결정) | 2 | 0.7 / 0.005 / 0.0002 | -40000 -45 | 0.95 0.99 0.01 |
| 백보드 | 3 | 0.6 / 0.005 / 0.0002 | -80000 -60 | 0.95 0.99 0.01 |
| 림 | 3 | 0.6 / 0.005 / 0.0002 | -80000 -80 | 0.95 0.99 0.01 |

새 공/골대 geom margin=0이고 floor는 기존 margin=0.001m를 유지한다. 높은 priority의 geom 파라미터가 접촉에 적용되므로 공-바닥은 공, 공-백보드/림은 각 hoop geom 값을 사용한다. solref 두 음수는 direct (-stiffness,-damping) 형식이며 별도 restitution 속성으로 설정한 것이 아니다. [MuJoCo solver parameters](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters)

바닥 테스트에서 1.8m 아래쪽 높이 낙하의 첫 반발 아래쪽 높이는 **1.060573m**다. 접촉 직전/이탈 수직 속도 비율의 유효 반발은 **0.770295**다. 채택한 FIBA 2024 반발 범위에 이 단일 테스트가 들어오는지 --check에서 확인한다. 반발 높이만 맞춘 것이며 실제 공 재료·압력·표면 인증의 검증은 아니다.

이 soft-contact 모델은 침투를 허용한다. 최대 침투는 바닥 낙하 약 25.16mm, 고속 비행 후 바닥 약 33.27mm, 보드 8.54mm, 림 8.88mm다. 충돌 속도/시간 step에 따라 유효 반발이 달라져 단일 상수 coefficient of restitution으로 해석하면 안 된다. 공기저항·Magnus·구체 변형/내부 압력은 없다.

물리 timestep은 기존과 같은 1/450초다. 빠른 얇은 림 충돌은 더 작은 timestep·다양한 impact angle/speed의 수렴 검증이 향후 필요하다. 이번 범위의 시험은 유한 상태이고 물리 경고가 없었다.

## 4. 기존 Humanoid 통합

기존 SMPLHMapping은 nq160/nv159/nu153만 받아들이고 StablePD는 free root 이후 모든 dof가 hinge인 구조를 가정한다. 공이 추가된 nq167/nv165 모델에 기존 코드를 직접 넣으면 크기/게인 배열이 맞지 않는다. 이 파일들을 수정하지 않았다.

`HumanoidControllerBridge`는 원본 Humanoid-only MjModel/MjData에서 기존 Loader/PD+COM을 그대로 사용한다. 매 스텝 통합 모델의 human qpos/qvel/ctrl/time을 shadow data에 복사하고 mj_forward 후 기존 제어기로 구한 153개 토크를 통합 model.ctrl에 전달한다. 전체 물리 적분은 통합 모델에서 한다. 공은 freejoint 중력·접촉으로 움직이며 per-step 외력은 넣지 않는다.

초기 floor alignment는 ball/hoop가 포함된 ground()를 호출하지 않고 원본 human 모델에서 계산한다. Source NPZ의 모든 pose, root 및 시간을 보존하고 기존 world Z 상수 offset만 적용한다. gain_scale은 원본 human 질량/다리길이에 기반하며 공 질량으로 변경되지 않는다.

22개 테스트 중 하나는 공이 바닥에 튀는 통합 모델을 1793스텝 동안 실행해 기존 M3의 240개 저장 상태 전체와 비교했다. human qpos 최대 차이는 **0**이었다. 현재는 공/골대와 인체가 물리적으로 분리되어 이 bridge가 유효하다. 향후 손-공 접촉/연결이 생기면 shadow 모델이 hand contact force를 모르므로 이 구조를 그대로 실제 release 제어기로 사용하면 안 된다.

## 5. 물리 시험 결과

| 시험 | 위치 xyz m | 초기 속도 xyz m/s | 결과 |
|---|---|---|---|
| freefall | [2.0, 0.0, 2.0] | [0.0, 0.0, 0.0] | 무접촉 낙하, analytic 최대 오차 3.270mm |
| floor | [2.0, -1.0, 1.921] | [0.0, 0.0, 0.0] | 첫 아래쪽 반발 1.060573m, 유효 반발 0.770295 |
| backboard | [4.35, 0.55, 3.65] | [3.0, 0.0, 0.0] | 보드 접촉 후 vx=-2.099192m/s로 반전; 림 접촉 없음 |
| rim | [4.334999999999999, 0.0, 3.64] | [0.0, 0.0, -1.0] | 림 capsule 접촉·반발, 구멍 통과 아님 |
| pass | [4.571999999999999, 0.0, 3.74] | [0.0, 0.0, -1.0] | 수직 중앙 통과; plane entry 0.288525s; 림 접촉 없음 |
| launch | [0.0, 0.0, 2.0] | [4.571999999999999, 0.0, 5.945] | 테스트 발사 통과; plane entry 0.997176s; 림/보드 접촉 없음 |

모든 시험에서 인체-공 접촉 기록 0, NaN/Inf 없음, 7종 MuJoCo warning 카운터 합 0, --check 통과다. contacts.csv/summary의 contact_samples는 힘>0.1N인 접촉 기록 수이며 여러 capsule이 같은 시각에 접촉하면 여러 개로 센다.

추가 초기 조건 변경 시험: 위치 [2,-0.5,2.2]m, 속도 [1.5,0.4,0.6]m/s, 0.3초. 자유 비행 최대 analytic 오차 3.270mm, 경고 0으로 통과했다.
- Analytic 비교는 초기 위치+v*t+(0,0,-g*t²/2)다. Euler 적분 때문에 누적 오차가 남는다. max_precontact_analytic_position_error_m는 첫 힘이 실린 접촉 전 구간만 포함한다. 기존 max_analytic_position_error_m는 접촉 이후까지 포함하므로 충돌 반발 이후의 큰 값은 analytic free-flight 예측과의 차이이며 물리 실패로 읽으면 안 된다.
- 바닥 첫 peak는 접촉 후 첫 상승/하강 전환 때 공 아래쪽 높이로 측정한다. 유효 restitution은 첫 impact 이전/첫 floor 이탈 vz 비율이고 시간해상도 영향을 받는다.
- 통과 event는 아래 방향 림 plane crossing에서 구멍 clearance를 검사하고, sphere plane 단면이 hole 안에 유지되는지 확인한 뒤 공 전체가 림 아래에 나오면 latch한다. 위 방향 crossing·옆으로 빗나간 ball은 테스트에서 제외되는 것을 확인했다. RL reward/공식 shot scoring 시스템은 아니다.
- 처음에는 보드 충돌 뒤 다른 림에 부딪혀 final vx가 다시 바뀌는 복합 시나리오를 관찰했다. 최종 보드 시험은 y=0.55m에서 진행해 림 접촉을 분리했다. 이를 성공한 단일 보드 반발로 오해하지 않도록 별도 조건으로 정리했다.

## 6. 실행 명령어

터미널 위치:

```bash
cd "/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim"
```

전체 물리 검사 및 XML/CSV 생성:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/m3_mpl_cache \
../../../basketball-r1/.venv/bin/python examples/m4_basketball.py --check
```

전체 시험의 GIF까지 저장(화면 창 없이 MuJoCo renderer 사용):

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/m3_mpl_cache \
../../../basketball-r1/.venv/bin/mjpython examples/m4_basketball.py --video --check
```

GUI: Humanoid+공+골대, 테스트 발사와 통과 궤적, 4초 동안 fitted motion 및 공 비행:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/m3_mpl_cache \
../../../basketball-r1/.venv/bin/mjpython examples/m4_basketball.py \
  --case launch --duration 4 --gui --video --check \
  --output outputs/m4_basketball/gui
```

초기 위치/속도 변경(테스트용 API):

```bash
PYTHONDONTWRITEBYTECODE=1 ../../../basketball-r1/.venv/bin/python \
  examples/m4_basketball.py --case freefall \
  --position 2 -0.5 2.2 --velocity 1.5 0.4 0.6 --duration 0.3 \
  --check --output outputs/m4_basketball/custom_initial
```

`--case rim --gui --video`, `--case pass --gui --video`, `--case backboard --gui --video`로 각 장면을 선택할 수 있다. --position/--velocity 단위는 m와 m/s이고 단일 case에만 적용한다. 기본 시나리오에 맞춘 --check를 임의 시나리오에 적용하면 의도한 조건과 달라 실패할 수 있다. --output은 기본으로 outputs/m4_basketball이며 단일 실행의 summary.csv에는 선택한 case만 들어간다.
GUI는 실행이 끝나면 닫힌다. 이 파일에는 무한 반복 옵션이 없다. 저장한 GIF는 브라우저에서 반복 재생할 수 있다. GUI를 먼저 닫으면 전체 검증 시간이 부족하므로 완료로 세면 안 된다. macOS mjpython이 내부 예외에도 exit code 0을 보인 경우가 있어 log의 완료 메시지 및 최신 결과 파일도 확인했다.

## 7. 시각화와 결과 파일

![Humanoid, 농구공, 백보드와 림 및 실제 비행 궤적](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/outputs/m4_basketball/gui/scene_preview.png)

- [4초 GUI 비행/통과 GIF](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/outputs/m4_basketball/gui/launch.gif)
- [림 충돌 GIF](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/outputs/m4_basketball/rim.gif) / [백보드 충돌 GIF](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/outputs/m4_basketball/backboard.gif) / [중앙 통과 GIF](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/outputs/m4_basketball/pass.gif)
- [물리 시험 비교 그래프](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/outputs/m4_basketball/ball_physics_comparison.png)
- 노란 점은 실제 공 위치 이력이며 scene 전용이고 물리에 영향을 주지 않는다. 초기 속도는 TEST launcher에서 지정하며 손의 이동에서 얻은 것이 아니다. GUI launch/sync 완료 및 저장 render 이미지 확인을 수행했으며 foreground screen 캡처는 하지 않았다.
- MP4/FFmpeg에 의존하지 않고 PIL로 GIF를 저장한다. offscreen 버퍼는 960×540이고 물리 상태를 30Hz로 렌더한다. GIF 시간 precision에 맞춰 30/30/40ms 간격을 번갈아 사용한다. 마지막 endpoint frame 때문에 GIF 길이는 물리 구간보다 최대 한 프레임 길다.
- outputs: body_01_basketball.xml, summary.csv, case별 timeseries.csv/contacts.csv/summary.json/states.npz/GIF, run.log, tests.log, synthetic_regression.log, custom_initial/, gui/, preservation audit, comparison graph를 저장했다. states.npz는 integrated qpos preview 자료이고 기존 SMPL Motion NPZ schema와 구분한다.

## 8. 기존 기능 회귀 및 보존

- 단위 테스트 22/22 통과: 기존 16개 + 신규 M4 6개(구조/관성/actuator 보존, collision filtering, 초기 조건/spin, passage 방향/clearance, oblique passage, 4초 controller parity).
- 기존 합성 모션의 네 체형 --all --check 회귀도 통과: 기존 source/PD/COM 사용, 낙상/경고 없음.
- M4 기존 Python 125개 hash 변화 없음. 원본 환경 및 raw dataset hash 변화 없음, input fitted NPZ SHA-256는 기존 M3와 동일하다.
- 기존 Python SyntaxWarning/Torch/SMPL 안내는 log에 남으며 MuJoCo warning 카운터와 구분했다.
- Git Commit/Push/Branch 생성/Pull Request/Issue 수정은 수행하지 않았다. 결과는 기존 Git-ignore outputs 아래에 보관한다.

회귀 명령:

```bash
PYTHONDONTWRITEBYTECODE=1 ../../../basketball-r1/.venv/bin/python \
  examples/m3_free_throw.py --all --check \
  --models ../../../basketball-r1/SMPLSim/outputs/smplh_physics \
  --output outputs/m4_basketball/synthetic_regression
```

전체 unit test는 기존 test_smplh_motion.ROOT만 테스트 프로세스에서 원본 XML 경로로 설정했다(파일 변경 없음):

```bash
PYTHONDONTWRITEBYTECODE=1 ../../../basketball-r1/.venv/bin/python - <<'PYTEST'
from pathlib import Path
import sys, unittest
root = Path.cwd()
sys.path.insert(0, str(root / "tests"))
import test_smplh_motion
test_smplh_motion.ROOT = root.parents[2] / "basketball-r1/SMPLSim"
result = unittest.TextTestRunner(verbosity=2).run(
    unittest.defaultTestLoader.discover(str(root / "tests")))
raise SystemExit(not result.wasSuccessful())
PYTEST
```

## 9. 변경 파일

- 신규 [smpl_sim/physics/basketball.py](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/smpl_sim/physics/basketball.py): 파생 XML builder, 테스트 초기 조건 API, 기존 제어기 bridge, geometric passage, 시험 시나리오.
- 신규 [smpl_sim/physics/__init__.py](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/smpl_sim/physics/__init__.py): 물리 확장 모듈.
- 신규 [examples/m4_basketball.py](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/examples/m4_basketball.py): physics tests/GUI/GIF/CSV CLI.
- 신규 [tests/test_basketball.py](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/tests/test_basketball.py): 통합·회귀 검사.
- 신규 [M4_1_BASKETBALL_ENVIRONMENT.md](/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/M4_1_BASKETBALL_ENVIRONMENT.md): 본 문서.
- 기존 M1~M3 소스 파일은 변경하지 않았다. 파생 scene XML만 outputs에 생성했다.

## 10. M4-2 손-공 연결 계획 (미구현)

1. 현재 body-only Fitting에서 wrist/finger rotation의 실제 릴리스 정답이 부족함을 확인하고 손 위치·grasp frame·ball contact points를 별도로 정의한다.
2. 단계별 hand-ball collision filtering을 설계한다. 현재 전체 human-ball 충돌 비활성은 환경 검증용이며 M4-2에서는 손에만 선택적으로 허용하고 불필요한 몸통/다리 충돌을 제어한다.
3. Hold→release 상태 전이를 명시하고 임시 제약/접촉 기반 grasp를 비교한다. 손과 공의 제약이 있는 동안 접촉/외력 coupling을 반영할 integrated controller adapter를 설계해야 하며 M4-1의 isolated shadow bridge를 그대로 사용하면 안 된다.
4. 실제 릴리스 API는 손의 material-point/Jacobian 속도와 회전에서 공의 world linear 및 적절한 frame angular velocity를 계산한다. 현재 test_ball_initial_state의 임의 launch 값을 학습용 실제 release 경로로 연결하지 않는다.
5. Release 시 constraint 제거/impulse·에너지 연속성, 공 궤적·hand contact·rim/board 충돌을 검증한다. 속도/각도/timestep sweep 및 ground/rim rebound 파라미터 민감도를 평가한다.
6. 위 물리/연결 검증 후에만 별도 목표로 shot detector/학습 환경을 설계한다. 이번 단계에 PPO/reward를 추가하지 않았다.

Issue에 직접 체크할 완료 항목:

- [x] 독립 sphere/freejoint 농구공과 static 백보드/구멍 있는 림
- [x] 기준·단위·배치 및 마찰/반발 파라미터 문서화
- [x] 기존 body_01 관절/actuator/PD+COM/NPZ 보존
- [x] 자유낙하·바닥·백보드·림·통과·초기 상태 변경 검증
- [x] GUI 비행 궤적 및 GIF 저장
- [x] 22개 unit test 및 네 체형 합성 모션 회귀 통과
- [ ] 손-공 hold/release 및 coupling-aware 통합: M4-2
- [ ] 실제 손목/손가락 릴리스 정답·공기저항/속도별 접촉 수렴 검증: 이후 과제
