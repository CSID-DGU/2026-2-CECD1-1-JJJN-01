# M3-1 기존 합성 자유투 검증 결과

검증일: 2026-10-10. 팀 저장소 `mujoco/SMPLSim`에서 기존 구현을 실행했다. 이 데이터는 물리 시뮬레이션/모션 인터페이스 검증용 합성 데이터이며 실제 농구 자유투 정답 데이터가 아니다.

## 실행 환경 및 변경 범위

- macOS 27.0.1, Apple Silicon arm64, Python 3.12.13, MuJoCo 3.15.0.
- 기존 브랜치 `dev/mincheol`; 검증 시 HEAD `a631b8ac48458bd371b38db9653540a988e82888`.
- 실행한 Python 모듈은 팀 저장소의 `smpl_sim/motion/smplh_motion.py`임을 `environment.json`으로 확인했다.
- 런타임은 기존 `basketball-r1/.venv`, XML은 `basketball-r1/SMPLSim/outputs/smplh_physics/body_01.xml`~`body_04.xml`을 읽기 전용으로 참조했다. 모델 파일과 가상환경은 복사하지 않았다. 기존 생성 XML은 이 실행에서 SMPL+H pkl을 다시 읽을 필요가 없다.
- 기존 Python 소스, 테스트, 문서는 수정하지 않았다. 신규 파일은 이 검증 문서와 `outputs/m3_1_verification/`, GUI 검증 실행 산출물뿐이다. Git의 branch/commit/push/PR/Issue 수정은 수행하지 않았다.
- 원본 393개 파일과 팀 기존 Python 116개 파일을 해시 점검했다. 팀 Python은 전부 불변이며 원본에서 변경 감지된 파일은 macOS 메타데이터 `.DS_Store` 1개뿐이다(변경 원인은 확인하지 않음). 원본 코드·XML·모델 내용에는 차이가 없다. 상세 목록은 `preservation_check.json`에 기록했다.
- `outputs/*`는 기존 `.gitignore` 규칙으로 제외되므로 CSV/NPZ/로그는 로컬 산출물이다. 공유가 필요하면 사용자가 별도로 관리해야 한다.

## 기존 구현 분석

| 파일 | 현재 동작 |
|---|---|
| `smpl_sim/motion/synthetic_free_throw.py` | 고정된 관절별 intrinsic XYZ Euler 궤적을 quintic smoothstep으로 생성한 뒤 axis-angle로 변환한다. |
| `examples/m3_free_throw.py` | 공통 NPZ 생성/로드, 체형별 바닥 높이 정렬, 기존 M3 제어 실행, 전체/단계/관절별 CSV, 임계값 검사와 출처 기록을 수행한다. |
| `tests/test_synthetic_free_throw.py` | NPZ 계약, 동작 순서/각도, smoothstep 경계 및 pose 프레임 점프를 검사하는 3개 테스트다. |
| `M3_1_SYNTHETIC_FREE_THROW.md` | 기존 구현 설명 및 원본 개발 디렉터리용 명령어를 담는다. 팀 디렉터리에서 실행할 명령은 아래를 사용한다. |

NPZ는 `schema_version=1`, `poses=(91,156)`, `root_position=(91,3)`, `fps=30`, `coordinate_system=smpl`, `translation_kind=root_position`이다. 52개 관절 각각의 3차원 axis-angle을 rad 단위로 담는다. 0초와 3초를 포함한 91개 샘플이며 길이는 `(91-1)/30=3초`다.

| 단계 | 시간 (초) | 동작 |
|---|---:|---|
| stand | 0~0.35 | 초기 서기 자세 |
| knee_bend | 0.35~0.85 | 무릎 x 0.10→약 0.35 rad, 고관절/발목을 함께 굽힘 |
| knee_extend | 0.85~1.30 | 무릎을 0.10 rad로 복귀 |
| arm_raise | 1.30~1.95 | 어깨 x 0→-1.15 rad, 팔꿈치 y를 좌우 ±0.90 rad로 굽힘 |
| elbow_extend | 1.95~2.50 | 팔꿈치를 좌우 ±0.10 rad로 펴고 손목 x를 -0.30 rad로 준비 |
| wrist_flick | 2.50~3.00 | 손목 x -0.30→0.20→0.10 rad |

`u³(10−15u+6u²)`의 원래 연속 함수는 각 전환 경계의 속도와 가속도가 0이다. NPZ를 읽어 실행할 때는 아래의 선형 hinge 보간을 사용하므로 실행 목표 궤적 전체가 C2 연속인 것은 아니다.

Root 위치와 회전 필드는 존재한다. 이 합성 소스에서는 root position이 고정이고 root axis-angle은 항상 0이다. 체형별로 최초 바닥 정렬에 필요한 일정한 높이 offset만 더한다. 물리 root는 free joint이며 액추에이터가 없어 위치/자세 궤적을 직접 추종하지 않는다. 시뮬레이션 중 root 변위는 물리와 관절 토크에 의해 발생한다.

`examples.m3_control.run`, `BalancedPD`, `StablePDController`, COM 피드백 규칙을 그대로 재사용한다. 체형별 질량·다리 길이에 따른 기존 gain scaling도 유지한다. 외력 지지, 중간 qpos 재설정, PPO는 없다. 양쪽 팔은 대칭적인 관절 동작이며 슈팅 손/보조 손의 실제 역할 구분은 없다.

## 관절/좌표계 및 보간 검증

모든 체형은 52개 인체 body + world, root free joint 1개 + hinge 153개 = 관절 154개, 액추에이터 153개, `nq=160`, `nv=159`다. 각 non-root SMPL+H 관절은 이름으로 해당 MuJoCo `*_x`, `*_y`, `*_z`에 대응한다. axis-angle의 각 성분을 hinge angle에 직접 넣는 것이 아니라 회전행렬을 거쳐 intrinsic XYZ로 분해한다. 30개 손가락 body의 90개 hinge도 물리 모델에 포함되지만 이번 합성 입력의 손가락 목표는 모두 0이다.

- SMPL 좌표는 Y-up, MuJoCo world는 Z-up이다. 기존 root 회전 C는 `(x,y,z)→(z,x,y)`를 적용한다. 전역 root 회전은 `C · R_root`이고 non-root 관절의 부모 기준 회전은 유지한다.
- `qpos[:3]`는 world root position(m), `qpos[3:7]`은 quaternion **wxyz**다. SciPy의 xyzw 순서를 명시적으로 바꾼다. root axis-angle=0일 때 world root quaternion은 `(0.5,0.5,0.5,0.5)`이다.
- `root_position`은 골반 위치 그 자체다. `smpl_transl` 입력은 shaped pelvis offset을 추가하는 별도 규약이며 기존 테스트에서 확인됐다.
- hinge angle은 시간축 unwrap 후 선형 보간, root position도 선형 보간, root quaternion은 SLERP한다. 구간 밖의 시간은 양 끝으로 clamp한다.
- 범위 초과, nonfinite 및 XYZ gimbal lock을 거부한다. 실제 영상 모션의 연속 Euler 분기/특이점 처리는 현재 준비되어 있지 않다.

네 체형×91프레임×51개 non-root 관절의 `mj_forward` 부모 기준 회전을 원본 axis-angle 회전행렬과 비교한 최대 성분 오차는 **1.11e-15**다. 기존 회귀 테스트에서는 root의 비영 회전과 compound shoulder/knee/finger 회전도 검증한다.

| 수치 검증 | 결과 |
|---|---:|
| 최대 원본 axis-angle 성분 프레임 차이 | 0.122438 rad |
| 최대 hinge 30Hz 프레임 차이 | 0.122438 rad |
| 450Hz 샘플 최대 hinge 차이 | 0.008163 rad |
| 중간 프레임 선형 보간 오차 | 8.33e-16 rad |
| root quaternion norm 오차 | 0 |
| 원본 root position/rotation peak-to-peak | 0 m / 0 rad |
| 선형 보간 구간 사이 최대 hinge 속도 점프 | **1.450667 rad/s** |

위 결과는 위치 연속성(C0)을 확인한다. 30FPS 선형 보간에서 속도 연속성(C1), 가속도 연속성(C2)은 확보되지 않는다. 기존 smoothstep 경계 테스트만으로 실제 실행 목표의 C2 연속성을 주장하면 안 된다.

## 실제 물리 실행 결과

비GUI `--all --check`가 네 체형에서 통과했다. 체형당 450Hz 물리 1350 step, 30Hz 지표 90 sample, 총 3초를 실행했다. 지표는 1/30초부터 3초까지이며 warmup 구간을 버리지 않았다. 동일 source pose와 일정한 root offset, 원본 NPZ SHA-256 불변을 기존 실행 코드가 검사했다.

Body RMSE는 손가락과 root를 제외한 63 hinge, Task RMSE는 양쪽 Hip_x/Knee_x/Shoulder_x/Elbow_y/Wrist_x의 10 hinge 기준이다. 오차는 COM 보정 전 공통 원본 목표와 실제 관절각의 차이다. 153 hinge 전체와 보정 후 목표 오차는 `all_joint_summary.csv`에 별도로 집계했다.

| 체형 | Body RMSE rad | Task RMSE rad | 균형 % | 양발 접촉 % | 최대 골반 tilt ° | 발목 이동 mm | 낙상 |
|---|---:|---:|---:|---:|---:|---:|---:|
| body_01 | 0.032432 | 0.073176 | 100 | 100 | 3.329 | 2.209 | 0 |
| body_02 | 0.032039 | 0.073555 | 100 | 100 | 1.668 | 1.552 | 0 |
| body_03 | 0.034405 | 0.072862 | 100 | 100 | 5.441 | 4.090 | 0 |
| body_04 | 0.030932 | 0.072317 | 100 | 100 | 1.700 | 2.166 | 0 |

COM 이동은 첫 기록 시점(1/30초) 대비 최대 수평 변위다. 지지 여유는 접촉 중인 발 box의 투영 hull에 대한 COM 투영의 최소 내측 거리다.

| 체형 | COM 최대 이동 mm | 최소 지지 여유 mm | 좌/우 최소 접촉력 N | 최대 root 높이 목표 오차 mm | 초기 root 높이 offset mm |
|---|---:|---:|---:|---:|---:|
| body_01 | 21.497 | 103.322 | 107.22 / 346.48 | 30.053 | 0.000 |
| body_02 | 19.247 | 94.777 | 278.78 / 270.16 | 25.739 | -91.973 |
| body_03 | 24.937 | 49.652 | 25.96 / 422.62 | 34.631 | 92.252 |
| body_04 | 9.177 | 101.405 | 284.62 / 180.26 | 27.397 | 35.866 |

관절별 RMSE (좌/우, rad):

| 체형 | Hip_x | Knee_x | Shoulder_x | Elbow_y | Wrist_x |
|---|---:|---:|---:|---:|---:|
| body_01 | 0.02335 / 0.02043 | 0.06499 / 0.05993 | 0.10132 / 0.10128 | 0.09574 / 0.09580 | 0.05432 / 0.05433 |
| body_02 | 0.02408 / 0.02040 | 0.06567 / 0.05958 | 0.10248 / 0.10242 | 0.09574 / 0.09579 | 0.05437 / 0.05437 |
| body_03 | 0.02347 / 0.02027 | 0.06444 / 0.06010 | 0.10034 / 0.10028 | 0.09572 / 0.09580 | 0.05430 / 0.05430 |
| body_04 | 0.02636 / 0.02253 | 0.05859 / 0.05259 | 0.10149 / 0.10152 | 0.09586 / 0.09590 | 0.05433 / 0.05433 |

모든 체형의 토크 포화 관절 비율은 기록 시점에서 0, MuJoCo 물리 경고는 0이다. Python의 기존 docstring escape 관련 SyntaxWarning과 torch.jit FutureWarning은 발생했다. 이번 검증 범위에서는 해당 소스를 수정하지 않았다.

body_03은 몸통 RMSE·골반 기울기·발목 이동이 가장 크며 최소 지지 여유가 가장 작다. body_02의 task RMSE가 가장 크지만 체형 간 차이는 약 0.00124 rad로 작다. 손목 높이 증가량은 체형 순서대로 약 0.449/0.405/0.492/0.469 m여서 동일 관절각이 동일한 손끝 공간 궤적을 보장하지 않는다. 실제 자유투 retargeting에서 endpoint·release constraint가 추가로 필요하다.

## 지표 해석의 한계

- 접촉은 해당 발목/발가락 floor 접촉 법선력 합이 1N을 넘는지로 판단한다. 관측 시점마다 양발 모두 접촉했다.
- `foot_drift_m`는 초기 발목 body 위치 대비 XY 변위의 양발 최대값이다. 접촉점 상대 접선 속도, 발바닥 회전, 발별 미끄러짐은 기록하지 않는다. 이번 결과는 작은 발목 drift를 확인했으며 물리적 무미끄러짐을 증명하지 않는다.
- 균형 판정은 낙상이 아니고, 골반 tilt<25°, 한 발 이상 접촉, 지지 여유≥-0.02m다. 낙상은 root 높이<초기 65%, tilt>35°, 또는 발 이외 body의 유효 floor 접촉이다.
- 낙상·접촉·균형은 30Hz 기록값이며 샘플 사이 사건을 놓칠 수 있다. nonfinite 상태는 매 450Hz 물리 step에서 검사한다.
- 지지 hull은 발 box의 투영이며 CoP/ZMP 또는 외란 안정성 검증이 아니다. 이번 검증은 단일 3초 deterministic rollout으로 국한된다.
- root 궤적 추종, 실제 농구공/골대/릴리스/슛 성공률, 손가락 그립, 비대칭 shooting/support hand, 실제 영상 모션 추출은 구현되지 않았다.

## 테스트와 실행 명령

아래 모든 명령은 **팀 저장소 `mujoco/SMPLSim`에서** 실행한다. 기존 문서의 `../.venv/bin/...` 경로는 팀 디렉터리에서는 사용할 수 없다.

```bash
cd "/Users/jomincheol/Desktop/02_Project/2학기/종합설계/MuJoCo/2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim"
export PYTHONDONTWRITEBYTECODE=1
LOCAL_M3_ENV="../../../basketball-r1"

# 네 체형 비GUI 실행 + 기존 임계값 검사
"$LOCAL_M3_ENV/.venv/bin/python" examples/m3_free_throw.py --all --check \
  --models "$LOCAL_M3_ENV/SMPLSim/outputs/smplh_physics" \
  --output outputs/m3_1_verification

# 동일 source로 GUI 실행 (이번 검증에서의 오류 여부는 아래 GUI 결과 참고)
"$LOCAL_M3_ENV/.venv/bin/mjpython" examples/m3_free_throw.py --all --gui --check \
  --models "$LOCAL_M3_ENV/SMPLSim/outputs/smplh_physics" \
  --source outputs/m3_1_verification/synthetic_free_throw.npz \
  --output outputs/m3_1_verification_gui
```

기존 unittest가 XML을 팀 `outputs/smplh_physics`에 고정해서 읽으므로 제외된 XML만 기존 환경으로 리다이렉트했다. 테스트/실행 코드는 팀 저장소에서 import하며 파일 수정이나 symlink 생성은 하지 않았다. 아래와 같이 재현할 수 있다.

```bash
"$LOCAL_M3_ENV/.venv/bin/python" - <<'PYTEST'
from pathlib import Path
import sys, unittest
team = Path.cwd()
sys.path.insert(0, str(team / 'tests'))
import test_smplh_motion
test_smplh_motion.ROOT = team.parents[2] / 'basketball-r1/SMPLSim'
suite = unittest.defaultTestLoader.discover(str(team / 'tests'))
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(not result.wasSuccessful())
PYTEST
```

**8/8 통과**: 기존 M3-0 회귀 5개(4체형 FK/이름, 공통 모션, translation 규약, 보간/정규화/clamp, 잘못된 입력/gimbal 거부) + 합성 모션 3개(NPZ 계약, 순서/각도, 경계/프레임 연속성).

## GUI 검증 결과

최초 `mjpython examples/m3_free_throw.py --all --gui --check` 실행은 첫 자식 `body_01`에서 **SIGSEGV(11)**로 종료했다. 바닥 정렬 reference와 joint table까지만 저장했으며 timeseries는 생성되지 않았다. 로그는 `outputs/m3_1_verification/gui.log`에 보존했다. 원인은 확정하지 않았고 코드 수정도 하지 않았다.

이후 같은 팀 소스·공통 NPZ·XML·런타임으로 체형을 각각 `--body body_0i --gui --check`로 직접 실행했다. **네 체형 모두 종료 코드 0, 3초 완료, 기존 check 통과**했다. `gui_parity.csv`에서 비GUI 결과와 모든 숫자 summary 항목의 최대 차이는 네 체형 모두 0이었다. GUI 창을 화면 캡처로 시각 검수한 결과는 포함하지 않는다.

`--all --gui`의 일괄 실행 경로 재현성은 추가 확인이 필요하다. 현재 확인된 개별 GUI 실행 명령은 아래와 같다(위 실행 위치와 환경 변수 설정을 유지한다).

```bash
for body in body_01 body_02 body_03 body_04; do
  "$LOCAL_M3_ENV/.venv/bin/mjpython" examples/m3_free_throw.py --body "$body" --gui --check \
    --models "$LOCAL_M3_ENV/SMPLSim/outputs/smplh_physics" \
    --source outputs/m3_1_verification/synthetic_free_throw.npz \
    --output outputs/m3_1_verification_gui_direct
done
```

개별 실행의 통합 `summary.csv`는 마지막 체형 결과로 덮어써진다. 각 체형 `*_summary.csv`는 모두 유지된다. GUI 비교 결과는 `outputs/m3_1_verification/gui_parity.csv`, 개별 GUI CSV는 `outputs/m3_1_verification_gui_direct/`에 있다. 최초 실패 산출물은 `outputs/m3_1_verification_gui/`에 별도 보존했다.

## 저장된 파일 및 완료 항목

- 새 검증 문서: `M3_1_VERIFICATION.md`.
- `outputs/m3_1_verification/summary.csv`, `phase_summary.csv`, `task_joint_summary.csv`: 기존 코드의 전체/6단계/10개 task 관절 요약.
- `all_joint_summary.csv`: 기록된 153 hinge×4체형 = 612행의 관절별 집계(각 90샘플).
- `balance_contact_audit.csv`: COM·최소 접촉력·지지 여유·발목 drift·root 오차 집계.
- `motion_conversion_audit.csv`: FK·보간·root·동일 source 수치 검증.
- 체형별 `*_timeseries.csv`, `*_joint_tracking.csv`, `*_joints.csv`, `*_reference.npz`, `*_summary.csv`, `*_phase_summary.csv`, `*_task_joint_summary.csv`.
- 공통 `synthetic_free_throw.npz`, `reference_metadata.json`, `environment.json`, `preservation_check.json`, `headless.log`, `tests.log`, `REPORT.md`; GUI 실행 로그/산출물은 별도 보관한다.

Issue에서 체크 가능한 항목(실제 Issue는 수정하지 않음):

- [x] 기존 4개 구현 파일 분석 및 M3-0 PD + COM 재사용 확인
- [x] 3초/30FPS/91프레임 SMPL+H axis-angle 공통 NPZ 확인
- [x] 네 체형 동일 관절 모션·일정한 바닥 offset·source 불변 검증
- [x] 네 체형 비GUI 연속 물리 실행 및 기존 check 통과
- [x] 네 체형 개별 GUI 3초 실행·check 통과·비GUI 정량 결과 동일
- [ ] `--all --gui` 최초 SIGSEGV 원인 확정 및 일괄 실행 재현성 확인
- [x] 관절별 오차/COM/양발 접촉/발목 drift/낙상 CSV 저장
- [x] 관절 대응·root 규약·보간·위치 연속성 검증 및 테스트 8개 통과
- [x] 원본 환경/기존 코드 보존, 새 기능/PPO/Git 변경 작업 없음
- [ ] 실제 영상 기반 기준 자유투 모션 확보
- [ ] 움직이는 root 궤적과 체형별 공간 endpoint retargeting
- [ ] 접촉점 속도 기반 발별 미끄러짐·물리 step별 낙상 판정
- [ ] 속도/가속도 연속 보간 및 실제 모션 Euler 특이점 처리
- [ ] 공/손가락 그립/릴리스/슛 성공률 검증

다음 단계는 실제 영상/모션 데이터의 좌표·관절 순서·단위·root translation 규약을 먼저 확정하고 이 NPZ 계약으로 변환하는 것이다. 그 전에 GUI 재현성과 모델 경로 설정을 정리하고 보간/접촉 지표의 한계를 해결해야 한다. 이번 작업에서는 새 기능 구현을 진행하지 않았다.
