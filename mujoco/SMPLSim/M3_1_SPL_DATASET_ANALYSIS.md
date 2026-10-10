# M3-1B SPL 실제 자유투 데이터셋 분석 및 변환 설계

분석일: 2026-10-10. **분석과 설계만 수행했다. SMPL+H fitting, 데이터 변환, 시뮬레이션, PPO는 이번 단계에서 구현/실행하지 않았다.**

## 1. 결론과 조사 범위

로컬 SPL 데이터는 실제 markerless capture 기반 자유투 **5명·583시도·2세션·139,920프레임**이다. 관절의 3D 위치와 일부 공 추적 데이터를 제공하며 SMPL+H pose/회전 데이터 자체는 제공하지 않는다. 몸통/사지 모션은 3D fitting으로 근사 복원할 수 있다. 2025년 데이터는 상세 손가락 위치도 제공하므로 hand fitting을 시도할 수 있지만 전체 회전의 유일한 복원이나 공의 실제 release 정답을 보장하지 않는다.

첫 기준 모션은 **2025-12-18 세션의 품질을 점검한 단일 시도**로 시작하는 것이 적절하다. 2024년은 상세 손가락 데이터가 없어 hand fitting 기준으로 사용할 수 없다. `made`는 슛 성공 라벨이며 좋은 자세나 기준 기술 동작이라는 인증은 아니다.

조사 대상은 원본 `SPL-Open-Data/`의 Git 내부 파일을 제외한 전체 594개 파일(1,333,840,834 bytes, 약 1.33 GB)이다. JSON 583개를 모두 파싱하여 각 240프레임의 키, finite 여부, 시간 간격, 관절 수 및 공 가용성을 점검했다. 데이터셋 체크아웃 HEAD는 `a3f9cffbde917b1e1747cedd6ec25dfab18c6051`이다. 공식 웹 문서 및 라이선스도 확인했다. 아래 실측 통계는 이 로컬 체크아웃을 기준으로 한다.

## 2. 실제 디렉터리와 데이터 수

```text
MuJoCo/
├── SPL-Open-Data/                         # 원본, 읽기 전용 조사
│   ├── .git/                             # 분석 통계에서 제외
│   ├── .gitignore
│   ├── LICENSE
│   ├── README.md
│   ├── assets/mlse-banner.jpeg
│   └── basketball/freethrow/
│       ├── README.md
│       ├── animate.py
│       ├── usage.ipynb
│       ├── assets/
│       │   ├── coordinate_system.png
│       │   ├── ball_coords.png
│       │   ├── markers.png
│       │   └── shot_animation.gif
│       └── data/
│           ├── 2024-08-28/P0001/BB_FT_P0001_T0001.json ... T0125.json
│           └── 2025-12-18/
│               ├── P0001/BB_FT_P0001_T0001.json ... T0088.json
│               ├── P0002/BB_FT_P0002_T0001.json ... T0093.json
│               ├── P0003/BB_FT_P0003_T0001.json ... T0090.json
│               ├── P0004/BB_FT_P0004_T0001.json ... T0089.json
│               └── P0005/BB_FT_P0005_T0001.json ... T0098.json
├── basketball-r1/SMPLSim/                 # 기존 개발 환경, 수정하지 않음
└── 2026-2-CECD1-1-JJJN-01/mujoco/SMPLSim/ # 문서 작성 위치
```

전체 파일 구성: JSON 583, Markdown 2, Python 1, notebook 1, PNG 3, GIF 1, JPEG 1, 확장자 없는 LICENSE/.gitignore 2. 영상 원본, 카메라 calibration, participant_information.json, 참가자 신장·질량·성별·손잡이 정보는 이 체크아웃에 없다. 최상위 README의 일반 예시 트리와 실제 구조가 다르므로 실제 `data/session/participant` 경로를 사용해야 한다.

| 세션 | 참가자 | 시도 | made / missed | FPS | 프레임/시도 | 마지막 시각 |
|---|---|---:|---:|---:|---:|---:|
| 2024-08-28 | P0001 | 125 | 88 / 37 | 30 | 240 | 7.966 s |
| 2025-12-18 | P0001 | 88 | 61 / 27 | 60 | 240 | 3.983 s |
| 2025-12-18 | P0002 | 93 | 55 / 38 | 60 | 240 | 3.983 s |
| 2025-12-18 | P0003 | 90 | 81 / 9 | 60 | 240 | 3.983 s |
| 2025-12-18 | P0004 | 89 | 61 / 28 | 60 | 240 | 3.983 s |
| 2025-12-18 | P0005 | 98 | 43 / 55 | 60 | 240 | 3.983 s |
| 합계 | 고유 5명, 세션별 참가 6건 | 583 | 389 / 194 | 혼합 | 139,920 전체 | 혼합 |

P0001은 두 세션에 모두 있다. 공식 문서상 동일 participant ID는 세션 간 동일인이다. 따라서 trial의 고유 키는 **(session, participant_id, trial_id)**여야 한다. 파일명만으로 합치면 P0001의 서로 다른 세션 시도가 충돌한다.

## 3. 데이터 스키마와 시간

전 JSON의 최상위 키와 프레임 키는 동일했다. player 키 69개도 전체 프레임에서 동일했지만 키가 존재하는 것과 값이 유효한 것은 다르다.

| 경로 | 형식 | 의미/단위 |
|---|---|---|
| sampling_rate | integer | 30 또는 60 FPS |
| trial_date | string | 세션 날짜 YYYY-MM-DD |
| participant_id / trial_id | string | P0001 등 / T0001 등 |
| result | string | made 또는 missed |
| landing_x, landing_y | number | 골대 10ft 높이 평면의 이론적 공 통과 위치, 골대 앞 기준 별도 좌표계, inch |
| entry_angle | number | 수평 대비 공 진입각, degree |
| tracking | array[240] | 프레임 시퀀스 |
| tracking[i].frame | integer | 0~239의 연속 프레임 ID |
| tracking[i].time | integer | trial 상대 시간, **밀리초로 해석됨** |
| tracking[i].data.player | dict[69] | 이름→[x,y,z], feet; null/NaN 가능 |
| tracking[i].data.ball | array[3] | 공 중심 [x,y,z], feet; NaN 가능 |

`time`의 단위는 명시적 schema 필드가 아니라 실측 검증으로 확인했다. 모든 trial에서 0부터 시작하고 엄격히 증가하며 `frame*1000/sampling_rate`와 최대 0.667ms 이내로 일치한다. 30FPS 간격은 33/34ms, 60FPS 간격은 16/17ms다. nominal 마지막 시각은 239/30=7.966667s, 239/60=3.983333s이고 저장 정수 시간은 각각 7966/3983ms다. 전체 frame ID에 누락/중복은 없다.

향후 loader는 원본 time을 보관하되, 이 데이터에서는 검증된 `frame/sampling_rate`의 등간격 시간을 사용해 정수 ms 양자화로 인한 불필요한 속도 jitter를 피할 수 있다. 서로 다른 FPS를 혼합하지 않는다. 같은 240프레임이 같은 길이를 의미하지 않는다.

예시 파일은 `2025-12-18/P0001/BB_FT_P0001_T0001.json`이다. frame=0, time=0의 LEFT_WRIST는 `[19.6822115409, -25.5693322530, 3.8947203354]` ft이다. frame별 회전, quaternion, 관절각, confidence, 카메라 정보, 접촉력, COM, 명시적 release frame/phase 라벨은 없다. COM을 MID_HIP와 동일시하면 안 된다.

### 결측 및 품질 전수 검사

| 항목 | 2024-08-28 | 2025-12-18 |
|---|---:|---:|
| 프레임 | 30,000 | 109,920 |
| 관절 키/프레임 | 69 | 69 |
| 기본적으로 관측되는 keypoint | 27 | 69 |
| 상세 손가락 40개 | 전 프레임 null | 관측 가능 |
| MID_HIP, NECK | 전 프레임 null | 관측 가능 |
| 인체 전체 결측 프레임 | 기본 27개는 모두 유효 | 267 (모든 69개 동시 결측) |
| 부분 인체 결측 프레임 | 상세 42개 상시 미제공 | 0 |
| 전체 인체 관측이 완전한 trial | 상세 미제공으로 0/125 | 434/458 |
| 인체 결측이 있는 trial | 상세 42개 상시 미제공 | 24/458 |
| 유효한 공 중심 프레임 | 16,297 (54.32%) | 109,297 (99.43%) |
| trial별 공 finite 비율 | 45.83~80.83% | 86.67~100% |

2024년의 유효한 27개는 얼굴 5 + 어깨/팔꿈치/손목/고관절/무릎/발목 좌우 12 + 발 landmark 6 + coarse THUMB/PINKY 좌우 4이다. 상세 손가락 40개와 MID_HIP/NECK 2개, 총 42개는 모든 프레임에서 `[null,null,null]`이다. 2025년의 267개 결측 프레임은 동일 시점의 모든 player point가 invalid하며, 이는 18,423개의 invalid 3D vector이다. 결측을 0좌표로 채우거나 긴 구간을 무조건 보간하면 안 된다.

공 결측에 bare `NaN`이 실제 JSON에 들어 있다. 전 파일에서 NaN/Infinity 토큰 합계는 41,109개였고 Python json은 이를 허용하지만 strict JSON parser는 실패할 수 있다. 미래 importer에서 null/NaN을 명시적으로 마스크해야 한다. 원본 파일은 정규화해 덮어쓰지 않는다.

Finite 데이터도 이상치가 있다. 양손 손목 프레임 차분 속도의 trial별 최댓값 중앙값은 2024년 2.58m/s, 2025년 4.05m/s다. 그러나 `2025-12-18/P0002/T0009`의 RIGHT_WRIST frame237→238에서 약 **772.56m/s**의 차분값이 나온다. 좌표는 `[-7.943,11.919,30.998]`→`[23.762,0.935,8.222]` ft로 급변한다. 이는 모션 기준으로 그대로 사용할 수 없는 추적 이상치 사례다. 이 값은 실제 손목 속도라는 주장이나 센서 오류 원인 확정이 아니다. 결측이 없는 P0002/T0041에서도 손목 차분 최대 약 37.13m/s가 있어 finite 검사만으로 품질을 보장할 수 없다.

왼쪽 상완/전완/대퇴/하퇴의 `(max length−min length)/median length` 중 최댓값을 trial별로 조사했을 때 중앙값은 2024년 0.517, 2025년 0.318이다. marker 정의/노이즈에 따른 분절 길이 변화가 커서 가변 뼈 길이를 그대로 MuJoCo에 전달하면 안 된다. 안정적인 고정체형 skeleton fitting이 필요하다. 이는 좌측 4개 분절의 탐색 지표이며 전체 관절 정확도를 측정한 결과가 아니다.

## 4. 좌표계와 단위 정규화

로컬 `assets/coordinate_system.png`와 공식 README를 확인했다. 코트 중앙 원점, 코트 길이 방향 +x, 그림 위쪽 +y, 오른손 법칙의 위쪽 +z다. 인체/공은 **Z-up, right-handed, feet**이며 현재 SMPL 입력은 **Y-up, meters**다. landing_x/y는 별도 hoop coordinate 및 inch이므로 player/ball 좌표와 혼합하지 않는다.

권장 절차는 다음과 같다.

1. 모든 인체/공 xyz에 0.3048을 곱해 meter로 변환한다. landing 좌표에는 0.0254, 진입각 필요 시 π/180을 각각 적용한다.
2. 시퀀스 최초 유효 pelvis 근처의 고정 XY 원점을 선택하고 전체 시퀀스에 동일하게 적용한다. 매 프레임 pelvis를 0으로 만드는 처리로 실제 root motion을 지우지 않는다. 원래 court offset은 sidecar에 기록한다.
3. 골반/어깨/발 또는 공 진행 방향으로 heading을 추정하되 측면/뒤집힘을 검증한다. 고정 yaw 정렬을 player와 ball에 동일하게 적용한다. README의 +x 방향을 모든 trial의 실제 몸 방향으로 단정하지 않는다.
4. 현재 `SMPL_TO_WORLD=C`는 `(x,y,z)_SMPL→(z,x,y)_MuJoCo`다. 정규화 Z-up point를 `C⁻¹`로 바꾸면 Y-up SMPL point가 된다. C⁻¹는 `(x,y,z)_world→(y,z,x)_SMPL`이다. 이미 Z-up인 SPL에 C를 다시 직접 적용하면 틀린다.
5. 바닥/landmark 높이는 정지 발 구간과 모델의 heel/toe 관측점을 함께 이용해 추정한다. 모든 ankle/heel marker를 z=0으로 강제하지 않는다. landmark 중심이 실제 접촉면은 아니다.

정규화 식(설계)은 `p_smpl=C⁻¹ Rz(-ψ) (0.3048 p_SPL−o)`이며 o는 meter 단위의 고정 world origin, ψ는 고정 yaw다. 회전도 같은 basis/heading 변환을 적용하고 root를 제외한 local 회전과 global 회전을 구분한다. handedness 검증은 det(R)=+1 및 좌우 hip/wrist 방향으로 수행한다.

## 5. 현재 SMPL+H 구조와 관절 매핑

팀 `smpl_joint_names.py`의 SMPLH_BONE_ORDER_NAMES는 **52개**다. root 1 + body non-root 21 + 양손 15씩 30이다. pose는 52×3=156 axis-angle 성분(rad), beta는 16개다. SMPL의 L_Hand/R_Hand를 별도 추가하면 이 SMPL+H의 52개 순서와 맞지 않는다.

```text
Pelvis
├─ L_Hip → L_Knee → L_Ankle → L_Toe
├─ R_Hip → R_Knee → R_Ankle → R_Toe
└─ Torso → Spine → Chest → Neck → Head
                         ├─ L_Thorax → L_Shoulder → L_Elbow → L_Wrist
                         │                                  └─ 각 Finger1→2→3 ×5
                         └─ R_Thorax → R_Shoulder → R_Elbow → R_Wrist
                                                            └─ 각 Finger1→2→3 ×5
```

MuJoCo는 52 human body + world, free root 1개 및 XYZ hinge 153개, actuator 153개, nq160/nv159다. SMPL 순서와 MuJoCo 트리 순서는 다르므로 기존 이름 기반 mapping을 재사용해야 한다. MuJoCo 손가락 hinge가 존재한다는 사실과 실제 SPL 손 pose를 이미 추종했다는 주장은 구별한다.

다음 표에서 “직접 위치 후보”는 해부학적으로 대응할 위치 관측을 의미하며, 회전을 직접 복사하거나 SPL marker와 regressed SMPL 관절 중심이 완전히 같다는 뜻은 아니다. 좌우 두 이름을 한 행으로 썼다. 피부/추정 marker 차이는 fitting에서 고려한다.

| SPL marker | SMPL+H 후보 | 분류/처리 |
|---|---|---|
| LEFT_HIP / RIGHT_HIP | L_Hip / R_Hip | 직접 위치 후보 |
| LEFT_KNEE / RIGHT_KNEE | L_Knee / R_Knee | 직접 위치 후보 |
| LEFT_ANKLE / RIGHT_ANKLE | L_Ankle / R_Ankle | 직접 위치 후보; 실제 접촉면과 다름 |
| LEFT_SHOULDER / RIGHT_SHOULDER | L_Shoulder / R_Shoulder | 직접 위치 후보; L/R_Thorax와 혼동하지 않음 |
| LEFT_ELBOW / RIGHT_ELBOW | L_Elbow / R_Elbow | 직접 위치 후보 |
| LEFT_WRIST / RIGHT_WRIST | L_Wrist / R_Wrist | 직접 위치 후보; 손 방향은 별도 추정 |
| MID_HIP | Pelvis | 근사 위치 anchor; 2024는 두 hip 평균으로 추정 |
| NECK | Neck | 근사 위치 후보; 2024는 어깨 평균/몸통 fitting으로 추정 |
| NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR | Head 주변 mesh landmark | 5개 모두 독립 회전 관절이 아님; extra vertex landmark 관측 모델 필요 |
| LEFT_BIG_TOE / RIGHT_BIG_TOE | L_Toe / R_Toe 또는 mesh toe landmark | SMPL toe joint와 detector marker 차이를 검증 |
| LEFT_SMALL_TOE / RIGHT_SMALL_TOE | mesh small-toe landmark | 별도 MuJoCo 작은발가락 hinge 없음 |
| LEFT_HEEL / RIGHT_HEEL | mesh heel landmark | 별도 heel 회전 관절 없음; foot orientation/contact 보조 |
| LEFT_THUMB / RIGHT_THUMB | Thumb MCP 주변 landmark | coarse point; Thumb1으로 단정하지 않음 |
| LEFT_PINKY / RIGHT_PINKY | Pinky MCP 주변 landmark | coarse point; 상세 손 key와 중복 가중 방지 |

손가락은 FIRST=Thumb, SECOND=Index, THIRD=Middle, FOURTH=Ring, FIFTH=Pinky이다. 아래 S는 LEFT/RIGHT, s는 L/R이며 각 행은 양손 각각에 적용된다. 손가락 root의 실제 joint center 오차/축은 MANO 템플릿에서 검증해야 한다.

| SPL 세부 marker | SMPL+H/MANO 후보 | 비고 |
|---|---|---|
| S_FIRST_FINGER_CMC | s_Thumb1 | thumb base 후보 |
| S_FIRST_FINGER_MCP | s_Thumb2 | thumb MCP 후보 |
| S_FIRST_FINGER_IP | s_Thumb3 | thumb IP 후보 |
| S_FIRST_FINGER_DISTAL | thumb fingertip mesh vertex | 4번째 회전 관절 아님 |
| S_SECOND_FINGER_MCP / PIP / DIP | s_Index1 / Index2 / Index3 | 3개의 위치 후보 |
| S_SECOND_FINGER_DISTAL | index fingertip vertex | endpoint 관측 |
| S_THIRD_FINGER_MCP / PIP / DIP | s_Middle1 / Middle2 / Middle3 | 3개의 위치 후보 |
| S_THIRD_FINGER_DISTAL | middle fingertip vertex | endpoint 관측 |
| S_FOURTH_FINGER_MCP / PIP / DIP | s_Ring1 / Ring2 / Ring3 | 3개의 위치 후보 |
| S_FOURTH_FINGER_DISTAL | ring fingertip vertex | endpoint 관측 |
| S_FIFTH_FINGER_MCP / PIP / DIP | s_Pinky1 / Pinky2 / Pinky3 | 3개의 위치 후보 |
| S_FIFTH_FINGER_DISTAL | pinky fingertip vertex | endpoint 관측 |

2025 P0001/T0001을 조사했을 때 LEFT_THUMB는 LEFT_FIRST_FINGER_MCP와, LEFT_PINKY는 LEFT_FIFTH_FINGER_MCP와 프레임별로 정확히 동일했다. 이 샘플에서 THUMB와 CMC는 median 약 25.9mm 차이가 난다. coarse point를 상세 CMC와 같다고 간주하면 안 된다. 모든 trial의 중복 정책은 후속 loader에서 검사해야 한다.

**SPL에서 직접 관측하지 못하는 SMPL 관절:** Torso, Spine, Chest, L/R_Thorax는 각 관절에 해당하는 독립 marker가 없다. 체간·어깨 관측과 pose prior로 분배해야 한다. Head는 얼굴 점을 이용한 간접 방향 추정이다. Pelvis/Neck도 marker 중심과 SMPL regressor 중심의 차이가 존재한다. 작은발가락, heel, 얼굴점, fingertip, ball은 52개 회전 관절로 1:1 추가할 수 없다. extra landmark 관측을 사용하거나 해당 항목을 fitting loss에서 제외한다.

표는 SPL 69개(기본 27 + 상세 손 40 + MID_HIP/NECK 2)를 모두 포함한다. 모델의 52개와 관측점의 69개는 같은 종류의 개수 비교가 아니다. 52개의 kinematic joints와 라이브러리 forward의 `joints` 반환 길이도 같다고 가정하면 안 된다. smplx는 얼굴/발/손끝의 extra vertex joints를 덧붙일 수 있다. named observation adapter와 명시적 52개 pose slicing이 필요하다.

## 6. Root와 손 회전 복원 가능성

Root position은 유효 MID_HIP 또는 양쪽 hip 중점으로 초기화할 수 있다. 2025년 MID_HIP와 hip 평균의 차이는 전수 최대 약 21.0mm였다. NECK와 shoulder 평균의 차이도 최대 약 79.3mm라서 둘을 정확한 동의어로 취급하지 않는다. fitting에서 shaped pelvis 위치와 관측 중심 차이를 보정해야 한다.

Root orientation은 좌우 hip 방향, pelvis→상체 방향을 정규직교화하여 초기화하고 어깨/발 방향으로 앞뒤 부호를 검증할 수 있다. pose의 기준 축과 SMPL rest frame을 함께 맞춰야 한다. 상대 점만으로 bone roll이 결정되지 않고, 자세가 퇴화하거나 관측이 가려지면 방향이 불안정하다. heading/rotation의 시간 prior와 SO(3) 기반 평활화가 필요하다. 2024 데이터도 기본 hip/shoulder로 이 초기화는 가능하다.

손은 wrist + MCP 및 손가락 chains로 palm plane과 굴곡을 추정할 수 있다. 2025년 손가락 위치 fitting은 가능하지만 각 bone의 축 방향 twist, 말단 관절 전체 3DOF 및 실제 힘/그립을 유일하게 식별할 수 없다. joint axis, 생리적 limits, MANO hand prior 및 temporal constraints가 필요하다. 손가락 distal point는 마지막 bone의 endpoint 관측으로 사용한다. 2024년은 THUMB/PINKY와 wrist로 palm 방향의 일부 정보를 얻을 수 있으나 상세 15관절/손 회전을 복원하는 충분한 데이터는 아니다. 중립 손 또는 명시적 prior 추정 결과로 표시해야 한다.

**현재 parser와 모션의 손 기준 자세 확인이 선행되어야 한다.** `_load_smplh_parser`는 use_pca=False/num_betas=16이며 flat_hand_mean을 명시하지 않는다. 설치된 smplx의 기본값은 flat_hand_mean=False이고 forward에서 `full_pose += pose_mean`을 수행한다. 반면 SMPLHMapping은 NPZ rotvec를 추가 mean 없이 해석한다. 기존 M3-1A의 “손 입력 0”을 parser의 “평평한 손 0”과 같다고 볼 수 없다. XML은 parser 기준 rest skeleton에서 생성되므로 평균을 단순히 더하거나 빼기만 하면 된다고 단정해서도 안 된다. 미래 exporter는 parameter pose / effective full pose / XML rest basis의 관계를 확인하고 한 프레임 손가락 FK·joint position·fingertip을 SMPL+H와 MuJoCo 사이에서 비교해야 한다. 이번 작업에서는 이를 수정하거나 model forward로 fitting을 수행하지 않았다.

## 7. 권장 fitting 설계 (미구현)

```text
SPL JSON + 원본 시간/단위/유효 mask
  → 품질 검사·clip 선정·단위/좌표/heading 정규화
  → named observation mapping (관절 + 필요한 mesh landmark)
  → source shape와 root 초기화
  → robust SMPL+H sequence fitting (몸통/사지 → 손 → 시간 joint optimization)
  → canonical SMPL+H pose / root sequence / metadata
  → 기존 SMPLHMotion NPZ / SMPLHMapping
  → 기존 BalancedPD + COM, 네 target shape 비교
```

### 최적화 변수와 절차

변수는 프레임별 local rotations R_t,j, global root rotation R_t,0, translation t_t 및 참가자에 공통인 beta_s∈R16이다. 첫 실험은 현재 male model로 제한한 effective shape fitting으로 시작할 수 있으나 참가자의 성별/실제 체형을 단정해서는 안 된다. 고정 zero beta baseline과 source beta fitting을 비교해 형상/동작 변수의 혼동을 평가한다.

1. CPU에서 PyTorch의 미분 가능한 SMPL+H forward를 사용한다. 현재 parser의 `get_joints_verts`/`forward`는 fitting 관측 생성 후보지만 `.float()`에 맞춘 float32 baseline과 finite gradient 검사가 필요하다. no_grad 및 numpy 변환을 사용하는 `get_offsets`/`get_mesh_offsets`를 최적화 루프에 쓰면 안 된다.
2. 몸통/사지/root만 낮은 시간 해상도에서 초기화한다. 입력 위치로 단순 IK를 초기값에 사용할 수 있지만 축 twist는 prior로 제한한다. 참가자 공통 beta는 여러 안정 프레임/시도를 이용해 강하게 정규화하며 번갈아 최적화한다.
3. 초기 손 pose는 neutral/low-dimensional MANO prior로 제한하고, 유효한 2025년 손 keypoint에서 손/손목을 추가 fitting한다. pretrained prior 파일을 갖고 있다고 가정하지 않는다. use_pca=False의 최종 45성분/손 axis-angle 형식을 유지하되 PCA prior를 사용한다면 명시적으로 expand한다.
4. warm-start Adam으로 초기 안정화 후 CPU L-BFGS로 수렴을 다듬는 방법을 우선 검토한다. 프레임별 독립 fitting은 초기화에만 쓰고 마지막에는 겹치는 시간 window를 공동 최적화해 경계 불연속을 방지한다. CPU가 첫 재현 기준이고 MPS 가속은 검증 후 선택 사항이다.
5. SO(3) tangent increment 또는 연속 회전 표현으로 최적화하고 출력할 때 local axis-angle로 바꾼다. rotvec의 ±π 분기값을 단순 차분하여 smoothness loss를 구성하지 않는다.

이 설계는 3D 위치 관측을 사용하는 SMPLify 계열 목적함수의 응용 제안이다. 원래 이미지의 2D projection fitting 코드를 그대로 적용한다는 뜻이 아니다. 구현된 solver나 수렴/추종 보장은 아직 없다.

### 목적함수

관측 모델 h_k(beta,R_t,t_t)는 mapped SMPL joint 또는 mesh landmark다. 측정 p_t,k는 정규화 meter 좌표이며 mask m_t,k와 품질 weight w_t,k를 둔다.

\[
L_{pos}=\frac{\sum_{t,k} m_{t,k}w_{t,k}\,\rho(\|h_k(\beta,R_t,t_t)-p_{t,k}\|/\sigma_k)}{\max(1,\sum_{t,k} m_{t,k}w_{t,k})}
\]

ρ는 Huber 등의 robust loss, σ_k는 body/hand/face별 관측 허용 오차 scale이다. 데이터에 confidence가 없으므로 bone-length/velocity/finite 검사 기반 heuristic weight를 **추정 품질**로 표시한다. NaN을 남긴 채 0 mask를 곱하면 NaN이 전파되므로 finite 관측만 선택해 계산한다. 몸/손/얼굴 loss를 그룹별로 정규화하여 40개 손 marker가 몸 loss를 압도하지 않게 한다. 중복 coarse/detailed 점에는 가중치를 중복 부여하지 않는다.

전체 목적은 아래 항목의 가중합으로 설계한다.

\[
L=\lambda_b L_{body}+\lambda_h L_{hand}+\lambda_f L_{landmark}
+\lambda_p L_{pose\ prior}+\lambda_{\beta}\|\beta\|^2
+\lambda_{lim}L_{anatomical\ limits}+\lambda_vL_{velocity}
+\lambda_aL_{acceleration}+\lambda_cL_{contact}+\lambda_{col}L_{collision}.
\]

- 관절/root 회전 속도는 Log(R_t,jᵀR_t+1,j)/Δt 기반으로, translation은 차분/Δt로 제한한다. 속도 변화/Δt로 가속도 loss를 둔다. 두 FPS에 같은 raw 차분 penalty를 적용하지 않는다.
- 팔꿈치/무릎/손가락의 생리적 관절 제약 및 spine/shoulder 분배 prior를 둔다. 기존 XML의 범위 검사는 넓으며 해부학적 유효성의 증거가 아니다.
- 정지 발 구간에는 foot landmark 속도/바닥 침투/정지 anchor loss를 적용한다. 실제 들린 발/heel은 고정하지 않는다. foot contact label은 데이터에 없으므로 높이와 속도로 추정하고 별도 표시한다.
- 손 mesh self-collision 및 선택적 공 sphere proximity는 후속 항목이다. 공 중심 finite 값만으로 완전한 그립/접촉력이나 release를 알 수 없고 공 회전/radius 측정도 제공되지 않는다. 첫 body fitting loss에 ball을 인체 관절처럼 추가하지 않는다.
- loss 가중치/오차 임계값은 단일 trial pilot의 재투영/3D 잔차와 물리 결과로 결정한다. 이 문서의 탐색 필터 숫자는 fitting acceptance 기준이 아니다.

### beta, root translation과 체형별 retargeting

Source beta_s는 참가자별로 고정하고 각 프레임에서 변하지 않게 한다. 16차원 shape를 sparse/noisy joint 위치만으로 모두 정확히 식별할 수 없으므로 강한 prior와 불확실성 기록이 필요하다. 신장/질량 메타데이터가 없고 beta로 실제 질량을 역추정할 수 없다. meter 단위이므로 자유 global scale로 단위/체형 오차를 흡수하는 방식을 기본으로 하지 않는다.

Target beta는 기존 body_01=zeros16, body_02=[2,0,…], body_03=[-2,0,…], body_04=[0,2,0,…]를 유지한다. source fitting과 target shape 최적화는 다른 작업이다. 첫 비교는 동일 fitted angular motion + 체형별 일정한 바닥 정렬을 네 target에 적용하고 기존 추종/균형 지표를 평가한다. 이후 실제 retargeting은 target beta를 고정한 채 foot contact·wrist/ball release endpoint 등 공간 제약을 재최적화한다. 네 target에 원본의 모든 위치를 동시에 정확히 맞추는 것은 서로 다른 limb lengths 때문에 일반적으로 불가능하다.

권장 NPZ root 규약은 `coordinate_system=smpl`, `translation_kind=root_position`이다. 각 frame에 fitting된 pelvis의 실제 Y-up meter 위치를 넣는다. SMPL library의 transl은 root joint 그 자체가 아니라 `J_pelvis=J_rest(beta)+transl` 규약의 이동이다. source beta로 얻은 transl을 다른 target에 그대로 넣으면 shaped pelvis offset이 바뀔 수 있다. 기존 `smpl_transl` 변환은 모델 body_pos의 shaped pelvis offset을 추가하므로 source/target beta 및 rest basis 일치를 먼저 검증해야 한다.

pose[:,:3]에는 SMPL 기준 global orientation, 나머지에는 정확한 SMPLH_BONE_ORDER_NAMES 순서의 local axis-angle을 넣는다. root rotation의 세계 좌표 변환과 손 mean pose 변환을 적용했는지 sidecar에 기록한다.

## 8. 기존 모션/제어 코드 재사용 범위

| 기존 코드 | 재사용 | 부족한 부분 |
|---|---|---|
| SMPLHMotion | poses(N,156), root_position(N,3), fps, schema_version=1 NPZ 저장/로드 | raw SPL JSON, missing mask, source ID/beta/접촉/공/비등간격 timestamps는 지원하지 않음 |
| SMPLHMapping | 이름 기반 rotvec→XYZ hinge, C root 변환, 범위·gimbal 검사 | fitting/IK/특이점 해결/hand rest calibration은 없음 |
| BoundMotion | root SLERP, hinge/root 위치 보간 | hinge 선형 보간은 C0; 실제 빠른 손목의 속도/가속도 연속 보간 없음 |
| m3_control.run / BalancedPD | 기존 PD + COM, root 초기화, CSV 기록 | floating root는 직접 servo하지 않아 fitted root 궤적을 보장하지 않음 |
| examples/m3_control.py --motion-file | 실제 fitting NPZ의 generic 실행 입구 | --check는 imported motion에 허용되지 않고 기본 summary는 첫 1초를 제외함; imported 평가 기준은 후속 설계 필요 |
| examples/m3_free_throw.py | 합성 회귀 기준/평가 형식 참고 | 91프레임·30FPS·3초 고정 검사. 240프레임 60FPS 실데이터에는 그대로 쓸 수 없음 |
| SMPLH_Parser | 미분 가능한 mesh/joint 관측 후보 | extra landmark 이름 adapter와 pose_mean/rest 정합 검증 필요 |

SMPLHMotion은 임의 양의 FPS를 받을 수 있다. 처음에는 60FPS의 모션 길이를 유지하는 편이 좋다. 30FPS로 바꾼다면 위치/회전별 올바른 재샘플링과 손목/release peak 보존, 필요 시 anti-alias filtering을 검토한다. 기존 물리 450Hz 및 지표 30Hz는 입력 FPS와 다르다. `m3_control.run`은 frames=round(duration×30)으로 마지막 실행 길이를 반올림하므로 239/60초 시퀀스는 4초 실행되어 끝을 약 1/60초 hold할 수 있다. 평가에 원본/출력/물리 시간을 분리해 기록한다. 실제 자유투를 합성 검사의 3초에 맞추기 위해 임의로 time-warp하지 않는다.

최종 NPZ 외 별도 provenance sidecar 설계: dataset commit 및 source JSON 상대 경로/hash, session/participant/trial, clip start/end·원본 timestamps, nominal/input/output FPS, court→normalized→SMPL 변환/원점/heading/바닥 추정, source beta/model 설정, joint order 및 hand mean/rest 규약, 관측 valid mask/접촉 추정/release 후보, fitting 잔차와 필터/solver 설정, 합성 여부 및 라이선스 출처. NPZ 자체 schema를 이번 단계에서 바꾸지 않았다.

## 9. 라이선스 및 사용 조건

공식 저장소와 로컬 LICENSE는 **CC BY-NC-SA 4.0**을 표기한다. 데이터 공유/가공 시 출처 표시, 라이선스 링크, 변경 사실 표시, 비상업적 사용, 가공 데이터의 동일 라이선스 공유 조건을 확인해야 한다. 이 내용은 [공식 SPL LICENSE](https://github.com/Sport-Performance-Lab/SPL-Open-Data/blob/main/LICENSE)와 [Creative Commons 공식 요약](https://creativecommons.org/licenses/by-nc-sa/4.0/)에서 확인했다.

SPL LICENSE에는 표준 CC 표기 외에도 프로 스포츠 조직 또는 금융 분석 회사와 관련된 employee/contractor/주요 주주 등에 대해 유료 written license 없이는 어떤 사용도 금지한다는 추가 문구가 있다. 같은 파일의 “추가 제한 없음” 문구와 함께 존재하므로 이를 단순한 무제한 CC 데이터로 설명하면 안 된다. 이 추가 문구와 CC 조건의 법적 정합성은 이번 기술 분석에서 판정하지 않았다. 해당 범주의 사용/외부 제공이 예정되면 제공자에게 적용 범위를 확인해야 한다. 현재 로컬 학술 분석 작업은 계속 진행했으며 데이터 배포는 하지 않았다.

보고서의 권장 출처 표기는 “SPL Open Data, Maple Leaf Sports & Entertainment Sport Performance Lab, session/trial 및 사용 commit, CC BY-NC-SA 4.0, 좌표 정규화 및 SMPL+H fitting 등 수행한 변경 명시”이다. 원본 데이터 및 파생 motion을 팀 코드의 라이선스만으로 자동 재배포하지 않는다. 코드와 파생 데이터의 라이선스는 구별한다. 별도의 SMPL+H/MANO 모델 파일도 [MANO 모델 제공 조건](https://mano.is.tue.mpg.de/)에 따라 관리하고, 데이터 라이선스가 모델 재배포 권한을 주는 것은 아니다.

## 10. 권장 구현 순서와 다음 작업

다음 작업은 모두 **후속 단계 제안**이며 이번에는 구현하지 않았다.

1. **M3-1B-1: read-only SPL loader와 품질 리포트.** session/participant/trial key, null/NaN mask, 단위·FPS 검사, finite 값의 bone-length/속도/손가락/공 이상치, 전체 결측 연속 구간 길이를 기록한다. 참가자별 train/eval 분리가 필요해지면 같은 P0001의 두 세션을 서로 다른 사람으로 나누지 않는다.
2. **M3-1B-2: 한 trial의 좌표 검증/clip 선정.** 2025년 후보를 3D 시각화하고 최초 정지 발·굴곡·팔 준비·release 후보·follow-through를 확인한다. 공과 손의 분리 및 상승 궤적으로 release 후보를 추정하되 ground truth가 아님을 기록한다. 공 입출력 잡음 구간과 실제 시도 구간을 분리하고 trial 전체 길이를 3초로 고정하지 않는다.
3. **M3-1B-3: 모델 observation/rest 검증.** SPL→SMPL named map, mesh extra landmark, hand pose_mean, XML rest offsets와 root transl 규약을 단일 알려진 pose의 FK로 확인한다. 먼저 body 중심점, 이어 손가락/손끝 position parity를 검사한다. 이 검사가 손 fitting export의 선행 조건이다.
4. **M3-1C: body/root fitting pilot.** beta=0 baseline 및 고정 source beta, robust 위치 loss + pose/시간/관절 제약. per-joint/phase residual, root 연속성, bone invariance, Jacobian/finite gradients를 평가한다. 3D 위치 잔차 허용 기준을 pilot에서 정한다.
5. **M3-1D: 2025년 hand fitting.** palm/손목 frame과 손끝 위치, MANO prior 및 손 mean 정합을 검증한다. 충분히 관측되지 않은 twist와 occluded frame을 추정값으로 구분한다. 2024년은 sparse-hand baseline으로 별도 취급한다.
6. **M3-1E: NPZ export와 네 체형 물리 평가.** 정규화/손 rest/FK gate 통과 후 common angular baseline을 기존 generic `--motion-file` 경로에 입력한다. 관절 오차, root 오차, COM/양발 접촉/접촉점 slip, 낙상, 토크 포화 및 손목 공간 궤적을 비교한다. fit가 좋더라도 unactuated root와 물리 contact 때문에 실패할 수 있다. target contact/endpoint retargeting은 다음 단계로 분리한다.

### 시작 후보 (자동 탐색 결과, 최종 선정 아님)

다음 임시 필터를 2025년 trial에 적용했을 때 5개가 남았다: made, 69개 인체 point 전 프레임 finite, 공 finite≥99%, 양손 손목 차분 최고속도<10m/s, 좌측 4분절 각각의 length relative range<20%. 마지막 조건은 좌측만 검사했다. 우측/손가락 잡음·공 속도·clip 및 release 품질은 아직 심사하지 않았으므로 “clean ground truth”로 부르지 않는다.

| 세션 | 참가자/시도 | 최대 손목 차분 m/s | 좌측 length relative range 최대 |
|---|---|---:|---:|
| 2025-12-18 | P0001/T0087 | 3.541 | 19.64% |
| 2025-12-18 | P0002/T0042 | 3.072 | 19.67% |
| 2025-12-18 | P0002/T0049 | 2.349 | 17.88% |
| 2025-12-18 | P0003/T0063 | 3.417 | 18.38% |
| 2025-12-18 | P0005/T0033 | 3.785 | 19.86% |

우선 확인 후보는 P0002/T0049다(최대 손목 차분 2.35m/s, 좌측 length relative range 최대 17.88%, 모든 인체/공 finite). 그래도 발 landmark z 범위가 약 0.120~0.208m여서 marker 높이를 바닥과 같다고 놓을 수 없다. 이 후보를 실제로 fitting하거나 reference NPZ로 변환하지 않았다.

## 11. 이번 단계 산출물 및 완료 체크리스트

팀 저장소 신규 파일은 **`M3_1_SPL_DATASET_ANALYSIS.md` 1개**다. 전수 분석용 임시 스크립트/해시/집계는 `/tmp`에만 두었고 원본 JSON/모델을 팀 저장소로 복사하지 않았다. 기존 `M3_1_VERIFICATION.md`는 이전 단계 파일이며 이번 단계에서 변경하지 않았다.

- [x] 전체 파일/폴더 및 자유투 583시도 전수 조사
- [x] 참가자·세션·FPS·timestamp·좌표/단위 확인
- [x] 69개 관절 키 및 2024/2025 실제 관측 가용성 구분
- [x] 손목/손가락/공 데이터와 null/NaN/좌표 이상치 확인
- [x] SMPL+H 52관절 대응/미대응 및 root/hand 복원 한계 정리
- [x] robust fitting loss, 시간/회전 제약, source/target beta와 root 처리 설계
- [x] 기존 모션/제어 재사용 범위와 hand mean/rest 선행 검증 명시
- [x] 공식 라이선스·추가 사용 조건·파생 데이터 관리 확인
- [x] 다음 구현 순서 및 임시 시작 후보 제시
- [ ] 실제 자유투 기준 clip의 시각 검수/최종 선정
- [ ] SMPL+H fitting 및 실제 모션 NPZ 생성
- [ ] 실제 SPL 모션의 네 체형 MuJoCo 물리 검증

체크리스트는 문서에서만 작성했으며 GitHub Issue를 변경하지 않았다. 분석 단계에서 새 unit test나 물리 simulation을 수행하지 않았다. 데이터/스키마 전수 검사와 문서 검증이 이번 검증 결과다. 기존 M1~M3-1A Python 코드, 원본 데이터, 원본 개발 환경의 수정 및 Git branch/commit/push/PR 작업은 없다.

## 12. 근거와 재현 메모

주요 근거는 로컬 JSON 전수 스캔, 로컬 README/LICENSE/좌표 이미지, 팀 `smpl_joint_names.py`, `smpl_parser.py`, `smpl_local_robot.py`, `smplh_motion.py`, `m3_control.py`, `m3_free_throw.py` 및 기존 설치 smplx `body_models.py`다. 관측/fitting 권고는 위 사실과 아래 원자료를 바탕으로 한 **설계 판단**이다.

- [SPL 공식 저장소](https://github.com/Sport-Performance-Lab/SPL-Open-Data)
- [공식 자유투 문서](https://github.com/Sport-Performance-Lab/SPL-Open-Data/blob/main/basketball/freethrow/README.md)
- [SPL LICENSE](https://github.com/Sport-Performance-Lab/SPL-Open-Data/blob/main/LICENSE), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- [SMPL+H/MANO 원 논문 및 모델](https://mano.is.tue.mpg.de/)
- [SMPLify 모델 fitting 원자료](https://smplify.is.tue.mpg.de/)
- [SMPLify-X 원 논문](https://arxiv.org/abs/1904.05866)
- [smplx 공식 모델 구현](https://github.com/vchoutas/smplx/blob/main/smplx/body_models.py), [extra vertex selector](https://github.com/vchoutas/smplx/blob/main/smplx/vertex_joint_selector.py)

공식 animate.py는 현재 JSON의 LEFT_/RIGHT_와 다른 L_/R_ marker 이름을 기본 connections 및 hip centering에 사용하고 interval을 30FPS로 고정한다. usage.ipynb의 25 keypoint/30FPS 설명도 현재 69-key/60FPS 세션과 맞지 않는다. 이 유틸리티/노트북을 그대로 실행한 그림을 검증 근거로 삼지 않았고 수정하지 않았다. 미래 시각화는 현재 JSON 스키마와 FPS를 기준으로 만들어야 한다.

이번 전수 검사 실행: 기존 Python 3.12 런타임으로 `/tmp/m3_spl_audit.py`를 실행했다. 복사/변환/fitting/시뮬레이션 명령은 이번 단계에서 실행하지 않았다. 간단한 참가자별 파일 수 확인은 다음과 같이 재현할 수 있다.

```bash
# MuJoCo 루트에서, 설치/원본 파일 수정 없이 실행
PYTHONDONTWRITEBYTECODE=1 basketball-r1/.venv/bin/python - <<'PYCOUNT'
from pathlib import Path
from collections import Counter
root = Path('SPL-Open-Data/basketball/freethrow/data')
paths = sorted(root.glob('*/*/*.json'))
print('trials:', len(paths))
print(Counter((p.parts[-3], p.parts[-2]) for p in paths))
PYCOUNT
```

최종 보존 검사: 분석 전후 데이터셋 파일 594개의 SHA-256은 모두 동일했다. 팀 기존 Python 116개도 모두 동일했다. 원본 데이터/기존 코드를 변경하지 않았음을 내용 해시로 확인했다.
