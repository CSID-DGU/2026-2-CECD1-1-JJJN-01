# 백엔드가 만들어야 할 것 — 프론트 기준 정리

작성일 2026-11-07. 프론트(React 웹)는 mock 으로 끝까지 동작하는 상태이고, 백엔드가 아래 **두 개의 HTTP API** 만 만들어 주면 바로 붙습니다.
이 문서는 "HTTP API 를 많이 안 만져 봤다" 는 전제로 씁니다. 이미 아는 부분은 건너뛰어도 됩니다.

> 프론트 쪽 계약 원본은 `frontend/src/types/analysis.ts` 입니다. 이 문서와 그 파일이 다르면 파일이 맞습니다.
> 필드 이름이나 구조를 바꾸고 싶으면 **바꿔도 됩니다.** 프론트에서 고칠 곳은 세 파일뿐이라 부담이 작습니다. 단, 바꾸기 전에 한 번 이야기해 주세요.

---

## 0. 한 장 요약

```
[사용자 폰 브라우저]                              [백엔드 GPU 서버]

 1. 종목 고름 (스쿼트 / 데드리프트 / 벤치프레스)
 2. 내 영상 올림 (+ 코치 영상 고름)
 3. "분석 시작" 누름
        │
        │  POST /analysis   (영상 파일 + 종목 + 코치 선택)
        ├──────────────────────────────────────────────►  작업을 큐에 넣고
        │  ◄──────────────────────────────────────────┤  { "jobId": "..." } 를 즉시 돌려줌
        │                                                 (분석은 백그라운드에서 진행)
 4. "분석 중" 화면
        │  GET /analysis/{jobId}     ← 1.5초마다 반복
        ├──────────────────────────────────────────────►
        │  ◄──────────────────────────────────────────┤  { "status": "extracting" }
        │  GET /analysis/{jobId}
        ├──────────────────────────────────────────────►
        │  ◄──────────────────────────────────────────┤  { "status": "scoring" }
        │          ... status 가 done 또는 failed 가 될 때까지 ...
        │  ◄──────────────────────────────────────────┤  { "status": "done", "result": { ...큰 JSON... } }
 5. 결과 화면 (점수 · 3D 뷰어 · 구간별 비교 · 피드백)
```

왜 하나로 안 합치나: 추출 → 리타겟팅 → 스코어링 → LLM 까지 수십 초 이상 걸립니다. 요청 하나가 그만큼 열려 있으면 모바일에서 끊기기 쉽고, "분석 중" 화면에서 단계별 진행도 못 보여 줍니다. 그래서 **"요청 받기"** 와 **"상태 물어보기"** 를 나눴습니다.

---

## 1. 꼭 알아야 할 HTTP 용어 (3분)

- **엔드포인트**: `메서드 + 경로` 한 쌍. 예) `POST /analysis`, `GET /analysis/abc123`.
- **GET / POST**: GET 은 "달라", POST 는 "이거 받아서 처리해". POST 에는 몸통(body)이 붙습니다.
- **multipart/form-data**: 파일을 보낼 때 쓰는 body 형식. HTML `<form>` 으로 파일 올리는 것과 같습니다. FastAPI 면 `UploadFile` + `Form`, Spring 이면 `MultipartFile` + `@RequestParam` 으로 받습니다.
- **JSON 응답**: `Content-Type: application/json` 헤더와 함께 JSON 문자열을 돌려주면 됩니다. 프론트는 이 헤더를 보고 JSON 으로 파싱합니다.
- **상태 코드**: 성공이면 `200`(또는 `202`), 요청이 잘못됐으면 `400`, 못 찾으면 `404`, 서버 오류면 `500`. 실패 응답에는 `{ "message": "사용자에게 보여줄 문장" }` 을 넣어 주면 프론트가 그대로 화면에 띄웁니다.
- **CORS**: 브라우저는 프론트 주소(예 `http://localhost:5173`)와 백엔드 주소(예 `http://gpu-server:8000`)가 다르면 기본적으로 요청을 막습니다. 백엔드가 응답에 `Access-Control-Allow-Origin` 헤더를 붙여야 합니다. 프레임워크마다 CORS 미들웨어가 있으니 **개발 중엔 모든 origin 허용**으로 켜 두세요. 이걸 빼먹으면 "서버에 연결할 수 없어요" 라고만 나와서 원인 찾기가 어렵습니다.

---

## 2. API 1 — 분석 요청

### `POST /analysis`

body: `multipart/form-data`

| 필드 | 타입 | 필수 | 설명 |
| --- | --- | --- | --- |
| `video` | 파일 | O | 사용자 영상 원본. 전처리 없이 폰에서 찍은 그대로 옵니다. mp4 / mov / webm, 100MB 이하, 30초 이하 (프론트에서 1차로 거르지만 서버에서도 검사하세요) |
| `exercise` | 문자열 | O | `"squat"` / `"deadlift"` / `"bench_press"` 중 하나 |
| `coachId` | 문자열 | △ | 기본 제공 코치 영상을 고른 경우. 아래 3절의 id 중 하나 |
| `coachVideo` | 파일 | △ | 사용자가 코치 영상을 직접 올린 경우. `coachId` 와 둘 중 하나만 옵니다 |

응답 (`200` 또는 `202`, JSON):

```json
{ "jobId": "0f3a9c1e" }
```

`jobId` 는 서버가 정하는 아무 문자열입니다 (UUID, 증가 숫자 등). 프론트는 이 값을 들고 2번 API 를 반복 호출합니다.

서버가 할 일:

1. 영상 두 개(또는 사용자 영상 + 코치 id)를 받아서 저장
2. 작업 레코드를 만들고 `status = queued` 로 둠
3. **즉시** `jobId` 를 돌려줌 — 여기서 분석을 끝까지 기다리면 안 됩니다
4. 백그라운드(워커 · 큐 · 스레드 무엇이든)에서 추출 → 리타겟팅 → 스코어링 → LLM 을 돌리며 `status` 를 올림

에러 예: 영상 형식이 아니면 `400` + `{ "message": "mp4, mov, webm 영상만 올릴 수 있어요." }`

---

## 3. API 2 — 작업 상태 조회 (폴링)

### `GET /analysis/{jobId}`

프론트가 **1.5초마다** 부릅니다. `status` 가 `done` 또는 `failed` 가 되면 멈춥니다.

응답 (`200`, JSON):

```json
{
  "jobId": "0f3a9c1e",
  "status": "retargeting",
  "progress": 0.4
}
```

| 필드 | 타입 | 필수 | 설명 |
| --- | --- | --- | --- |
| `jobId` | 문자열 | O | 그대로 되돌려 줌 |
| `status` | 문자열 | O | 아래 표 중 하나 |
| `progress` | 숫자 0~1 | X | 있으면 좋고 없어도 됨. 지금 프론트는 안 씀 |
| `result` | 객체 | `done` 일 때 O | 4절의 분석 결과 |
| `error` | 문자열 | `failed` 일 때 O | 사용자에게 그대로 보여줄 문장. 예) `"영상에서 사람을 찾지 못했어요. 전신이 나오게 다시 찍어 주세요."` |

`status` 값과 프론트 "분석 중" 화면의 표시:

| `status` | 화면에 뜨는 글 | 의미 |
| --- | --- | --- |
| `queued` | (스피너만) | 큐에서 대기 |
| `extracting` | 동작 읽는 중 | 영상 → 스켈레톤 추출 |
| `retargeting` | 내 체형에 맞추는 중 | 코치 스켈레톤을 사용자 체형으로 |
| `scoring` | 비교하는 중 | CP-DTW 정렬 · 구간 판정 · 점수 |
| `llm` | 피드백 쓰는 중 | LLM 문장 생성 |
| `done` | → 결과 화면으로 이동 | `result` 필수 |
| `failed` | 실패 화면 + "영상 다시 올리기" | `error` 필수 |

단계를 건너뛰어도 됩니다 (예: LLM 이 아직 없으면 `scoring` 다음 바로 `done`). 순서만 지켜 주세요.
없는 `jobId` 면 `404` + `{ "message": "분석 작업을 찾을 수 없어요." }`.

---

## 4. `result` 의 모양 (가장 중요)

`status: "done"` 일 때 `result` 에 들어갈 JSON 입니다. 전체 예시는 맨 아래 부록에 있습니다.

```
result
├── jobId            문자열
├── exercise         "squat" | "deadlift" | "bench_press"
├── score            { total, bySegment? }
├── segments[]       체크포인트 구간 목록 (판정 · 타이밍 · 차이 큰 뼈)
├── student          사용자 스켈레톤 시퀀스
├── coachRetargeted  사용자 체형으로 리타겟팅된 코치 스켈레톤 시퀀스
├── alignment        DTW 정렬 경로 [[사용자프레임, 코치프레임], ...]
└── feedback?        LLM 피드백 (아직 없으면 빼도 됨)
```

### 4-1. `score`

```json
{ "total": 72, "bySegment": { "seg-1": 71, "seg-2": 38 } }
```

- `total`: 0~100 정수. 화면에 큰 숫자로 뜨고 80 이상 초록 / 60 이상 노랑 / 그 아래 빨강.
- `bySegment`: 구간 id → 점수. 없어도 됩니다. 있으면 구간 상세에 "71점" 식으로 보입니다.
- **DTW 비용을 어떻게 0~100 으로 바꿀지는 백엔드가 정합니다.** 프론트는 숫자만 받습니다.

### 4-2. `segments[]` — 체크포인트 구간

슬라이드의 "일치 · 누락 · 추가 · 대체" + "slow / fast" 판정이 여기 들어갑니다. 구간 하나:

```json
{
  "id": "seg-1",
  "label": "하강",
  "verdict": "match",
  "studentRange": [0.3, 1.5],
  "coachRange": [0.3, 1.3],
  "timing": { "label": "slow", "ratio": 1.2 },
  "deviations": [
    { "bone": "torso", "angleDiffDeg": 15.9 },
    { "bone": "left_shin", "angleDiffDeg": 6.1 }
  ]
}
```

| 필드 | 설명 |
| --- | --- |
| `id` | 구간 식별자. `feedback.items[].segmentId` 와 `score.bySegment` 의 키로 씁니다. 형식 자유 |
| `label` | 화면에 보이는 이름. 한국어. 예) 시작 자세 / 하강 / 바닥 유지 / 상승 / 마무리 자세 |
| `verdict` | `match`(일치) / `missing`(누락: 코치엔 있는데 사용자엔 없음) / `extra`(추가: 사용자에게만 있음) / `substitute`(대체) |
| `studentRange` | 사용자 영상에서 이 구간의 [시작초, 끝초]. `missing` 이면 생략 가능 |
| `coachRange` | 코치 영상에서의 [시작초, 끝초]. `extra` 면 생략 가능 |
| `timing` | `match` 일 때만. `label` 은 `ok` / `slow` / `fast`, `ratio` = 사용자 소요 시간 ÷ 코치 소요 시간 |
| `deviations` | 이 구간에서 차이가 컸던 뼈 상위 몇 개. 없으면 `[]`. `bone` 은 5절의 뼈 이름, `angleDiffDeg` 는 사용자 − 코치 각도(도) |

`studentRange` 는 타임라인 블록 길이와 "구간 누르면 3D 뷰어가 그 시점으로 이동" 에 쓰이므로 **사용자 영상 기준 초 단위**로 정확히 주세요. 구간들은 시간순으로 정렬해서 보내 주세요.

### 4-3. `student` / `coachRetargeted` — 스켈레톤 시퀀스

둘 다 같은 형식입니다.

```json
{
  "jointNames": ["nose", "neck", "left_shoulder", "..."],
  "fps": 30,
  "frames": [
    [[0.01, 1.52, 0.0], [0.0, 1.38, 0.0], "..."],
    [[0.01, 1.52, 0.0], [0.0, 1.38, 0.0], "..."]
  ]
}
```

- `jointNames[j]`: j 번째 관절의 이름. **순서와 개수는 자유**입니다. 프론트는 이름만 보고 뼈를 잇습니다 (5절).
- `fps`: 프레임 간격. `student.fps` 로 재생 시간을 계산합니다.
- `frames[t][j]`: t 번째 프레임, j 번째 관절의 `[x, y, z]`.
- 좌표: **미터 단위, y 가 위**, 사람이 바닥(y≈0)에 서 있는 기준이면 가장 좋습니다. 다른 좌표계면 알려 주세요. 프론트가 바운딩 박스로 카메라를 맞추기 때문에 스케일이 조금 달라도 보이긴 합니다.
- `coachRetargeted` 는 **리타겟팅이 끝난** 코치, 즉 사용자 체형(뼈 길이)으로 옮겨진 결과여야 합니다. 이게 서비스의 핵심 차별점입니다.

크기 감: 30fps × 4초 × 15관절 × 3 = 5,400 개 숫자. JSON 으로 수백 KB 정도. 지금은 괜찮고, 길어지면 나중에 압축이나 바이너리로 바꿉니다.

### 4-4. `alignment` — DTW 정렬 경로

```json
[[0, 0], [1, 1], [2, 1], [3, 2], "..."]
```

`[사용자 프레임 번호, 코치 프레임 번호]` 쌍의 목록. 3D 뷰어에서 사용자 프레임 i 를 보여줄 때 어떤 코치 프레임을 겹쳐 그릴지 여기서 찾습니다. CP-DTW 가 만든 경로를 그대로 주면 됩니다. 짝이 없는 사용자 프레임은 빠져 있어도 되고(프론트가 직전 코치 프레임을 재사용), 한 사용자 프레임에 코치 프레임이 여럿 매핑되면 첫 번째를 씁니다.

### 4-5. `feedback` — LLM 피드백 (형식 미정)

```json
{
  "summary": "전체 흐름은 좋아요. 바닥에서 멈추는 구간이 빠졌고, 내려갈 때 상체가 많이 숙여져요.",
  "items": [
    { "segmentId": "seg-1", "text": "하강이 코치보다 1.2배 느려요. 가슴을 들고 시선을 앞에 두세요." },
    { "text": "전체적으로 ..." }
  ]
}
```

- `summary`: 한두 문장. `items[].text`: 구간별 코멘트. `segmentId` 가 있으면 "하강 구간 보기" 링크가 붙어서 누르면 뷰어가 그 구간으로 갑니다.
- **LLM 단계가 아직 없으면 `feedback` 을 통째로 빼세요.** 프론트가 "피드백이 들어올 자리예요" 라고 표시합니다.
- 이 형식은 가정입니다. LLM 출력이 다른 모양이면 그쪽에 맞춰 프론트를 고치겠습니다.

---

## 5. 관절 이름 · 뼈 이름 (프론트가 아는 이름)

프론트는 아래 **관절 이름** 쌍으로 뼈를 긋습니다. `jointNames` 에 이 이름이 있으면 뼈가 이어지고, 없으면 그 뼈는 건너뜁니다 (점만 찍힘). MediaPipe · COCO 등 어떤 모델을 쓰든 **출력 관절 이름을 이걸로 바꿔서** 보내 주세요. `neck` 과 `pelvis` 는 모델에 없으면 양 어깨 · 양 골반의 중점으로 만들어 주세요.

| 뼈 이름 (`deviations[].bone`) | 잇는 관절 | 화면 표시 |
| --- | --- | --- |
| `torso` | `pelvis` → `neck` | 몸통 |
| `head` | `neck` → `nose` | 머리 |
| `left_clavicle` / `right_clavicle` | `neck` → `left_shoulder` / `right_shoulder` | 왼쪽 · 오른쪽 어깨선 |
| `left_upper_arm` / `right_upper_arm` | `left_shoulder` → `left_elbow` / 오른쪽 동일 | 위팔 |
| `left_forearm` / `right_forearm` | `left_elbow` → `left_wrist` / 오른쪽 동일 | 아래팔 |
| `left_pelvis` / `right_pelvis` | `pelvis` → `left_hip` / `right_hip` | 골반 |
| `left_thigh` / `right_thigh` | `left_hip` → `left_knee` / 오른쪽 동일 | 허벅지 |
| `left_shin` / `right_shin` | `left_knee` → `left_ankle` / 오른쪽 동일 | 정강이 |

필요한 관절 이름 15개: `nose, neck, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, pelvis, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle`

관절을 더 보내도 됩니다 (손 · 발 끝 등). 뼈 정의는 `frontend/src/utils/skeleton.ts` 의 `BONES` 에 있고, 뼈를 추가하고 싶으면 거기에 한 줄만 넣으면 됩니다.

---

## 6. 기본 제공 코치 영상 (미리 계산해 둘 것)

프론트가 `coachId` 로 보내는 값입니다. 백엔드는 이 id 에 해당하는 코치 영상을 갖고 있고, **스켈레톤을 미리 추출해 저장해 두면** 매 요청마다 코치 쪽 추출을 건너뛸 수 있습니다.

| `coachId` | 종목 | 제목 |
| --- | --- | --- |
| `squat-highbar-full` | squat | 하이바 스쿼트 |
| `squat-lowbar` | squat | 로우바 스쿼트 |
| `deadlift-conventional` | deadlift | 컨벤셔널 데드리프트 |
| `deadlift-sumo` | deadlift | 스모 데드리프트 |
| `bench-flat` | bench_press | 플랫 벤치프레스 |

이 목록은 `frontend/src/constants/coachVideos.ts` 에 있고 **임시**입니다. 실제로 확보한 코치 영상에 맞춰 id · 제목을 바꿔 주시면 프론트 상수만 같이 고치면 됩니다. 나중에는 `GET /coach-videos` 로 백엔드에서 목록을 내려주는 쪽이 맞지만 MVP 에선 상수로 갑니다.

---

## 7. 해야 할 일 체크리스트

순서대로 하면 프론트와 가장 빨리 붙습니다.

**1단계 — 껍데기 (하루면 됨, 분석 없이도 프론트 전체 흐름 확인 가능)**
- [ ] `POST /analysis`: 파일 저장하고 `jobId` 반환
- [ ] `GET /analysis/{jobId}`: 시간이 지나면 `queued → extracting → … → done` 으로 바뀌는 가짜 상태 + `frontend/src/mocks/analysisResult.ts` 와 같은 모양의 고정 `result`
- [ ] CORS 허용
- [ ] 프론트 `.env` 에 `VITE_API_BASE_URL=http://서버주소:포트` 넣고 전체 흐름이 도는지 확인

**2단계 — 진짜 파이프라인**
- [ ] `extract(video) → SkeletonSequence`: 관절 이름을 5절로 맞추기. **중간 산출물(스켈레톤 JSON)을 파일로 저장** — 알고리즘 튜닝 때 재추출을 피하기 위해
- [ ] 기본 코치 영상 5개 스켈레톤 미리 추출해서 저장
- [ ] `retarget(coachSkeleton, studentSkeleton) → coachRetargeted`
- [ ] `score(student, coachRetargeted) → score, segments, alignment` (CP-DTW + 체크포인트 판정)
- [ ] 각 단계 사이에 `status` 갱신
- [ ] 실패 시 `status = failed` + 사용자용 `error` 문장 (사람 미검출, 영상 손상, 시간 초과 등)

**3단계 — 나중**
- [ ] `llm` 단계와 `feedback`
- [ ] 영상 · 작업 레코드 보관 기간, 정리 정책
- [ ] 결과 JSON 이 커지면 압축 또는 스켈레톤을 별도 엔드포인트로 분리

---

## 8. 아직 합의 안 된 것 (`TODO(backend)`)

프론트 코드에 `TODO(backend)` 로 표시된 곳입니다. 결정되면 알려 주세요.

| 항목 | 프론트의 현재 가정 |
| --- | --- |
| 엔드포인트 경로 · 필드 이름 | `POST /analysis`, `GET /analysis/{jobId}`, 위 필드 이름 |
| `status` 단계 이름 · 순서 | `queued, extracting, retargeting, scoring, llm, done, failed` |
| 관절 세트 (MediaPipe 33 / COCO 17 / SMPL 24 …) | 이름만 5절에 맞추면 무엇이든 됨 |
| 좌표계 · 단위 | 미터, y 위 |
| 점수 산식 (DTW 비용 → 0~100) | 백엔드가 정함 |
| LLM 피드백 형식 | `{ summary, items[{ segmentId?, text }] }` |
| 업로드 제한 | 100MB, 30초, mp4 / mov / webm |
| 기본 코치 영상 id · 썸네일 · 길이 | 6절 표 |

---

## 9. 프론트에 붙여서 테스트하는 법

1. `frontend/.env` 파일을 만들고 (`.env.example` 복사) `VITE_API_BASE_URL=http://localhost:8000` 처럼 백엔드 주소를 적습니다. 끝에 `/` 는 빼세요.
2. `cd frontend && pnpm install && pnpm dev` → 폰 또는 브라우저에서 `http://localhost:5173`
3. 브라우저 개발자 도구(F12) → Network 탭에서 `analysis` 요청을 보면 프론트가 보내는 body 와 서버 응답을 그대로 볼 수 있습니다.
4. `VITE_API_BASE_URL` 을 비우면 다시 mock 으로 돌아갑니다.

프론트 쪽에서 참고할 파일 (여기만 보면 됩니다):

| 파일 | 내용 |
| --- | --- |
| `frontend/src/types/analysis.ts` | API 계약 타입. 이 문서의 원본 |
| `frontend/src/api/analysis.ts` | 실제로 요청을 보내는 코드. 어떤 URL · 필드를 쓰는지 |
| `frontend/src/mocks/analysisApi.ts` | 가짜 서버. 백엔드가 흉내 내야 할 동작 그대로 |
| `frontend/src/mocks/analysisResult.ts` | 가짜 결과 데이터. `result` JSON 의 실제 예시를 만드는 코드 |

---

## 부록 — `GET /analysis/{jobId}` 완료 응답 전체 예시

```json
{
  "jobId": "0f3a9c1e",
  "status": "done",
  "progress": 1,
  "result": {
    "jobId": "0f3a9c1e",
    "exercise": "squat",
    "score": {
      "total": 72,
      "bySegment": { "seg-0": 96, "seg-1": 71, "seg-2": 38, "seg-3": 68, "seg-4": 93 }
    },
    "segments": [
      {
        "id": "seg-0",
        "label": "시작 자세",
        "verdict": "match",
        "studentRange": [0.0, 0.3],
        "coachRange": [0.0, 0.3],
        "timing": { "label": "ok", "ratio": 1.0 },
        "deviations": []
      },
      {
        "id": "seg-1",
        "label": "하강",
        "verdict": "match",
        "studentRange": [0.3, 1.5],
        "coachRange": [0.3, 1.3],
        "timing": { "label": "slow", "ratio": 1.2 },
        "deviations": [
          { "bone": "torso", "angleDiffDeg": 15.9 },
          { "bone": "left_shin", "angleDiffDeg": 6.1 }
        ]
      },
      {
        "id": "seg-2",
        "label": "바닥 유지",
        "verdict": "missing",
        "studentRange": [1.5, 1.67],
        "coachRange": [1.3, 1.8],
        "deviations": []
      },
      {
        "id": "seg-3",
        "label": "상승",
        "verdict": "match",
        "studentRange": [1.67, 3.17],
        "coachRange": [1.8, 2.8],
        "timing": { "label": "slow", "ratio": 1.5 },
        "deviations": [
          { "bone": "torso", "angleDiffDeg": 12.4 },
          { "bone": "left_thigh", "angleDiffDeg": 8.2 }
        ]
      },
      {
        "id": "seg-4",
        "label": "마무리 자세",
        "verdict": "match",
        "studentRange": [3.17, 3.43],
        "coachRange": [2.8, 3.1],
        "timing": { "label": "ok", "ratio": 1.0 },
        "deviations": []
      }
    ],
    "student": {
      "jointNames": [
        "nose", "neck", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "pelvis", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
      ],
      "fps": 30,
      "frames": [
        [[0.05, 1.62, 0.0], [0.0, 1.46, 0.0], [0.0, 1.44, 0.19], [0.0, 1.44, -0.19], "... 15개 ..."],
        "... 프레임 수만큼 (예: 103개) ..."
      ]
    },
    "coachRetargeted": {
      "jointNames": ["... student 와 같은 이름 ..."],
      "fps": 30,
      "frames": ["... 코치 프레임 수만큼 (예: 93개) ..."]
    },
    "alignment": [[0, 0], [1, 1], [2, 2], "... [사용자프레임, 코치프레임] ..."],
    "feedback": {
      "summary": "전체 동작 흐름은 좋아요. 바닥에서 잠깐 멈추는 구간이 빠졌고, 내려갈 때 상체가 코치보다 많이 숙여져요.",
      "items": [
        { "segmentId": "seg-1", "text": "하강이 코치보다 1.2배 느려요. 가슴을 들고 시선을 앞에 두면 몸통 각도가 잡혀요." },
        { "segmentId": "seg-2", "text": "바닥에서 바로 튕기듯 올라와요. 한 박자 멈췄다가 올라오는 연습을 해 보세요." },
        { "segmentId": "seg-3", "text": "상승이 코치보다 1.5배 느리고, 왼쪽 허벅지 각도가 8° 차이 나요." }
      ]
    }
  }
}
```
