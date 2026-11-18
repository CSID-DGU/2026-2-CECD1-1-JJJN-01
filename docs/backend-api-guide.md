# 백엔드 ↔ 프론트 API 연동 정리

작성일 2026-11-07. 프론트는 mock 으로 끝까지 동작하는 상태이고, 백엔드가 아래 **두 개의 HTTP API** 를 만들어 주면 붙습니다.
계약 원본은 `frontend/src/types/analysis.ts` 입니다. 구조를 바꾸고 싶으면 바꿔도 됩니다. 프론트에서 고칠 파일이 세 개뿐이라 부담이 작습니다. 바꾸기 전에 한 번만 이야기해 주세요.

---

## 한 장 요약

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
(분석이 확실히 30초 안에 끝나고 시연용이면 하나로 합쳐도 됩니다. 그 경우 `POST /analysis` 가 아래 `result` 를 바로 돌려주면 되고, 폴링 API 는 없어집니다.)

---

## 백엔드가 프론트에게 넘겨줘야 하는 것

아래 표의 "프론트 가정" 은 지금 프론트가 그렇게 돌아간다는 뜻입니다. 같으면 "그대로", 다르면 실제 값을 알려 주세요.

| # | 넘겨줄 것 | 프론트 가정 |
| --- | --- | --- |
| 1 | **서버 주소** (예 `http://1.2.3.4:8000`) | 프론트 `.env` 의 `VITE_API_BASE_URL` 에 넣음 |
| 2 | **CORS 허용 여부** | 프론트 주소(`http://localhost:5173` 등)에서 오는 요청을 허용해야 함. 개발 중엔 전체 허용 |
| 3 | **분석 요청 API** 경로 · 메서드 | `POST /analysis`, body 는 `multipart/form-data` |
| 4 | 요청 필드 이름 | `video`(파일, 필수) · `exercise`(`squat` / `deadlift` / `bench_press`) · `coachId`(문자열) 또는 `coachVideo`(파일) 중 하나 |
| 5 | 요청 응답 | `{ "jobId": "문자열" }` 을 즉시 반환 |
| 6 | **상태 조회 API** 경로 | `GET /analysis/{jobId}` |
| 7 | `status` 값 목록과 순서 | `queued → extracting → retargeting → scoring → llm → done`, 실패 시 `failed`. 단계를 건너뛰어도 됨 |
| 8 | 완료 응답 | `status: "done"` 일 때 `result` 필드에 결과 JSON (아래 참고) |
| 9 | 실패 응답 | `status: "failed"` + `error: "사용자에게 보여줄 문장"` |
| 10 | HTTP 에러 형식 | 4xx / 5xx 일 때 `{ "message": "사용자에게 보여줄 문장" }`. 프론트가 그대로 화면에 띄움 |
| 11 | **관절 이름** | 아래 15개. 다른 이름을 쓰면 목록을 주세요 |
| 12 | 좌표계 · 단위 | 미터, y 가 위, 발이 y≈0 |
| 13 | **기본 코치 영상 id 목록** | 아래 5개 (임시). 실제 확보한 영상에 맞춰 알려 주세요 |
| 14 | 업로드 제한 | 100MB, 30초, mp4 / mov / webm. 서버 기준이 다르면 알려 주세요 |
| 15 | 점수 산식 | DTW 비용 → 0~100 정수. 프론트는 숫자만 받음 (80 이상 초록 / 60 이상 노랑 / 그 아래 빨강) |
| 16 | LLM 피드백 형식 | 아직 없으면 `feedback` 을 빼고 보내면 됨. 생기면 모양을 알려 주세요 |

---

## `result` JSON 모양

```
result
├── jobId            문자열
├── exercise         "squat" | "deadlift" | "bench_press"
├── score            { total: 0~100, bySegment?: { 구간id: 점수 } }
├── segments[]       구간 목록 (아래)
├── student          사용자 스켈레톤 { jointNames[], fps, frames[t][j] = [x, y, z] }
├── coachRetargeted  사용자 체형으로 리타겟팅된 코치 스켈레톤 (같은 형식)
├── alignment        DTW 경로 [[사용자프레임, 코치프레임], ...]
└── feedback?        { summary, items[{ segmentId?, text }] }   ← 없으면 생략
```

구간 하나 (`segments[]`):

```json
{
  "id": "seg-1",
  "label": "하강",
  "verdict": "match",
  "studentRange": [0.3, 1.5],
  "coachRange": [0.3, 1.3],
  "timing": { "label": "slow", "ratio": 1.2 },
  "deviations": [{ "bone": "torso", "angleDiffDeg": 15.9 }]
}
```

- `verdict`: `match`(일치) / `missing`(누락) / `extra`(추가) / `substitute`(대체)
- `studentRange` · `coachRange`: 각 영상에서의 [시작초, 끝초]. 누락이면 `studentRange`, 추가면 `coachRange` 생략 가능
- `timing`: 일치 구간만. `label` 은 `ok` / `slow` / `fast`, `ratio` = 사용자 시간 ÷ 코치 시간
- `deviations`: 차이 큰 뼈 상위 몇 개. 없으면 `[]`. `bone` 은 아래 뼈 이름
- 구간은 시간순 정렬

스켈레톤 (`student`, `coachRetargeted`):

```json
{
  "jointNames": ["nose", "neck", "left_shoulder", "..."],
  "fps": 30,
  "frames": [[[0.01, 1.52, 0.0], [0.0, 1.38, 0.0], "..."], "..."]
}
```

`jointNames` 의 순서 · 개수는 자유입니다. 프론트는 이름만 보고 뼈를 잇습니다.

완전한 예시는 `frontend/src/mocks/analysisResult.ts` 가 만들어 내는 값과 같습니다. 1단계로 이 모양의 고정 JSON 을 돌려주는 가짜 서버만 만들어도 프론트 전체 흐름을 붙여 볼 수 있습니다.

---

## 관절 이름 15개 · 뼈 이름

관절: `nose, neck, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, pelvis, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle`
(`neck` · `pelvis` 가 모델에 없으면 양 어깨 · 양 골반의 중점으로 만들어 주세요)

뼈 (`deviations[].bone`): `torso, head, left_clavicle, right_clavicle, left_upper_arm, right_upper_arm, left_forearm, right_forearm, left_pelvis, right_pelvis, left_thigh, right_thigh, left_shin, right_shin`
(정의는 `frontend/src/utils/skeleton.ts` 의 `BONES`)

---

## 기본 코치 영상 id (임시)

`squat-highbar-full`, `squat-lowbar`, `deadlift-conventional`, `deadlift-sumo`, `bench-flat`
(`frontend/src/constants/coachVideos.ts`. 백엔드는 이 id 의 코치 스켈레톤을 미리 추출해 두면 됩니다)

---

## 붙여서 테스트

1. `frontend/.env` 에 `VITE_API_BASE_URL=http://서버주소:포트` (끝에 `/` 없이)
2. `cd frontend && pnpm install && pnpm dev` → `http://localhost:5173`
3. 브라우저 개발자 도구 Network 탭에서 `analysis` 요청의 body 와 응답을 그대로 볼 수 있음
4. `VITE_API_BASE_URL` 을 비우면 다시 mock

프론트에서 볼 파일: `types/analysis.ts`(계약) · `api/analysis.ts`(요청 코드) · `mocks/analysisApi.ts`(백엔드가 흉내 낼 동작) · `mocks/analysisResult.ts`(결과 예시)
