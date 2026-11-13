# backend — 분석 API 서버

프론트(`frontend/`)와 붙는 HTTP API. 계약은 `frontend/src/types/analysis.ts`,
설명은 `docs/backend-api-guide.md`. 분석 엔진은 `engine/pipeline/` (같은 저장소).

## 환경 구성 (처음 한 번)

저장소 루트에 가상환경 3개와 WHAM 체크아웃이 있어야 한다. 코드는 git 에 있지만 아래는 git 에 없다 (`.gitignore`).

| 경로 | 내용 | 만드는 법 |
|---|---|---|
| `.venv` | 백엔드 + 엔진 (PAN · IK · 채점). 시스템 torch 2.8 cu128 을 공유 | `bash scripts/setup_venvs.sh backend` |
| `.venv-pose` | MediaPipe 추출기 | `bash scripts/setup_venvs.sh pose` |
| `.venv-wham` | WHAM 추출기 (자체 torch, 약 11GB) | `third_party/WHAM` 을 먼저 둔 뒤 `bash scripts/setup_venvs.sh wham` |
| `third_party/WHAM` | WHAM 코드 + 체크포인트(6GB) + SMPL 모델 | 기존 서버에서 통째로 복사 (`checkpoints/`, `dataset/body_models/` 포함) |
| `third_party/pan-motion-retargeting/data_preprocess/Mixamo/Mixamo/*.npy` | PAN 학습 데이터 (150MB). priors 재생성 · `engine/tests` · `pipeline.harness` 에만 쓰임, 서버 실행에는 불필요 | 기존 서버에서 복사 (`PAN_TRAIN_NPY_DIR` 로 위치 변경 가능) |
| `engine/pretrained_mixamo`, `engine/data_preprocess/Mixamo/Mixamo`, `engine/runs`, `engine/pipeline/data` | PAN 가중치 · Mixamo 데이터 · 코치 사전계산 결과 · 테스트 영상 | 기존 서버에서 복사 |
| `backend/.env` | `GEMINI_API_KEY=...` | 직접 작성 |

위치를 바꾸고 싶으면 환경변수로 덮어쓴다: `PIPELINE_REPO`, `PIPELINE_PYTHON` (backend), `PIPELINE_POSE_PYTHON`,
`PIPELINE_WHAM_PYTHON`, `WHAM_DIR` (engine). 기본값은 모두 저장소 루트 기준 상대 위치다.

## 실행

```bash
cd backend
../.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
# 분석 없이 프론트만 붙여 볼 때: ANALYSIS_RUNNER=demo 를 앞에 붙인다
```

프론트: `frontend/.env` 에 `VITE_API_BASE_URL=http://<서버주소>:8000` (끝 `/` 없이) 후 `pnpm dev`.

테스트 (브라우저 없이 계약 검사):
```bash
../.venv/bin/python tests/test_api.py
```

## 현재 단계: 2단계 (실제 분석)

- `POST /analysis` 는 업로드를 검사·저장하고 `jobId` 를 바로 돌려준다 (202).
  검사: 종목(지금은 스쿼트만), `coachId` / `coachVideo` 중 하나, 형식(mp4/mov/webm), 100MB, 30초, ffprobe 로 읽히는지.
- 워커 1개가 작업을 순서대로 하나씩 실제로 분석한다 (`app/pipeline_runner.py`).

  | status | 하는 일 | 시간 (사용자 영상 1개) |
  |---|---|---|
  | `extracting` | ffmpeg 정규화(H.264 · 30fps · 회전 적용 · 긴 변 1920) → `pipeline.run_video` (WHAM → IK) | 2–3분 (30초 영상이면 5분 이상) |
  | `retargeting` | `pipeline.coach_to_user` (PAN) | 약 10초 |
  | `scoring` | motion_scoring + 결과 변환 | 1초 미만 |
  | `llm` | Gemini 로 한국어 피드백 생성 (`app/feedback.py`) | 5–10초 (과부하 시 대체 모델로 최대 90초) |

  코치 영상을 직접 올리면 코치도 같은 추출을 거쳐 `extracting` 이 두 배가 된다.
  `progress` 는 단계 표시(`[S0]`, `[S1]` …)로 대략 올린다.
- 실패하면 `status: failed` + 사용자용 `error` 문장: 사람 미검출, 사람이 보이는 시간이 짧음, 영상 손상,
  시간 초과, GPU 메모리 부족. 그 밖의 오류는 일반 문장 + `data/jobs/<jobId>/error.log`.
- `ANALYSIS_RUNNER=demo` 로 띄우면 분석 없이 기존 결과(`runs/coach_user/coach_to_student`)를 돌려준다 (프론트 작업용).
- 작업 폴더 `data/jobs/<jobId>/`: `upload_user.*`(원본), `user.mp4`(정규화), `user_run/`(run_video),
  `retarget/`(coach_to_user), `logs/*.log`, `job.json`, `result.json`, 실패 시 `error.log`.
- 작업 기록은 메모리에 있어서 서버를 재시작하면 진행 중이던 작업 조회는 404 가 된다.
- 요청 · 응답 기록 (`app/api_log.py`): `data/api.log`(모든 요청, 거절 포함, JSON 한 줄씩),
  `data/jobs/<jobId>/logs/api.log`(그 작업의 POST · 상태 변화), `data/jobs/<jobId>/api_response.json`(완료 시 프론트가 받은 응답 원본).
  POST 는 받은 항목과 파일 정보(이름 · 형식 · 크기 · 길이)만 남기고 영상은 남기지 않는다. GET 은 상태가 바뀔 때만 남긴다.
  서버 콘솔에도 한 줄 요약이 찍힌다. 테스트(`tests/test_api.py`)가 남긴 줄은 client 가 `testclient`.

테스트:
```bash
../.venv/bin/python tests/test_api.py            # 계약 · 거부 케이스 (demo, 몇 초)
../.venv/bin/python tests/test_pipeline_e2e.py   # 실제 분석: child_squat 성공 + 사람 없는 영상 실패 (약 4분)
```

## LLM 피드백 (`llm` 단계)

- `backend/.env` 에 `GEMINI_API_KEY=...` (Google AI Studio 키). 서버 시작 시 읽는다. `.env` 는 git 제외.
- 입력: 결과의 구간 정보(id, 이름, 판정, 시간, timing, 차이 큰 부위와 방향) + `poseMetrics`(관절 각도, 반복별 최저점, 범위)
  + 전체 템포, 거울 여부, 누락 · 추가 구간의 코치 · 사용자 움직임.
- 프롬프트: 수치를 근거로 쓰고, 반복되는 차이는 범위로 묶고, 최대 6개. 스쿼트는 깊이 · 상체 · 엉덩이 · 무릎 간격 · 속도를 먼저,
  팔은 마지막. `POSE_SAME_DEG`(8°) · `POSE_SAME_LEG`(다리 길이의 0.035배, 85 cm 다리면 3 cm) · `POSE_SAME_RATIO`(0.15)
  미만 차이는 말하지 않는다 (초안 값). 실제 기준값은 `poseMetrics.thresholds` 로 결과에 함께 실린다.
  골격 좌표 · 영상 · 점수(자리표시 0)는 보내지 않는다.
- 출력: JSON 스키마(`Feedback`)로 강제 → `{ summary, items[{ segmentId?, text }] }`. 없는 구간 id 는 지운다.
- 모델: `GEMINI_MODELS` 순서대로 시도 (기본 `gemini-3.5-flash, gemini-3.8-flash, gemini-3.1-flash-lite`).
  과부하(503) · 한도(429) · 시간 초과 · 모델 폐지(404) 면 다음 모델. 시도당 `LLM_TIMEOUT_SEC`(40), 전체 `LLM_DEADLINE_SEC`(90).
- 실패하면(키 없음 포함) `feedback` 만 빼고 작업은 `done`. 입력 · 응답 · 오류는 `data/jobs/<jobId>/logs/llm.log`.
- `LLM_PROVIDER=none` 이면 단계를 건너뛴다.

## 파일

| 파일 | 역할 |
|---|---|
| `app/main.py` | 엔드포인트, CORS(전체 허용), 업로드 검사, 오류 → `{ "message" }` |
| `app/jobs.py` | 작업 레코드, GPU 워커 1개(순서대로 1건씩), demo 러너 |
| `app/pipeline_runner.py` | 실제 분석 러너: 정규화 → 추출 → 리타겟팅 → 채점, 실패 → 사용자 문장 |
| `app/result_builder.py` | 파이프라인 결과 → `AnalysisResult` JSON |
| `app/pose_metrics.py` | 관절 각도 · 반복별 최저점 · 뼈 차이 방향 (`poseMetrics`, `deviations[].direction`) |
| `app/feedback.py` | LLM 피드백: 입력 요약, 프롬프트, Gemini 호출(모델 대체), 출력 검증 |
| `app/config.py` | 경로, 업로드 제한, 코치 영상 목록(`COACHES`), 점수 기본값 |

## 결과 변환 규칙 (`result_builder.py`)

- `student` / `coachRetargeted`: `user.bvh` / `coach_on_user_pan.bvh` → 공식 FK → 프론트 15관절 이름,
  미터, y 위, 바닥 y≈0, 첫 프레임 골반을 x=z=0 으로 옮김 (겹쳐 보기). `nose` 는 BigVegas 의 Head 관절.
- `segments`: motion_scoring 구간 그대로. `structure` → `verdict` (`substitution` → `substitute`),
  `timing` 은 motion_scoring 값 그대로(비율은 전체 템포 대비 상대값, 판정 없으면 `ok`, 1.0),
  `pose.worst` → `deviations` (뼈 이름을 프론트 이름으로). `angleDiffDeg` 는 뼈 방향 사이 각도(항상 0 이상)이고,
  `direction` / `components` 가 어느 쪽으로 다른지를 붙인다 (`app/pose_metrics.py`, 아래).
- `label`: 스쿼트 기준. 구간 동안 골반 높이 변화로 하강 / 상승 / 하강 · 상승 / 바닥 유지 /
  시작 자세 / 마무리 자세 / 선 자세 유지.
- `alignment`: motion_scoring DTW 경로(drop 제외) → `[사용자 프레임, 코치 프레임]`.
- `frameMap` (`app/frame_map.py`): 사용자 프레임마다 그릴 코치 프레임(소수)과 상태(together · coachAhead · coachWaits ·
  userOnly · outside). 짝이 여러 개면 범위의 가운데, 정렬 범위 밖은 클립 처음 · 끝과 직선으로 잇고, 9프레임 이동 평균으로
  건너뜀을 나눈다(항상 앞으로만 진행). 프론트는 있으면 이것을 쓰고 소수 프레임은 두 프레임을 섞어 그린다.
- 좌우 반전으로 채점된 경우(`mirrored`) 코치를 좌우 반전해서 보내고, 뼈 이름을 사용자 기준 좌우로 바꾼다.
- `poseMetrics` (`app/pose_metrics.py`, 채점 모듈은 그대로): 같은 SMPL-22 골격에서 직접 잰 관절 측정.
  - 몸 기준 좌표: x = 사람의 오른쪽(엉덩이 축), y = 위, z = 앞(발끝 방향).
  - 값: 무릎 굽힘 · 엉덩이 굽힘(좌우), 상체 숙임, 엉덩이 − 무릎 높이(cm), 골반 내려간 거리(cm), 무릎 간격 ÷ 발목 간격.
  - `reps`: 골반 높이 최저점(반복마다)을 DTW 정렬로 코치 최저점과 짝지음. 짝이 없으면 한쪽 null (반복 수 차이).
  - `standing`(선 자세 중앙값), `bottomSummary`(짝 맞은 반복들의 범위), `repCount`,
    `legLengthCm`(사용자 다리 길이), `thresholds`(말하지 않을 작은 차이 기준. 길이는 다리 길이 비율).
  - 길이 기준은 모두 다리 길이 비율: 최저점 최소 깊이 7 %, 선 자세 범위 3.5 %.
  - 코치가 사용자 체형으로 리타겟팅된 골격이라 cm 를 그대로 비교한다. `mirrored` 면 코치 좌우를 바꿔 맞춘다.
- `deviations[].direction`: 구간의 정렬된 프레임 쌍에서 (사용자 뼈 − 코치 뼈) 평균을 몸 기준 위 · 앞 · 바깥쪽 성분으로 나누고,
  제곱합이 `angleDiffDeg` 가 되도록 맞춘 값(`components`)의 가장 큰 성분.
- `score.total`: 산식 미정 → `config.DEFAULT_SCORE_TOTAL` (0). `feedback`: `llm` 단계 결과(실패 시 없음).

## 다음 단계

3단계: 모델 상주로 처리 시간 단축(지금은 작업마다 WHAM 모델 로딩 ~60초), 코치 영상 추가,
작업 기록 영속화 · 보관 기간 정리.
