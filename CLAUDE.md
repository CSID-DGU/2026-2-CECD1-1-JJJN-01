# 2026-2-CECD1-1-JJJN-01

종합설계 팀 프로젝트 — 리타겟팅 기반 자세 비교 코칭 서비스. 모노레포.

- `frontend/` — 사용자 웹 (React). 자세한 규칙은 `frontend/CLAUDE.md`, `frontend/README.md`
- `backend/` — FastAPI 분석 서버 (업로드 · 작업 큐 · 결과 변환 · Gemini 피드백). `backend/README.md`
- `engine/` — 분석 엔진. PAN 리타게팅 원본 코드(수정 5건) + `pipeline/` (영상→스켈레톤(WHAM/MediaPipe)→IK→PAN→`motion_scoring`). `engine/pipeline/README.md`
- `docs/backend-api-guide.md` — API 2개와 결과 JSON 형식 (프론트 ↔ 백엔드 계약)
- 저장소 루트의 `.venv`, `.venv-pose`, `.venv-wham`, `third_party/WHAM` 과 `engine/` 안의 가중치 · 데이터 · `runs/` 는 git 에 없다.
  만드는 법은 `backend/README.md` "환경 구성", `scripts/setup_venvs.sh`. 서버는 `backend/` 에서 `../.venv/bin/uvicorn app.main:app` 으로 띄운다.

## 서비스 한 줄 설명

사용자 영상과 코치 영상에서 스켈레톤을 뽑고, 코치 동작을 사용자 체형으로 리타겟팅한 뒤
CP-DTW(체크포인트 기반 DTW)로 구간별 비교 → 점수 · 피드백 · 3D 오버레이를 보여준다.
기존 서비스와의 차별점은 "리타겟팅된 코치 vs 나" 비교다.

## 결정 사항 (2026-11-07)

- MVP 흐름: 종목 선택(스쿼트 / 데드리프트 / 벤치프레스) → 내 영상 업로드 → 분석 중 → 결과
- 영상은 프론트에서 전처리 없이 원본 그대로 서버로 보낸다. 실시간 분석은 나중에 파이프라인 일부를 프론트로 옮겨서 검토
- T-pose 입력은 받지 않는다
- 코치 영상: 직접 업로드가 우선, 올리지 않으면 기본 제공 영상 중 하나
- 결과 화면 3종: DTW 점수 / LLM 피드백(형식 미정, 자리만 확보) / 3D 스켈레톤 뷰어(three.js)
- 로그인 · 분석 기록 저장은 MVP 제외. 모바일 뷰 기준
- 백엔드는 GPU 서버 하나로 시작하되 `extract(video→skeleton)` / `retarget` / `score` 모듈 경계를 분리하고,
  스켈레톤 중간 산출물을 저장한다 (알고리즘 튜닝 때 재추출 방지, 코치 영상은 미리 계산)
- 작업 상태는 `queued → extracting → retargeting → scoring → llm → done | failed` 로 폴링
- 패키지 매니저는 pnpm. lockfile 은 `pnpm-lock.yaml` 하나만 커밋한다

## 브랜치 · 이슈

- `feat/#5/mvp-ui` ← 이슈 #5 "[Front] MVP 화면 구현" (https://github.com/CSID-DGU/2026-2-CECD1-1-JJJN-01/issues/5)
- 브랜치 이름: `feat/#<이슈번호>/<설명>`, 커밋 메시지: `feat: ...` / `fix: ...` 형식

## 참고 자료

- 지도교수 미팅 슬라이드(PDF): Main Figure, 리타겟팅 네트워크(VPC / PAN), CP-DTW, 체크포인트 판정(일치 · 누락 · 추가 · 대체, slow/fast)
- 유사 서비스 레퍼런스: https://app.coachonai.com — 모바일 단일 컬럼, 다크 테마 + 초록 포인트. 우리는 밝은 테마로 간다
