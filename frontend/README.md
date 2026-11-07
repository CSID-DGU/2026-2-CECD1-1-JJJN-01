# frontend

리타겟팅 기반 자세 비교 코칭 — 웹 프론트엔드 (MVP)

React 19 · TypeScript · Vite · Tailwind CSS v4 · react-router v7 · three.js (react-three-fiber)

## 실행

```bash
cd frontend
pnpm install
pnpm dev         # http://localhost:5173
pnpm typecheck   # tsc -b --noEmit
pnpm lint        # eslint
pnpm format      # prettier --write
pnpm build
```

백엔드 없이 바로 돌아간다. `.env.example` 을 `.env` 로 복사해서 `VITE_API_BASE_URL` 을 채우면 실제 API 를 부르고,
비워 두거나 `VITE_USE_MOCK=true` 면 `src/mocks/` 의 가짜 응답으로 동작한다.

## 화면 흐름

| 경로         | 화면                                                                       |
| ------------ | -------------------------------------------------------------------------- |
| `/`          | 종목 선택 (스쿼트 / 데드리프트 / 벤치프레스)                               |
| `/upload`    | 내 영상 업로드 + 코치 영상(직접 올리기 또는 기본 제공) + 촬영 가이드       |
| `/analyzing` | 분석 대기. 백엔드가 단계별 상태를 주면 진행 단계 표시                      |
| `/result`    | 점수 · 3D 뷰어(나 vs 리타겟팅된 코치) · 구간 타임라인 · 구간 상세 · 피드백 |

## 폴더 구조

```
src/
  app/          라우터, 전역 Provider
  pages/        화면 하나 = 폴더 하나. 그 화면에서만 쓰는 컴포넌트는 안쪽 components/ 에
    ExerciseSelect/
    Upload/
    Analyzing/
    Result/
  components/   두 화면 이상에서 쓰는 공통 UI (Button, PageHeader, OptionCard …)
  hooks/        상태가 있는 로직 (분석 폴링, 영상 파일 검사, 프레임 재생)
  context/      한 번의 분석 흐름이 공유하는 상태
  api/          fetch 함수. 컴포넌트는 fetch 를 직접 하지 않는다
  types/        API 계약 타입. analysis.ts 가 핵심
  mocks/        백엔드 없이 돌리기 위한 가짜 API / 결과
  utils/        React 를 모르는 순수 함수
  constants/    종목, 코치 영상 목록, 업로드 제한, 경로
  styles/       Tailwind 진입점 + 디자인 토큰
```

규칙 세 가지:

1. 페이지 폴더가 자기 컴포넌트를 소유한다. 두 페이지 이상에서 쓰게 되는 순간 `src/components/` 로 올린다.
2. `utils/` 는 순수 함수만. 상태·훅이 필요하면 `hooks/`.
3. `api/` 는 fetch 만, `types/` 는 타입만. 백엔드 스키마가 바뀌면 이 두 폴더(+ `mocks/`)만 고친다.

## 백엔드와 맞출 것

`src/types/analysis.ts` 가 프론트–백엔드 계약이다. `TODO(backend)` 주석이 붙은 곳이 아직 합의 전이다.

- `POST /analysis` (multipart: `video`, `exercise`, `coachId | coachVideo`) → `{ jobId }`
- `GET /analysis/{jobId}` → `{ status, progress?, result?, error? }`
- 결과에 필요한 것: 총점, 구간(체크포인트)별 판정·타이밍·차이 큰 뼈, 사용자·코치(리타겟팅) 관절 시퀀스, DTW 정렬 경로, LLM 피드백

관절 세트는 이름 기준으로 그린다 (`src/utils/skeleton.ts`). 백엔드가 어떤 순서·개수로 보내든 `jointNames` 의 이름만
`BONES` 정의와 맞으면 뼈가 이어진다.

## 색 규칙

코치 = 파랑, 나 = 주황. 뷰어·타임라인·범례 어디서나 같다. 타이밍은 PDF 와 같이 빠름 = 빨강, 느림 = 파랑.
토큰은 `src/styles/index.css` 의 `@theme` 에 있다.
