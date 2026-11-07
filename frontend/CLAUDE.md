# frontend

React 19 · TypeScript · Vite · Tailwind CSS v4 · react-router v7 · three.js (@react-three/fiber + drei). 패키지 매니저는 pnpm.

```bash
pnpm install
pnpm dev         # http://localhost:5173
pnpm typecheck   # tsc -b --noEmit
pnpm lint        # eslint (lint:fix 로 자동 수정)
pnpm format      # prettier --write (format:check 로 검사만)
pnpm build
```

ESLint 는 `eslint.config.js`(flat), Prettier 는 `.prettierrc`. react-hooks 7 의 React Compiler 규칙이 켜져 있어서
effect 안에서 동기적으로 setState 하면 에러다 — "입력이 바뀌면 상태를 그 입력으로 키잉" 하는 식으로 푼다 (`hooks/useVideoFile.ts` 참고).
컴포넌트 파일은 컴포넌트만 export 한다 (Fast Refresh). context 객체 · 훅은 `context/analysisContextValue.ts`, `context/useAnalysis.ts` 처럼 따로 둔다.

백엔드 없이 돌아간다: `VITE_API_BASE_URL` 이 비어 있으면 `src/mocks/` 의 가짜 API 를 쓴다. (`.env.example` 참고)

## 폴더 규칙

```
src/
  app/          라우터, Provider
  pages/        화면 하나 = 폴더 하나 (index.tsx + components/)
  components/   두 화면 이상에서 쓰는 공통 UI
  hooks/        상태 있는 로직
  context/      분석 흐름 공유 상태 (AnalysisContext = Provider, useAnalysis = 훅)
  api/          fetch 함수만
  types/        API 계약. analysis.ts 가 핵심
  mocks/        가짜 API / 결과
  utils/        React 모르는 순수 함수
  constants/    종목, 코치 영상, 업로드 제한, 경로
  styles/       Tailwind 진입점 + @theme 토큰
```

1. 페이지 폴더가 자기 컴포넌트를 소유한다. 두 페이지 이상에서 쓰게 되면 그때 `src/components/` 로 올린다. 처음부터 공통으로 만들지 않는다.
2. `utils/` 는 순수 함수만. 상태 · 훅이 필요하면 `hooks/`.
3. `api/` 는 fetch 만, `types/` 는 타입만. 컴포넌트가 fetch 를 직접 하지 않는다. 백엔드 스키마가 바뀌면 `types/` + `api/` + `mocks/` 만 고친다.
4. import 는 `@/` 별칭을 쓴다 (`@/components/Button`).

## 화면 흐름

`/` 종목 선택 → `/upload` 내 영상 + 코치 영상 + 촬영 가이드 → `/analyzing` 단계별 진행 → `/result` 점수 · 3D 뷰어 · 구간 타임라인 · 구간 상세 · 피드백

결과 화면의 핵심 요소는 3D 뷰어(나 vs 리타겟팅된 코치)다. 타임라인 · 피드백에서 구간을 누르면 뷰어가 그 구간으로 이동한다.

## 백엔드 계약

`src/types/analysis.ts`. `TODO(backend)` 주석이 붙은 곳이 아직 합의 전이다. 관절은 이름 기준으로 뼈를 잇는다(`src/utils/skeleton.ts` 의 `BONES`) — 백엔드의 관절 순서 · 개수와 무관하게 이름만 맞추면 된다.

## 디자인

- 밝은 테마. 모바일 430px 단일 컬럼, 데스크톱에선 가운데 정렬. 하단 고정 CTA(`BottomBar`)
- 색 규칙은 어디서나 같다: 코치 = 파랑(`coach`), 나 = 주황(`me`), 빠름 = 빨강(`fast`), 느림 = 파랑(`slow`), 판정은 `good / warn / bad`
- 토큰은 `src/styles/index.css` 의 `@theme`. three.js 재질 색은 CSS 변수를 못 쓰므로 `SkeletonViewer` 에 같은 hex 를 적어둔다
- 아이콘 라이브러리 · 이모지 없이 간다 (`ExerciseMonogram` 처럼 글자 마크, 작은 인라인 SVG)
- 모든 것을 카드로 감싸지 않는다. 카드는 "고르는 것"(OptionCard)과 "상세"에만

## 현재 상태 (2026-11-07)

- 4개 화면 + mock 으로 끝까지 동작. `pnpm typecheck` · `pnpm lint` · `pnpm format:check` 통과
- `constants/dev.ts` 의 `SKIP_INPUT_GATES = true` 가 임시로 켜져 있다: 종목 · 영상 없이도 다음 화면으로 넘어간다. 실제 API 연결 전에 끈다
- 미확정: API 스키마, 촬영 가이드 문구(지금은 임시), LLM 피드백 형식, 점수 산식 · 표현
- 남은 일: 기본 코치 영상 실제 데이터, 실제 API 연결
