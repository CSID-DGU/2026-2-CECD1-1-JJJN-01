/**
 * 분석 API 타입 스텁.
 *
 * 이 파일은 프론트와 백엔드 사이의 계약이다.
 * 지금은 프론트 화면을 만들기 위한 "임시 형태"이고, 필드 이름·구조는
 * 백엔드(추출·리타겟팅·스코어링) 담당이 실제 구현에 맞춰 채우거나 바꾼다.
 *
 * 바꿀 때 규칙:
 *  - 여기 타입을 바꾸면 src/mocks/* 와 src/api/analysis.ts 만 같이 고치면 된다.
 *    (컴포넌트는 이 타입만 보고 그리므로 그 외는 컴파일 에러가 알려준다)
 *  - `TODO(backend)` 주석이 붙은 곳이 아직 합의 안 된 부분이다.
 */

import type { ExerciseType } from './exercise';

/* ------------------------------------------------------------------ */
/* 요청                                                                */
/* ------------------------------------------------------------------ */

/**
 * POST /analysis (multipart/form-data)
 * - video: 사용자 영상 원본 (전처리 없이 그대로)
 * - exercise: 종목
 * - coachId 또는 coachVideo 중 하나
 */
export interface CreateAnalysisRequest {
  video: File;
  exercise: ExerciseType;
  /** 기본 제공 코치 영상을 고른 경우 */
  coachId?: string;
  /** 코치 영상을 직접 올린 경우 */
  coachVideo?: File;
}

export interface CreateAnalysisResponse {
  jobId: string;
}

/* ------------------------------------------------------------------ */
/* 작업 상태                                                           */
/* ------------------------------------------------------------------ */

/**
 * GET /analysis/{jobId}
 * 단계가 바뀔 때마다 status 가 진행한다. 프론트는 done/failed 까지 폴링한다.
 * TODO(backend): 단계 이름·순서 확정
 */
export type JobStatus =
  | 'queued'
  | 'extracting' // 스켈레톤 추출
  | 'retargeting' // 코치 동작 → 사용자 체형
  | 'scoring' // CP-DTW
  | 'llm' // 피드백 문장 생성
  | 'done'
  | 'failed';

export interface AnalysisJob {
  jobId: string;
  status: JobStatus;
  /** 0~1. 없어도 된다. */
  progress?: number;
  /** status === 'done' 일 때만 */
  result?: AnalysisResult;
  /** status === 'failed' 일 때 사용자에게 보여줄 메시지 */
  error?: string;
}

/* ------------------------------------------------------------------ */
/* 스켈레톤                                                            */
/* ------------------------------------------------------------------ */

/** [x, y, z] (미터 단위 가정, y 가 위) TODO(backend): 좌표계·단위 확정 */
export type Vec3 = [number, number, number];

/**
 * 한 사람의 관절 시퀀스.
 * frames[t][j] = t 번째 프레임의 j 번째 관절 위치.
 * jointNames[j] 로 관절 이름을 알 수 있고, 뼈 연결은 프론트(utils/skeleton.ts)가
 * 이름 기준으로 만든다. 관절 개수·순서가 바뀌어도 이름만 맞으면 그려진다.
 * TODO(backend): 관절 세트 확정 (MediaPipe 33 / COCO 17 / SMPL 24 …)
 */
export interface SkeletonSequence {
  jointNames: string[];
  fps: number;
  frames: Vec3[][];
}

/**
 * DTW 정렬 경로. [사용자 프레임, 코치(리타겟팅) 프레임] 쌍의 목록.
 * 뷰어에서 사용자 프레임 i 를 보여줄 때 어떤 코치 프레임을 겹칠지 여기서 찾는다.
 * 체크포인트 방식에서 짝 없이 건너뛴 사용자 프레임은 목록에 없을 수 있다.
 */
export type AlignmentPath = Array<[studentFrame: number, coachFrame: number]>;

/* ------------------------------------------------------------------ */
/* 결과                                                                */
/* ------------------------------------------------------------------ */

/** 체크포인트 구간 판정 (PDF "Checkpoint" 슬라이드의 일치·누락·추가·대체) */
export type SegmentVerdict = 'match' | 'missing' | 'extra' | 'substitute';

export type TimingLabel = 'ok' | 'slow' | 'fast';

/** 구간 안에서 차이가 컸던 뼈 하나 */
export interface BoneDeviation {
  /** 예: "torso", "left_thigh" — utils/skeleton.ts 의 뼈 이름과 맞춘다 */
  bone: string;
  /** 사용자 − 코치 각도 차이 (도) */
  angleDiffDeg: number;
}

/**
 * 체크포인트로 나뉜 구간 하나.
 * 예: 스쿼트 = 시작 rest → 하강 → 바닥 pause → 상승 → 종료 rest
 */
export interface Segment {
  id: string;
  /** 화면에 보이는 이름. 예: "하강", "바닥 유지" */
  label: string;
  verdict: SegmentVerdict;
  /** 사용자 영상의 원본 시각 [시작, 끝] (초). missing 이면 없을 수 있다. */
  studentRange?: [number, number];
  /** 코치 영상의 시각 [시작, 끝] (초). extra 면 없을 수 있다. */
  coachRange?: [number, number];
  /** match 구간에서만 */
  timing?: {
    label: TimingLabel;
    /** 사용자 소요 시간 ÷ 코치 소요 시간 (slow ×1.5 → 1.5) */
    ratio: number;
  };
  /** 차이가 큰 뼈 상위 몇 개. 없으면 빈 배열 */
  deviations: BoneDeviation[];
}

/**
 * LLM 피드백.
 * TODO(backend): 형식 미정. 지금은 "요약 + 구간별 코멘트" 로 가정.
 */
export interface LlmFeedback {
  summary: string;
  items: Array<{
    /** 어느 구간에 대한 코멘트인지. 없으면 전체 코멘트 */
    segmentId?: string;
    text: string;
  }>;
}

export interface AnalysisScore {
  /** 0~100. TODO(backend): DTW 비용을 점수로 바꾸는 식 확정 */
  total: number;
  /** 구간별 점수. 없으면 total 만 보여준다. */
  bySegment?: Record<string, number>;
}

export interface AnalysisResult {
  jobId: string;
  exercise: ExerciseType;
  score: AnalysisScore;
  segments: Segment[];
  /** 사용자 스켈레톤 */
  student: SkeletonSequence;
  /** 사용자 체형으로 리타겟팅된 코치 스켈레톤 */
  coachRetargeted: SkeletonSequence;
  alignment: AlignmentPath;
  /** LLM 단계가 아직 없으면 undefined */
  feedback?: LlmFeedback;
}
