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
 *  - `[화면 미사용]` 이 붙은 필드는 백엔드가 보내 주지만 지금 화면을 그리는 데 쓰지 않는다.
 *    (응답 형식을 그대로 적어 둔 것. 쓰기 시작하면 표시를 뗀다)
 */

import type { ExerciseType } from './exercise';

/* ------------------------------------------------------------------ */
/* 요청                                                                */
/* ------------------------------------------------------------------ */

/**
 * POST /analysis (multipart/form-data)
 * - video: 사용자 영상 원본 (전처리 없이 그대로)
 * - exercise: 종목
 * - coachVideo: 코치 영상. 기본 제공 영상을 골라도 프론트가 그 파일을 그대로 올린다 (coachId 는 보내지 않는다)
 */
export interface CreateAnalysisRequest {
  video: File;
  exercise: ExerciseType;
  /** 직접 올린 영상, 또는 기본 제공 영상 파일 */
  coachVideo: File;
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
  /** [화면 미사용] 프론트는 요청할 때 받은 jobId 를 들고 있다 */
  jobId: string;
  status: JobStatus;
  /** [화면 미사용] 0~1. 분석 중 화면은 status(단계)만 보여준다 */
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
 * fps: 1초당 프레임 수. frames.length / fps = 재생 길이(초).
 * 백엔드는 student 와 coachRetargeted 를 같은 길이의 "재생 순서"로 맞춰서 보낸다:
 * student.frames[k] 와 coachRetargeted.frames[k] 가 같은 순간이다. (원본 영상의 프레임 순서 그대로가 아니다.
 * 한쪽만 움직이는 동안 다른 쪽은 같은 자세가 반복된다)
 * jointNames[j] 로 관절 이름을 알 수 있고, 뼈 연결은 프론트(utils/skeleton.ts)가
 * 이름 기준으로 만든다. 관절 개수·순서가 바뀌어도 이름만 맞으면 그려진다.
 * TODO(backend): 관절 세트 확정 (MediaPipe 33 / COCO 17 / SMPL 24 …)
 */
export interface SkeletonSequence {
  jointNames: string[];
  /** 재생 시계는 student.fps 하나다. [화면 미사용] coachRetargeted.fps (student.fps 와 같은 값으로 온다) */
  fps: number;
  frames: Vec3[][];
}

/**
 * 재생 프레임 k 에서 누가 움직이는지.
 * together: 둘 다 / coachAhead: 코치만 움직이고 나는 기다림 / coachWaits: 나만 움직이고 코치는 기다림
 */
export type FrameMapKind = 'together' | 'coachAhead' | 'coachWaits';

/**
 * 재생 프레임별 정보. 백엔드(app/result/frame_map.py)가 만든다.
 * 두 배열 모두 길이가 재생 프레임 수(= student.frames.length = coachRetargeted.frames.length)와 같다.
 * 프론트는 매핑을 직접 만들지 않고 읽어서 그리기만 한다.
 */
export interface FrameMap {
  /** 재생 프레임 k 에 그릴 코치 프레임. 지금 백엔드는 k 그대로(0, 1, 2, …) 보낸다. */
  coachFrame: number[];
  kind: FrameMapKind[];
}

/* ------------------------------------------------------------------ */
/* 결과                                                                */
/* ------------------------------------------------------------------ */

/** 체크포인트 구간 판정 (PDF "Checkpoint" 슬라이드의 일치·누락·추가·대체) */
export type SegmentVerdict = 'match' | 'missing' | 'extra' | 'substitute';

export type TimingLabel = 'ok' | 'slow' | 'fast';

/** 내 뼈가 코치의 뼈보다 어느 쪽으로 향해 있는지. inward / outward 는 몸 중심선 기준 */
export type DeviationDirection = 'up' | 'down' | 'forward' | 'backward' | 'inward' | 'outward' | 'left' | 'right';

/** 구간 안에서 차이가 컸던 뼈 하나 */
export interface BoneDeviation {
  /** 예: "torso", "left_thigh" — utils/skeleton.ts 의 뼈 이름과 맞춘다 */
  bone: string;
  /** 사용자 − 코치 각도 차이 (도) */
  angleDiffDeg: number;
  /** 차이가 가장 큰 방향. 백엔드가 계산하지 못한 경우에는 없다 */
  direction?: DeviationDirection;
  /**
   * [화면 미사용] angleDiffDeg 를 축별로 나눈 값 (도, 부호 있음).
   * 좌우가 있는 뼈는 up · forward · outward, 몸통 · 머리는 up · forward · right
   */
  components?: Partial<Record<'up' | 'forward' | 'outward' | 'right', number>>;
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
  /**
   * 재생 시간축의 [시작, 끝) (초). 재생 프레임 k 의 시각은 k / student.fps.
   * 사용자 원본 영상의 시각이 아니다 (코치만 움직이는 동안만큼 뒤로 밀린다. 원본 시각은 응답에 없다).
   * 구간은 다음 구간이 시작할 때까지 이어진다. missing 도 길이를 가진다 (코치가 그 동작을 하는 동안).
   * 재생할 프레임이 남지 않은 구간에는 없다.
   */
  studentRange?: [number, number];
  /** [화면 미사용] 코치 원본 영상의 시각 [시작, 끝] (초). 재생 시간축과 기준이 다르다. extra 면 없다. */
  coachRange?: [number, number];
  /** match 구간에서만 */
  timing?: {
    label: TimingLabel;
    /** 사용자 소요 시간 ÷ 코치 소요 시간 (slow ×1.5 → 1.5) */
    ratio: number;
  };
  /** 코치가 멈춘 곳에서 나는 얼마나 멈췄는지. 짧은 멈춤이 있는 구간에만 온다 */
  hold?: {
    coachSec: number;
    userSec: number;
    /** [화면 미사용] 내 멈춤 ÷ (코치 멈춤 × 전체 템포) */
    ratio: number;
  };
  /** 차이가 큰 뼈 상위 몇 개. 없으면 빈 배열 */
  deviations: BoneDeviation[];
}

/* ------------------------------------------------------------------ */
/* 자세 수치 (스쿼트)                                                   */
/* ------------------------------------------------------------------ */

/** 한 순간의 자세 수치. 백엔드 app/result/pose_metrics.py 의 pose() */
export interface PoseSnapshot {
  kneeFlexDeg: { left: number; right: number };
  hipFlexDeg: { left: number; right: number };
  trunkLeanDeg: number;
  hipKneeCm: number;
  pelvisDropCm: number;
  kneeWidthRatio: number;
}

export interface PoseComparison {
  user: PoseSnapshot | null;
  coach: PoseSnapshot | null;
  /** 사용자 − 코치. 둘 다 있을 때만 */
  diff?: PoseSnapshot;
}

type PoseMetricName = 'kneeFlexDeg' | 'hipFlexDeg' | 'trunkLeanDeg' | 'hipKneeCm' | 'pelvisDropCm' | 'kneeWidthRatio';
type MinMax = [number, number];

/**
 * [화면 미사용] 선 자세 · 반복별 바닥 자세의 관절 각도와 코치와의 차이.
 * 지금은 백엔드의 LLM 피드백 입력으로만 쓰이고 화면에는 그리지 않는다.
 * userSec · coachSec 는 재생 시간축이 아니라 각자 원본 영상의 시각이다.
 */
export interface PoseMetrics {
  legLengthCm: number;
  /** 이 값보다 차이가 작으면 "같다"고 보는 기준 */
  thresholds: { sameAngleDeg: number; sameHeightCm: number; sameKneeWidthRatio: number };
  repCount: { user: number; coach: number };
  standing: PoseComparison;
  reps: Array<{
    index: number;
    userSec: number | null;
    coachSec: number | null;
    segmentId: string | null;
    bottom: PoseComparison;
  }>;
  /** 짝지어진 반복들의 [최소, 최대]. 짝이 하나도 없으면 null */
  bottomSummary: ({ reps: number } & Record<PoseMetricName, { user: MinMax; coach: MinMax; diff: MinMax }>) | null;
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

export interface AnalysisResult {
  /** [화면 미사용] */
  jobId: string;
  exercise: ExerciseType;
  segments: Segment[];
  /** 사용자 스켈레톤 */
  student: SkeletonSequence;
  /** 사용자 체형으로 리타겟팅된 코치 스켈레톤 */
  coachRetargeted: SkeletonSequence;
  frameMap: FrameMap;
  /** [화면 미사용] 실제 백엔드는 항상 보낸다. mock 에는 없다 */
  poseMetrics?: PoseMetrics;
  /** LLM 단계가 아직 없으면 undefined */
  feedback?: LlmFeedback;
}
