import type { ExerciseType } from './exercise';

/** 기본 제공 코치 영상 한 개 */
export interface CoachVideoOption {
  /** 백엔드에 보낼 식별자 */
  id: string;
  exercise: ExerciseType;
  title: string;
  /** 어떤 타깃·변형인지 (예: "하이바 · 풀스쿼트") */
  description: string;
  /** TODO(backend): 정적 파일 URL 또는 CDN 주소 */
  thumbnailUrl?: string;
  durationSec?: number;
}

/** 사용자가 고른 코치 영상 */
export type CoachSelection = { kind: 'default'; coachId: string } | { kind: 'custom'; file: File };
