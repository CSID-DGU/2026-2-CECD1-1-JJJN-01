import type { ExerciseType } from './exercise';

/** 기본 제공 코치 영상 한 개 */
export interface CoachVideoOption {
  /** 화면에서 선택을 구분하는 식별자. 서버로 보내지 않는다 (올릴 파일 이름에만 쓴다) */
  id: string;
  exercise: ExerciseType;
  title: string;
  /** 어떤 타깃·변형인지 (예: "하이바 · 풀스쿼트") */
  description: string;
  /** 영상 파일 주소. 업로드 화면에서 재생해 보고, 분석을 시작할 때 이 파일을 서버로 올린다 */
  videoUrl: string;
  durationSec?: number;
}

/** 사용자가 고른 코치 영상 */
export type CoachSelection = { kind: 'default'; coachId: string } | { kind: 'custom'; file: File };
