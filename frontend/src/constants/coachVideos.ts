import squatCoachVideo from '@/posture/squat_coach.mp4';
import type { CoachVideoOption } from '@/types/coach';
import type { ExerciseType } from '@/types/exercise';

/**
 * 기본 제공 코치 영상. 종목당 하나만 둔다.
 *
 * 영상 파일은 프론트(src/posture/)에 들어 있고, 기본 영상을 고르면 그 파일을 직접 올린 영상처럼 서버로 보낸다
 * (api/coachVideo.ts). 백엔드는 코치 영상이 항상 업로드로 온다고 본다. 그래서 id 를 백엔드와 맞출 필요가 없다.
 * 영상이 없는 종목(데드리프트 · 벤치프레스)은 기본 영상 없이 직접 올리기만 된다.
 */
export const COACH_VIDEOS: CoachVideoOption[] = [
  {
    id: 'squat-default',
    exercise: 'squat',
    title: '기본 스쿼트',
    description: '기본으로 제공하는 코치 영상이에요',
    videoUrl: squatCoachVideo,
    durationSec: 5,
  },
];

/** 그 종목의 기본 코치 영상. 종목당 하나, 없으면 null */
export function coachVideoFor(exercise: ExerciseType): CoachVideoOption | null {
  return COACH_VIDEOS.find((v) => v.exercise === exercise) ?? null;
}
