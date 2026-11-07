import type { CoachVideoOption } from '@/types/coach';
import type { ExerciseType } from '@/types/exercise';

/**
 * 기본 제공 코치 영상 목록.
 * 같은 종목이라도 타깃(변형)에 따라 다른 영상을 쓸 수 있도록 종목당 여러 개를 둔다.
 * TODO(backend): 실제 id / 썸네일 / 길이로 교체. 백엔드는 이 id 로 미리 추출해 둔
 *                코치 스켈레톤을 찾는다.
 */
export const COACH_VIDEOS: CoachVideoOption[] = [
  {
    id: 'squat-highbar-full',
    exercise: 'squat',
    title: '하이바 스쿼트',
    description: '풀 스쿼트 · 대퇴 평행 아래까지',
    durationSec: 4,
  },
  {
    id: 'squat-lowbar',
    exercise: 'squat',
    title: '로우바 스쿼트',
    description: '힙 드라이브 위주 · 파워리프팅 스타일',
    durationSec: 4,
  },
  {
    id: 'deadlift-conventional',
    exercise: 'deadlift',
    title: '컨벤셔널 데드리프트',
    description: '좁은 스탠스 · 바 가까이',
    durationSec: 5,
  },
  {
    id: 'deadlift-sumo',
    exercise: 'deadlift',
    title: '스모 데드리프트',
    description: '넓은 스탠스 · 짧은 가동 범위',
    durationSec: 5,
  },
  {
    id: 'bench-flat',
    exercise: 'bench_press',
    title: '플랫 벤치프레스',
    description: '견갑 고정 · 아치 유지',
    durationSec: 4,
  },
];

export function coachVideosFor(exercise: ExerciseType): CoachVideoOption[] {
  return COACH_VIDEOS.filter((v) => v.exercise === exercise);
}
