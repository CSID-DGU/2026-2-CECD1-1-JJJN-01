import type { ExerciseMeta, ExerciseType } from '@/types/exercise';

/**
 * 종목 목록.
 * shootingGuide 는 임시 문구다. TODO: 팀에서 촬영 방향·거리 확정 후 교체
 */
export const EXERCISES: ExerciseMeta[] = [
  {
    id: 'squat',
    name: '스쿼트',
    description: '하강 · 바닥 유지 · 상승을 코치와 비교해요',
    shootingGuide: [
      '휴대폰을 옆에서, 허리 높이에 세로로 세워 주세요.',
      '머리부터 발끝까지 화면에 다 들어오게 2~3m 떨어져 주세요.',
      '한 번의 동작(내려갔다 올라오기)만 찍어 주세요.',
    ],
  },
  {
    id: 'deadlift',
    name: '데드리프트',
    description: '셋업 · 리프트 · 락아웃을 코치와 비교해요',
    shootingGuide: [
      '휴대폰을 옆에서, 허리 높이에 세로로 세워 주세요.',
      '바벨과 몸이 모두 보이게 2~3m 떨어져 주세요.',
      '한 번의 동작(들어 올려 내려놓기)만 찍어 주세요.',
    ],
  },
  {
    id: 'bench_press',
    name: '벤치프레스',
    description: '하강 · 가슴 터치 · 프레스를 코치와 비교해요',
    shootingGuide: [
      '휴대폰을 옆에서, 벤치 높이에 가로로 세워 주세요.',
      '바벨 경로와 팔 전체가 보이게 해 주세요.',
      '한 번의 동작(내렸다 밀어 올리기)만 찍어 주세요.',
    ],
  },
];

export const EXERCISE_BY_ID: Record<ExerciseType, ExerciseMeta> = Object.fromEntries(
  EXERCISES.map((e) => [e.id, e]),
) as Record<ExerciseType, ExerciseMeta>;

export function isExerciseType(value: unknown): value is ExerciseType {
  return typeof value === 'string' && value in EXERCISE_BY_ID;
}
