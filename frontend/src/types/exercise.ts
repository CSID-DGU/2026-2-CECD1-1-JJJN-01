/** 지원하는 운동 종목. 백엔드의 체크포인트 판정 규칙도 이 값으로 분기한다. */
export type ExerciseType = 'squat' | 'deadlift' | 'bench_press';

export interface ExerciseMeta {
  id: ExerciseType;
  /** 화면에 보이는 이름 */
  name: string;
  /** 한 줄 설명 */
  description: string;
  /**
   * 촬영 가이드 문구.
   * TODO: 팀에서 방향성 확정 후 교체 (현재는 임시 문구)
   */
  shootingGuide: string[];
}
