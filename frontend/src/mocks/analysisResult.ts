/**
 * 백엔드 없이 결과 화면을 확인하기 위한 결과. 실제 파이프라인 출력이다:
 * src/posture/ 의 테스트 쌍(squat_coach.mp4 → squat_student.mp4)을 돌린 결과를 그대로 넣었다.
 *  - posture_squat/result.json   backend build_result 응답 (LLM 피드백 포함)
 *  - posture_squat/overlay.mp4   engine pipeline.overlay.coach_on_video 출력 (같은 재생 순서, 30 fps)
 * 그래서 오버레이 영상 · 3D 뷰어 · 구간이 실제로 같은 순간을 가리키는지 확인할 수 있다.
 * 데드리프트 · 벤치프레스도 지금은 같은 결과를 쓴다 (화면 확인용, exercise 만 바꾼다).
 *
 * 다시 만들기 (runs/coach_user/posture_squat 이 있을 때):
 *   backend/ 에서 build_result('mock-posture-squat', 'squat', retarget_dir, with_score=True) + generate_feedback
 *   → result.json, engine/ 에서 coach_on_video → overlay.mp4 (ffmpeg -crf 28 로 줄여서 넣었다)
 *
 * 타입(src/types/analysis.ts)이 바뀌면 이 파일과 result.json 을 같이 고친다.
 */

import type { AnalysisResult } from '@/types/analysis';
import type { ExerciseType } from '@/types/exercise';
import overlayUrl from './posture_squat/overlay.mp4';
import postureSquat from './posture_squat/result.json';

export function buildMockResult(exercise: ExerciseType, jobId: string): AnalysisResult {
  const result = postureSquat as unknown as AnalysisResult;
  return { ...result, jobId, exercise, overlayUrl };
}
