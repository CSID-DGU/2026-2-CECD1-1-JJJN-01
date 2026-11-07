import { createContext } from 'react';
import type { AnalysisResult } from '@/types/analysis';
import type { CoachSelection } from '@/types/coach';
import type { ExerciseType } from '@/types/exercise';

/**
 * 한 번의 분석 흐름(종목 → 업로드 → 분석 → 결과)이 공유하는 상태의 모양.
 * Provider 는 AnalysisContext.tsx, 읽는 쪽은 useAnalysis.ts.
 * (Fast Refresh 때문에 컴포넌트 파일에서 context 객체를 같이 export 하지 않는다)
 */

export interface AnalysisState {
  exercise: ExerciseType | null;
  studentVideo: File | null;
  coachSelection: CoachSelection | null;
  jobId: string | null;
  result: AnalysisResult | null;
}

export interface AnalysisActions {
  setExercise: (exercise: ExerciseType) => void;
  setStudentVideo: (file: File | null) => void;
  setCoachSelection: (selection: CoachSelection | null) => void;
  setJobId: (jobId: string | null) => void;
  setResult: (result: AnalysisResult | null) => void;
  /** 처음부터 다시 */
  reset: () => void;
}

export type AnalysisContextValue = AnalysisState & AnalysisActions;

/** 컴포넌트에서는 이 객체 대신 useAnalysis() 를 쓴다 */
export const AnalysisContext = createContext<AnalysisContextValue | null>(null);
