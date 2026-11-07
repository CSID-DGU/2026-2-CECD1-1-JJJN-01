import { useCallback, useMemo, useState, type ReactNode } from 'react';
import type { AnalysisResult } from '@/types/analysis';
import type { CoachSelection } from '@/types/coach';
import type { ExerciseType } from '@/types/exercise';
import { isExerciseType } from '@/constants/exercises';
import { readStorage, removeStorage, writeStorage } from '@/utils/storage';
import { AnalysisContext, type AnalysisContextValue, type AnalysisState } from './analysisContextValue';

/**
 * 한 번의 분석 흐름(종목 → 업로드 → 분석 → 결과)이 공유하는 상태.
 * File 객체는 새로고침하면 사라지므로 저장하지 않고, 나머지는 sessionStorage 에 남긴다.
 */

const KEYS = {
  exercise: 'exercise',
  coachId: 'coachId',
  jobId: 'jobId',
  result: 'result',
} as const;

function loadInitialState(): AnalysisState {
  const storedExercise = readStorage<string>(KEYS.exercise);
  const storedCoachId = readStorage<string>(KEYS.coachId);
  return {
    exercise: isExerciseType(storedExercise) ? storedExercise : null,
    studentVideo: null,
    coachSelection: storedCoachId ? { kind: 'default', coachId: storedCoachId } : null,
    jobId: readStorage<string>(KEYS.jobId),
    result: readStorage<AnalysisResult>(KEYS.result),
  };
}

export function AnalysisProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AnalysisState>(loadInitialState);

  const setExercise = useCallback((exercise: ExerciseType) => {
    writeStorage(KEYS.exercise, exercise);
    // 종목이 바뀌면 코치 영상 선택은 무효
    removeStorage(KEYS.coachId);
    setState((s) => (s.exercise === exercise ? s : { ...s, exercise, coachSelection: null }));
  }, []);

  const setStudentVideo = useCallback((file: File | null) => {
    setState((s) => ({ ...s, studentVideo: file }));
  }, []);

  const setCoachSelection = useCallback((selection: CoachSelection | null) => {
    if (selection?.kind === 'default') writeStorage(KEYS.coachId, selection.coachId);
    else removeStorage(KEYS.coachId);
    setState((s) => ({ ...s, coachSelection: selection }));
  }, []);

  const setJobId = useCallback((jobId: string | null) => {
    if (jobId) writeStorage(KEYS.jobId, jobId);
    else removeStorage(KEYS.jobId);
    setState((s) => ({ ...s, jobId }));
  }, []);

  const setResult = useCallback((result: AnalysisResult | null) => {
    if (result) writeStorage(KEYS.result, result);
    else removeStorage(KEYS.result);
    setState((s) => ({ ...s, result }));
  }, []);

  const reset = useCallback(() => {
    Object.values(KEYS).forEach(removeStorage);
    setState({ exercise: null, studentVideo: null, coachSelection: null, jobId: null, result: null });
  }, []);

  const value = useMemo<AnalysisContextValue>(
    () => ({ ...state, setExercise, setStudentVideo, setCoachSelection, setJobId, setResult, reset }),
    [state, setExercise, setStudentVideo, setCoachSelection, setJobId, setResult, reset],
  );

  return <AnalysisContext.Provider value={value}>{children}</AnalysisContext.Provider>;
}
