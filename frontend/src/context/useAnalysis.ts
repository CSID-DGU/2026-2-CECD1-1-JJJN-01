import { useContext } from 'react';
import { AnalysisContext, type AnalysisContextValue } from './analysisContextValue';

/** 한 번의 분석 흐름이 공유하는 상태. <AnalysisProvider> 안에서만 쓸 수 있다. */
export function useAnalysis(): AnalysisContextValue {
  const ctx = useContext(AnalysisContext);
  if (!ctx) throw new Error('useAnalysis 는 <AnalysisProvider> 안에서만 쓸 수 있어요.');
  return ctx;
}
