import type { ReactNode } from 'react';
import { AnalysisProvider } from '@/context/AnalysisContext';

/** 앱 전체를 감싸는 Provider 들. 새 전역 상태가 생기면 여기에 겹친다. */
export function AppProviders({ children }: { children: ReactNode }) {
  return <AnalysisProvider>{children}</AnalysisProvider>;
}
