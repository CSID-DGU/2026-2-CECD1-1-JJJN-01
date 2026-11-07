import { useEffect, useState } from 'react';
import { getAnalysisJob } from '@/api/analysis';
import { JOB_POLL_INTERVAL_MS } from '@/constants/upload';
import type { AnalysisJob } from '@/types/analysis';

interface UseAnalysisJobResult {
  job: AnalysisJob | null;
  /** 네트워크 오류 등 폴링 자체가 실패한 경우 */
  error: string | null;
}

/** 어느 jobId 의 결과인지 같이 들고 있어서, jobId 가 바뀌면 이전 값이 새어 나가지 않는다 */
interface PollState {
  jobId: string;
  job: AnalysisJob | null;
  error: string | null;
}

/**
 * jobId 의 상태를 done/failed 가 될 때까지 주기적으로 조회한다.
 * 나중에 실시간(SSE/WebSocket)으로 바꾸더라도 이 훅의 반환값만 유지하면 화면은 안 바뀐다.
 */
export function useAnalysisJob(jobId: string | null): UseAnalysisJobResult {
  const [state, setState] = useState<PollState | null>(null);

  useEffect(() => {
    if (!jobId) return;

    let cancelled = false;
    let timer: number | null = null;

    const tick = async () => {
      try {
        const next = await getAnalysisJob(jobId);
        if (cancelled) return;
        setState({ jobId, job: next, error: null });
        if (next.status === 'done' || next.status === 'failed') return;
      } catch (e) {
        if (cancelled) return;
        const message = e instanceof Error ? e.message : '상태를 불러오지 못했어요.';
        setState((s) => ({ jobId, job: s?.jobId === jobId ? s.job : null, error: message }));
      }
      timer = window.setTimeout(tick, JOB_POLL_INTERVAL_MS);
    };

    void tick();

    return () => {
      cancelled = true;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [jobId]);

  if (!jobId || state?.jobId !== jobId) return { job: null, error: null };
  return { job: state.job, error: state.error };
}
