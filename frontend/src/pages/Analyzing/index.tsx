import { useEffect } from 'react';
import { Navigate, useNavigate } from 'react-router';
import { BottomBar } from '@/components/BottomBar';
import { Button } from '@/components/Button';
import { PageHeader } from '@/components/PageHeader';
import { Spinner } from '@/components/Spinner';
import { ROUTES } from '@/constants/routes';
import { useAnalysis } from '@/context/useAnalysis';
import { useAnalysisJob } from '@/hooks/useAnalysisJob';
import { ProgressSteps } from './components/ProgressSteps';

/**
 * 3단계: 분석이 끝날 때까지 기다린다.
 * 백엔드가 단계별 상태를 주면 ProgressSteps 가 진행을 보여주고, 안 주면 스피너만 돈다.
 */
export function AnalyzingPage() {
  const navigate = useNavigate();
  const { jobId, setResult, setJobId } = useAnalysis();
  const { job, error } = useAnalysisJob(jobId);

  // 여기서 jobId 를 비우면 안 된다. 라우터 전환이 끝나기 전에 다시 렌더되면
  // 아래 "jobId 없음 → 업로드로" 리다이렉트가 먼저 걸린다. jobId 는 다음 분석 요청이나 reset 에서 바뀐다.
  useEffect(() => {
    if (job?.status === 'done' && job.result) {
      setResult(job.result);
      navigate(ROUTES.result, { replace: true });
    }
  }, [job, setResult, navigate]);

  if (!jobId) return <Navigate to={ROUTES.upload} replace />;

  const failed = job?.status === 'failed';
  const status = job?.status ?? 'queued';

  return (
    <>
      <PageHeader title="분석 중" />

      <main className="flex flex-1 flex-col px-5 pt-6 pb-6">
        {failed ? (
          <div className="flex flex-1 flex-col items-center justify-center text-center">
            <h2 className="text-[20px] font-bold text-ink">분석을 끝내지 못했어요</h2>
            <p className="mt-2 max-w-[280px] text-[14px] leading-relaxed text-ink-muted">
              {job?.error ?? '알 수 없는 문제가 생겼어요. 영상을 다시 올려 주세요.'}
            </p>
          </div>
        ) : (
          <>
            <div className="flex flex-col items-center py-10">
              <Spinner size={44} className="text-ink" />
              <h2 className="mt-6 text-[20px] font-bold text-ink">코치와 비교하고 있어요</h2>
              <p className="mt-1 text-[14px] text-ink-muted">보통 1분 안에 끝나요. 화면을 닫지 말아 주세요.</p>
              {error && <p className="mt-3 text-[13px] text-warn">{error} 다시 시도하는 중…</p>}
            </div>
            <ProgressSteps status={status} />
          </>
        )}
      </main>

      {failed && (
        <BottomBar>
          <Button
            size="lg"
            block
            onClick={() => {
              setJobId(null);
              navigate(ROUTES.upload, { replace: true });
            }}
          >
            영상 다시 올리기
          </Button>
        </BottomBar>
      )}
    </>
  );
}
