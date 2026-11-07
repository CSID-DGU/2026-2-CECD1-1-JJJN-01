import type { JobStatus } from '@/types/analysis';
import { cn } from '@/utils/cn';

/** 화면에 보이는 단계. 백엔드 JobStatus 와 1:1 로 맞춘다. */
const STEPS: Array<{ status: JobStatus; label: string; detail: string }> = [
  { status: 'extracting', label: '동작 읽는 중', detail: '영상에서 관절 위치를 찾아요' },
  { status: 'retargeting', label: '내 체형에 맞추는 중', detail: '코치 동작을 내 몸 비율로 옮겨요' },
  { status: 'scoring', label: '비교하는 중', detail: '구간별로 타이밍과 자세를 맞춰 봐요' },
  { status: 'llm', label: '피드백 쓰는 중', detail: '비교 결과를 문장으로 정리해요' },
];

const ORDER: JobStatus[] = ['queued', 'extracting', 'retargeting', 'scoring', 'llm', 'done'];

interface ProgressStepsProps {
  status: JobStatus;
}

export function ProgressSteps({ status }: ProgressStepsProps) {
  const current = ORDER.indexOf(status);

  return (
    <ol className="flex flex-col gap-1">
      {STEPS.map((step) => {
        const index = ORDER.indexOf(step.status);
        const state: 'done' | 'active' | 'todo' = index < current ? 'done' : index === current ? 'active' : 'todo';
        return (
          <li key={step.status} className="flex items-start gap-3 py-2">
            <span
              aria-hidden="true"
              className={cn(
                'mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full border text-[11px] font-bold',
                state === 'done' && 'border-ink bg-ink text-white',
                state === 'active' && 'border-ink text-ink',
                state === 'todo' && 'border-line text-ink-faint',
              )}
            >
              {state === 'done' ? (
                <svg width="11" height="11" viewBox="0 0 12 12" fill="none">
                  <path
                    d="M2.5 6.5l2.3 2.3L9.5 4"
                    stroke="currentColor"
                    strokeWidth="1.8"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              ) : state === 'active' ? (
                <span className="size-2 animate-pulse rounded-full bg-ink" />
              ) : null}
            </span>
            <div>
              <div className={cn('text-[15px] font-semibold', state === 'todo' ? 'text-ink-faint' : 'text-ink')}>
                {step.label}
              </div>
              <div className={cn('text-[13px]', state === 'todo' ? 'text-ink-faint' : 'text-ink-muted')}>
                {step.detail}
              </div>
            </div>
          </li>
        );
      })}
    </ol>
  );
}
