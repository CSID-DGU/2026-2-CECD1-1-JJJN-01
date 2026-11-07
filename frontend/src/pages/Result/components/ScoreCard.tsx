import type { AnalysisScore } from '@/types/analysis';
import { cn } from '@/utils/cn';
import { scoreCaption, scoreTone, type ScoreTone } from '@/utils/segments';

interface ScoreCardProps {
  score: AnalysisScore;
}

const BAR_CLASS: Record<ScoreTone, string> = {
  good: 'bg-good',
  warn: 'bg-warn',
  bad: 'bg-bad',
};

/** 총점. 숫자 하나와 한 줄 해석만 보여준다. 구간별 점수는 타임라인에서 본다. */
export function ScoreCard({ score }: ScoreCardProps) {
  const total = Math.round(Math.min(Math.max(score.total, 0), 100));
  const barClass = BAR_CLASS[scoreTone(total)];

  return (
    <div>
      <div className="flex items-baseline gap-1.5">
        <span className="text-[56px] leading-none font-bold tracking-tight text-ink">{total}</span>
        <span className="text-[17px] font-semibold text-ink-faint">/ 100</span>
      </div>
      <p className="mt-2 text-[15px] text-ink-muted">{scoreCaption(total)}</p>
      <div
        className="mt-4 h-1.5 w-full overflow-hidden rounded-full bg-page"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={total}
        aria-label="코치 동작과의 일치도"
      >
        <div className={cn('h-full rounded-full transition-[width]', barClass)} style={{ width: `${total}%` }} />
      </div>
    </div>
  );
}
