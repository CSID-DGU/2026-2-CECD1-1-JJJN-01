import type { Segment } from '@/types/analysis';
import { cn } from '@/utils/cn';
import { segmentTone, VERDICT_LABEL, type Tone } from '@/utils/segments';

interface SegmentTimelineProps {
  segments: Segment[];
  /** 사용자 영상 전체 길이 (초) */
  totalSec: number;
  /** 지금 재생 위치 (초) */
  currentSec: number;
  selectedId: string | null;
  onSelect: (segment: Segment) => void;
}

const TONE_CLASS: Record<Tone, string> = {
  neutral: 'bg-page text-ink-muted',
  good: 'bg-good-soft text-good',
  bad: 'bg-bad-soft text-bad',
  warn: 'bg-warn-soft text-warn',
  slow: 'bg-coach-soft text-slow',
  fast: 'bg-bad-soft text-fast',
};

const DOT_CLASS: Record<Tone, string> = {
  neutral: 'bg-ink-faint',
  good: 'bg-good',
  bad: 'bg-bad',
  warn: 'bg-warn',
  slow: 'bg-slow',
  fast: 'bg-fast',
};

/** 블록 폭이 전체의 이 비율보다 좁으면 글자를 넣지 않는다 (잘린 글자보다 빈 블록이 낫다) */
const LABEL_MIN_RATIO = 0.16;

/**
 * 사용자 영상 시간축 위에 구간을 늘어놓는다.
 * 블록 길이는 내 영상에서 그 구간이 차지한 시간. 색은 판정(일치면 타이밍).
 * 좁은 블록은 글자 없이 색만 보이고, 아래 칩 줄에서 이름으로 고른다.
 */
export function SegmentTimeline({ segments, totalSec, currentSec, selectedId, onSelect }: SegmentTimelineProps) {
  const safeTotal = totalSec > 0 ? totalSec : 1;
  const playheadPct = Math.min(Math.max((currentSec / safeTotal) * 100, 0), 100);

  return (
    <div>
      <div className="relative">
        <div className="flex h-9 w-full gap-0.5 overflow-hidden rounded-[10px]" role="list" aria-label="구간 시간축">
          {segments.map((segment) => {
            const range = segment.studentRange;
            const duration = range ? Math.max(range[1] - range[0], 0) : 0;
            // 누락처럼 시간이 거의 없는 구간도 눌리도록 최소 폭을 준다
            const grow = Math.max(duration, safeTotal * 0.04);
            const showLabel = duration / safeTotal >= LABEL_MIN_RATIO;
            const selected = segment.id === selectedId;
            return (
              <button
                key={segment.id}
                type="button"
                role="listitem"
                onClick={() => onSelect(segment)}
                aria-label={segment.label}
                style={{ flexGrow: grow, flexBasis: 0 }}
                className={cn(
                  'min-w-0 overflow-hidden px-1.5 text-left text-[11px] leading-tight font-semibold whitespace-nowrap transition-shadow',
                  TONE_CLASS[segmentTone(segment)],
                  selected && 'inset-ring-2 inset-ring-ink',
                )}
              >
                {showLabel && segment.label}
              </button>
            );
          })}
        </div>
        <div
          aria-hidden="true"
          className="pointer-events-none absolute -top-1 -bottom-1 w-0.5 rounded-full bg-ink"
          style={{ left: `calc(${playheadPct}% - 1px)` }}
        />
      </div>

      <div className="-mx-5 mt-3 scrollbar-none flex gap-2 overflow-x-auto px-5" role="list" aria-label="구간 목록">
        {segments.map((segment) => {
          const selected = segment.id === selectedId;
          const tone = segmentTone(segment);
          return (
            <button
              key={segment.id}
              type="button"
              role="listitem"
              aria-pressed={selected}
              onClick={() => onSelect(segment)}
              className={cn(
                'flex h-8 shrink-0 items-center gap-1.5 rounded-full border px-3 text-[13px] font-semibold whitespace-nowrap transition-colors',
                selected ? 'border-ink bg-ink text-white' : 'border-line bg-surface text-ink',
              )}
            >
              <span className={cn('size-2 rounded-full', DOT_CLASS[tone])} aria-hidden="true" />
              {segment.label}
              {segment.verdict !== 'match' && (
                <span className={cn('text-[11px] font-medium', selected ? 'text-white/70' : 'text-ink-muted')}>
                  {VERDICT_LABEL[segment.verdict]}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
