import type { Segment } from '@/types/analysis';
import { cn } from '@/utils/cn';
import { segmentTone, TIMING_LABEL, VERDICT_LABEL, type Tone } from '@/utils/segments';

interface SegmentTimelineProps {
  segments: Segment[];
  /** 재생 전체 길이 (초) */
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

/** 범례 순서 */
const LEGEND_ORDER: Tone[] = ['good', 'slow', 'fast', 'bad', 'warn', 'neutral'];

/** 범례에 쓸 말. segmentTone 과 같은 기준: 일치 구간은 타이밍, 나머지는 판정 */
function legendLabel(segment: Segment): string {
  if (segment.verdict !== 'match') return VERDICT_LABEL[segment.verdict];
  if (segment.timing && segment.timing.label !== 'ok') return TIMING_LABEL[segment.timing.label];
  return VERDICT_LABEL.match;
}

/** 이 결과에 실제로 나온 색만, 색마다 뜻을 모아서 (예: 주황 = "추가 동작 · 다른 동작") */
function buildLegend(segments: Segment[]): Array<{ tone: Tone; label: string }> {
  const byTone = new Map<Tone, Set<string>>();
  for (const segment of segments) {
    const tone = segmentTone(segment);
    const labels = byTone.get(tone) ?? new Set<string>();
    labels.add(legendLabel(segment));
    byTone.set(tone, labels);
  }
  return LEGEND_ORDER.filter((tone) => byTone.has(tone)).map((tone) => ({
    tone,
    label: [...(byTone.get(tone) ?? [])].join(' · '),
  }));
}

/**
 * 재생 시간축(뷰어의 슬라이더와 같은 시간) 위에 구간을 늘어놓는다.
 * 블록의 위치 · 길이는 재생에서 그 구간이 차지한 시간 그대로라서, 재생선이 블록 경계를 지날 때 구간이 바뀐다.
 * 색은 판정(일치면 타이밍). 좁은 블록은 글자 없이 색만 보이고, 아래 칩 줄에서 이름으로 고른다.
 */
export function SegmentTimeline({ segments, totalSec, currentSec, selectedId, onSelect }: SegmentTimelineProps) {
  const safeTotal = totalSec > 0 ? totalSec : 1;
  const playheadPct = Math.min(Math.max((currentSec / safeTotal) * 100, 0), 100);
  const legend = buildLegend(segments);

  return (
    <div>
      <ul className="mb-3 flex flex-wrap gap-x-3 gap-y-1.5" aria-label="색 설명">
        {legend.map(({ tone, label }) => (
          <li key={tone} className="flex items-center gap-1.5 text-[12px] text-ink-muted">
            <span className={cn('size-2 rounded-full', DOT_CLASS[tone])} aria-hidden="true" />
            {label}
          </li>
        ))}
      </ul>

      <div className="relative">
        <div
          className="relative h-9 w-full overflow-hidden rounded-[10px] bg-page"
          role="list"
          aria-label="구간 시간축"
        >
          {segments.map((segment) => {
            const range = segment.studentRange;
            const duration = range ? Math.max(range[1] - range[0], 0) : 0;
            // 재생 시간이 없는 구간은 막대에 자리가 없다. 아래 칩 줄에서 고른다
            if (!range || duration <= 0) return null;
            // 재생선과 같은 축: 블록의 왼쪽 끝 = 구간 시작 시각. 폭을 부풀리거나 순서대로 이어 붙이지 않는다
            const leftPct = Math.min(Math.max((range[0] / safeTotal) * 100, 0), 100);
            const widthPct = Math.min((duration / safeTotal) * 100, 100 - leftPct);
            const showLabel = duration / safeTotal >= LABEL_MIN_RATIO;
            const selected = segment.id === selectedId;
            return (
              <button
                key={segment.id}
                type="button"
                role="listitem"
                onClick={() => onSelect(segment)}
                aria-label={segment.label}
                style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
                className={cn(
                  'absolute inset-y-0 min-w-1 overflow-hidden border-r-2 border-surface text-left text-[11px] leading-tight font-semibold whitespace-nowrap transition-shadow',
                  showLabel && 'px-1.5',
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
