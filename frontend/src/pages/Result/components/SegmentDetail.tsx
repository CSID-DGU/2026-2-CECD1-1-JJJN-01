import { Badge } from '@/components/Badge';
import type { Segment } from '@/types/analysis';
import { formatDegrees, formatRange, formatRatio, formatSeconds } from '@/utils/format';
import { DIRECTION_LABEL, TIMING_LABEL, TIMING_TONE, VERDICT_LABEL, VERDICT_TONE } from '@/utils/segments';
import { boneLabel } from '@/utils/skeleton';

interface SegmentDetailProps {
  segment: Segment;
}

/** 타임라인에서 고른(또는 재생 중인) 구간의 판정·타이밍·자세 차이 */
export function SegmentDetail({ segment }: SegmentDetailProps) {
  const { timing, hold, studentRange } = segment;

  return (
    <div className="rounded-card border border-line p-4">
      <div className="flex items-center gap-2">
        <span className="text-[16px] font-bold text-ink">{segment.label}</span>
        <Badge tone={VERDICT_TONE[segment.verdict]}>{VERDICT_LABEL[segment.verdict]}</Badge>
        {timing && (
          <Badge tone={TIMING_TONE[timing.label]}>
            {TIMING_LABEL[timing.label]}
            {timing.label !== 'ok' && ` ${formatRatio(timing.ratio)}`}
          </Badge>
        )}
      </div>

      {/* studentRange 는 원본 영상이 아니라 위 뷰어 · 타임라인과 같은 재생 시간축의 시각이다 */}
      {(studentRange || hold) && (
        <dl className="mt-3 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-[13px]">
          {studentRange && (
            <>
              <dt className="text-ink-faint">재생 구간</dt>
              <dd className="text-ink-muted tabular-nums">{formatRange(studentRange)}</dd>
            </>
          )}
          {hold && (
            <>
              <dt className="text-ink-faint">멈춤</dt>
              <dd className="text-ink-muted tabular-nums">
                코치 {formatSeconds(hold.coachSec, 2)} · 나 {formatSeconds(hold.userSec, 2)}
              </dd>
            </>
          )}
        </dl>
      )}

      {segment.verdict === 'missing' && (
        <p className="mt-3 text-[13px] leading-relaxed text-ink-muted">코치는 이 구간을 거치는데 내 동작에는 없어요.</p>
      )}
      {segment.verdict === 'extra' && (
        <p className="mt-3 text-[13px] leading-relaxed text-ink-muted">코치 동작에는 없는 움직임이 끼어 있어요.</p>
      )}

      {segment.deviations.length > 0 && (
        <ul className="mt-3 flex flex-col gap-1.5 border-t border-line pt-3">
          {segment.deviations.map((d) => (
            <li key={d.bone} className="flex items-center justify-between text-[13px]">
              <span className="flex items-center gap-2 text-ink">
                <span className="size-2 rounded-full bg-bad" aria-hidden="true" />
                {boneLabel(d.bone)}
              </span>
              <span className="text-ink-muted tabular-nums">
                {d.direction
                  ? `코치보다 ${DIRECTION_LABEL[d.direction]} ${formatDegrees(d.angleDiffDeg)}`
                  : `코치보다 ${formatDegrees(d.angleDiffDeg)} 차이`}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
