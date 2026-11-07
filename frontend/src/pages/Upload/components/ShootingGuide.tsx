import { UPLOAD_LIMITS } from '@/constants/upload';
import { formatBytes } from '@/utils/format';

interface ShootingGuideProps {
  lines: string[];
}

/**
 * 촬영 가이드. 접었다 펼 수 있다.
 * 문구는 constants/exercises.ts 의 shootingGuide 에서 온다. (TODO: 팀 확정 후 교체)
 */
export function ShootingGuide({ lines }: ShootingGuideProps) {
  return (
    <details className="group rounded-card bg-page px-4 py-3">
      <summary className="flex cursor-pointer list-none items-center justify-between text-[14px] font-semibold text-ink [&::-webkit-details-marker]:hidden">
        이렇게 찍으면 분석이 더 정확해요
        <svg
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          aria-hidden="true"
          className="text-ink-muted transition-transform group-open:rotate-180"
        >
          <path d="M6 9l6 6 6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </summary>
      <ol className="mt-3 flex flex-col gap-2 text-[14px] leading-relaxed text-ink-muted">
        {lines.map((line, i) => (
          <li key={i} className="flex gap-2">
            <span className="w-4 shrink-0 text-ink-faint">{i + 1}</span>
            <span>{line}</span>
          </li>
        ))}
        <li className="flex gap-2 border-t border-line pt-2">
          <span className="w-4 shrink-0" />
          <span>
            {UPLOAD_LIMITS.maxDurationSec}초 이내, {formatBytes(UPLOAD_LIMITS.maxBytes)} 이하의 mp4·mov·webm 영상
          </span>
        </li>
      </ol>
    </details>
  );
}
