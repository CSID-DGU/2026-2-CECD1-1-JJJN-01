import type { ReactNode } from 'react';
import { cn } from '@/utils/cn';

interface OptionCardProps {
  selected: boolean;
  onSelect: () => void;
  title: string;
  description?: string;
  /** 왼쪽에 둘 아이콘/모노그램 */
  leading?: ReactNode;
  /** 오른쪽 끝 (뱃지 등) */
  trailing?: ReactNode;
  disabled?: boolean;
}

/** 여러 개 중 하나를 고르는 카드. 종목 선택·코치 영상 선택에서 같이 쓴다. */
export function OptionCard({ selected, onSelect, title, description, leading, trailing, disabled }: OptionCardProps) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={selected}
      disabled={disabled}
      onClick={onSelect}
      className={cn(
        'flex w-full items-center gap-3 rounded-card border p-4 text-left transition-colors',
        selected ? 'border-ink bg-surface ring-1 ring-ink' : 'border-line bg-surface hover:border-ink-faint',
        disabled && 'cursor-not-allowed opacity-50',
      )}
    >
      {leading && <div className="shrink-0">{leading}</div>}
      <div className="min-w-0 flex-1">
        <div className="text-[15px] font-semibold text-ink">{title}</div>
        {description && <div className="mt-0.5 text-[13px] leading-snug text-ink-muted">{description}</div>}
      </div>
      {trailing && <div className="shrink-0">{trailing}</div>}
      <span
        aria-hidden="true"
        className={cn(
          'flex size-5 shrink-0 items-center justify-center rounded-full border',
          selected ? 'border-ink bg-ink' : 'border-line',
        )}
      >
        {selected && (
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
            <path
              d="M2.5 6.5l2.3 2.3L9.5 4"
              stroke="white"
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        )}
      </span>
    </button>
  );
}
