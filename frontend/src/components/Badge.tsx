import type { ReactNode } from 'react';
import { cn } from '@/utils/cn';

export type BadgeTone = 'neutral' | 'coach' | 'me' | 'good' | 'bad' | 'warn' | 'slow' | 'fast';

const TONE_CLASS: Record<BadgeTone, string> = {
  neutral: 'bg-page text-ink-muted',
  coach: 'bg-coach-soft text-coach',
  me: 'bg-me-soft text-me',
  good: 'bg-good-soft text-good',
  bad: 'bg-bad-soft text-bad',
  warn: 'bg-warn-soft text-warn',
  slow: 'bg-coach-soft text-slow',
  fast: 'bg-bad-soft text-fast',
};

interface BadgeProps {
  tone?: BadgeTone;
  children: ReactNode;
  className?: string;
}

/** 작은 라벨. 판정·타이밍·누구 것인지 표시 */
export function Badge({ tone = 'neutral', children, className }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex h-6 items-center rounded-full px-2 text-[12px] font-semibold whitespace-nowrap',
        TONE_CLASS[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}
