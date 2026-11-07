import type { ReactNode } from 'react';
import { cn } from '@/utils/cn';

interface BottomBarProps {
  children: ReactNode;
  /** 버튼 위에 둘 짧은 안내 문구 */
  note?: ReactNode;
  className?: string;
}

/** 화면 아래에 붙는 주요 버튼 영역. 스크롤 내용이 가려지지 않게 위쪽을 투명→흰색으로 흐린다. */
export function BottomBar({ children, note, className }: BottomBarProps) {
  return (
    <div
      className={cn(
        'sticky bottom-0 z-20 mt-auto bg-linear-to-t from-surface via-surface/95 to-surface/0 px-5 pt-6 pb-safe',
        className,
      )}
    >
      {note && <p className="mb-2 text-center text-[13px] text-ink-muted">{note}</p>}
      <div className="flex flex-col gap-2 pb-4">{children}</div>
    </div>
  );
}
