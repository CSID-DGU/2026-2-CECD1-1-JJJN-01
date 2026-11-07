import type { ReactNode } from 'react';
import { useNavigate } from 'react-router';
import { cn } from '@/utils/cn';

interface PageHeaderProps {
  title?: string;
  /** 뒤로가기 목적지. 없으면 뒤로가기 버튼을 안 그린다. */
  backTo?: string;
  /** 오른쪽 끝에 둘 것 (텍스트 버튼 등) */
  right?: ReactNode;
  className?: string;
}

/** 화면 맨 위 바. 뒤로가기 · 제목 · 오른쪽 액션 */
export function PageHeader({ title, backTo, right, className }: PageHeaderProps) {
  const navigate = useNavigate();

  return (
    <header
      className={cn(
        'sticky top-0 z-20 flex h-14 items-center gap-2 bg-surface/95 px-2 pt-safe backdrop-blur',
        className,
      )}
    >
      <div className="flex w-12 shrink-0 items-center">
        {backTo && (
          <button
            type="button"
            onClick={() => navigate(backTo)}
            aria-label="뒤로"
            className="flex size-10 items-center justify-center rounded-full text-ink hover:bg-page"
          >
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path
                d="M15 5l-7 7 7 7"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        )}
      </div>
      <h1 className="flex-1 truncate text-center text-[15px] font-semibold text-ink">{title}</h1>
      <div className="flex w-12 shrink-0 items-center justify-end">{right}</div>
    </header>
  );
}
