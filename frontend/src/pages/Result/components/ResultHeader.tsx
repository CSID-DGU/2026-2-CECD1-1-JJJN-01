import { Button } from '@/components/Button';

interface ResultHeaderProps {
  title: string;
  onBack: () => void;
  /** 다른 영상으로 다시 비교 (업로드 화면으로) */
  onRetry: () => void;
  /** 처음으로 (분석 상태를 비우고 종목 선택으로) */
  onHome: () => void;
}

/**
 * 결과 화면(넓은 화면) 맨 위 바. 모바일 화면의 PageHeader + 하단 고정 버튼(BottomBar)을 한 줄로 모았다.
 */
export function ResultHeader({ title, onBack, onRetry, onHome }: ResultHeaderProps) {
  return (
    <header className="sticky top-0 z-20 flex h-16 items-center gap-3 bg-surface/95 backdrop-blur">
      <button
        type="button"
        onClick={onBack}
        aria-label="뒤로"
        className="-ml-2 flex size-10 items-center justify-center rounded-full text-ink hover:bg-page"
      >
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M15 5l-7 7 7 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      <h1 className="flex-1 truncate text-[20px] font-bold text-ink">{title}</h1>
      <Button variant="ghost" onClick={onHome}>
        처음으로
      </Button>
      <Button onClick={onRetry}>다른 영상으로 다시 비교</Button>
    </header>
  );
}
