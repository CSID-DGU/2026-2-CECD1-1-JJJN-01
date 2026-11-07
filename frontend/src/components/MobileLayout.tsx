import { Outlet } from 'react-router';

/**
 * 모바일 폭 한 칸짜리 레이아웃.
 * 데스크톱에서는 가운데에 폰 폭으로 세워두고, 폰에서는 꽉 찬다.
 */
export function MobileLayout() {
  return (
    <div className="min-h-dvh bg-page">
      <div className="relative mx-auto flex min-h-dvh w-full max-w-[430px] flex-col bg-surface sm:border-x sm:border-line">
        <Outlet />
      </div>
    </div>
  );
}
