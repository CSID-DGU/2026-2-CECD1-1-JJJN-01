import { Outlet } from 'react-router';

/**
 * 넓은 화면 레이아웃 (결과 화면).
 * 데스크톱 폭 기준으로 가운데 정렬한다. 좁은 화면 대응(반응형)은 다음 단계에서 한다.
 */
export function WebLayout() {
  return (
    <div className="min-h-dvh bg-surface">
      <div className="mx-auto flex min-h-dvh w-full max-w-[1240px] flex-col px-8">
        <Outlet />
      </div>
    </div>
  );
}
