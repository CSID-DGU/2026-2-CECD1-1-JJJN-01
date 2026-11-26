import type { ReactNode } from 'react';
import type { Segment } from '@/types/analysis';
import { SegmentDetail } from './SegmentDetail';

interface ViewerPanelProps {
  /** 지금 보여줄 구간 (고른 구간, 재생 중이면 재생 위치의 구간). 없으면 피드백 요약을 보여준다 */
  segment: Segment | null;
  /** 3D 뷰어. 오버레이 영상이 없어서 3D 가 왼쪽에 있으면 비운다 */
  viewer?: ReactNode;
  /** 구간이 없을 때 3D 아래에 보여줄 피드백 요약 */
  summary?: string;
}

/**
 * 결과 화면 오른쪽 칸: 3D 뷰어(영상과 같은 재생 프레임) + 지금 구간의 상세.
 * 처음부터 첫 프레임의 스켈레톤을 보여주고, 구간을 누르면 그 구간으로 옮겨 간다.
 */
export function ViewerPanel({ segment, viewer, summary }: ViewerPanelProps) {
  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <p className="text-[15px] font-bold text-ink">
        {segment ? `${segment.label} 구간` : '3D로 보기'}
        {viewer && <span className="ml-2 text-[13px] font-medium text-ink-faint">드래그해서 돌려 보세요</span>}
      </p>
      {viewer && <div className="min-h-0 flex-1">{viewer}</div>}
      {segment ? (
        <SegmentDetail segment={segment} />
      ) : (
        summary && (
          <div className="border-l-2 border-coach pl-4">
            <p className="text-[12px] font-semibold text-coach">코치 피드백 요약</p>
            <p className="mt-1 text-[15px] leading-relaxed text-ink">{summary}</p>
          </div>
        )
      )}
    </div>
  );
}
