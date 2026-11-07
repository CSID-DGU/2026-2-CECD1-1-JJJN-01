import type { ReactNode } from 'react';

interface SectionHeadingProps {
  title: string;
  description?: string;
  /** 제목 줄 오른쪽 */
  action?: ReactNode;
}

/** 화면 안의 작은 섹션 제목 */
export function SectionHeading({ title, description, action }: SectionHeadingProps) {
  return (
    <div className="mb-3 flex items-end justify-between gap-3">
      <div>
        <h2 className="text-[17px] font-bold text-ink">{title}</h2>
        {description && <p className="mt-0.5 text-[13px] text-ink-muted">{description}</p>}
      </div>
      {action}
    </div>
  );
}
