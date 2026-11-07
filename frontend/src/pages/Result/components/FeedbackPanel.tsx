import type { LlmFeedback, Segment } from '@/types/analysis';

interface FeedbackPanelProps {
  feedback?: LlmFeedback;
  segments: Segment[];
  /** 코멘트가 가리키는 구간으로 이동 */
  onJump: (segmentId: string) => void;
}

/**
 * LLM 피드백 영역.
 * TODO(backend): 피드백 형식이 정해지면 LlmFeedback 타입과 이 컴포넌트를 함께 고친다.
 * 지금은 "요약 한 줄 + 구간별 코멘트" 로 가정하고 그린다.
 */
export function FeedbackPanel({ feedback, segments, onJump }: FeedbackPanelProps) {
  if (!feedback) {
    return (
      <div className="rounded-card border border-dashed border-ink-faint/60 p-5 text-center">
        <p className="text-[14px] font-semibold text-ink-muted">코치 피드백이 들어올 자리예요</p>
        <p className="mt-1 text-[13px] text-ink-faint">비교 결과를 문장으로 정리한 피드백이 여기에 보여요.</p>
      </div>
    );
  }

  const labelOf = (id: string) => segments.find((s) => s.id === id)?.label ?? id;

  return (
    <div>
      <p className="text-[16px] leading-relaxed font-medium text-ink">{feedback.summary}</p>
      {feedback.items.length > 0 && (
        <ul className="mt-4 flex flex-col gap-3">
          {feedback.items.map((item, i) => (
            <li key={i} className="border-l-2 border-line pl-3">
              {item.segmentId && (
                <button
                  type="button"
                  onClick={() => onJump(item.segmentId!)}
                  className="mb-1 text-[12px] font-semibold text-coach underline underline-offset-4"
                >
                  {labelOf(item.segmentId)} 구간 보기
                </button>
              )}
              <p className="text-[14px] leading-relaxed text-ink-muted">{item.text}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
