import type { Segment, SegmentVerdict, TimingLabel } from '@/types/analysis';

/** 화면에서 쓰는 색 톤. components/Badge 의 tone 과 같은 이름을 쓴다. */
export type Tone = 'neutral' | 'good' | 'bad' | 'warn' | 'slow' | 'fast';

export const VERDICT_LABEL: Record<SegmentVerdict, string> = {
  match: '일치',
  missing: '누락',
  extra: '추가',
  substitute: '대체',
};

export const VERDICT_TONE: Record<SegmentVerdict, Tone> = {
  match: 'good',
  missing: 'bad',
  extra: 'warn',
  substitute: 'warn',
};

export const TIMING_LABEL: Record<TimingLabel, string> = {
  ok: '알맞음',
  slow: '느림',
  fast: '빠름',
};

export const TIMING_TONE: Record<TimingLabel, Tone> = {
  ok: 'good',
  slow: 'slow',
  fast: 'fast',
};

/** 타임라인 블록 하나를 어떤 색으로 칠할지. 일치 구간은 타이밍으로, 나머지는 판정으로 */
export function segmentTone(segment: Segment): Tone {
  if (segment.verdict !== 'match') return VERDICT_TONE[segment.verdict];
  if (segment.timing) return TIMING_TONE[segment.timing.label];
  return 'good';
}

/** 사용자 영상의 시각(초)에 해당하는 구간 */
export function segmentAtTime(segments: Segment[], sec: number): Segment | null {
  for (const s of segments) {
    if (!s.studentRange) continue;
    const [a, b] = s.studentRange;
    if (sec >= a && sec < b) return s;
  }
  // 끝 프레임은 마지막 구간으로
  const last = [...segments].reverse().find((s) => s.studentRange);
  if (last?.studentRange && sec >= last.studentRange[1]) return last;
  return null;
}

export type ScoreTone = 'good' | 'warn' | 'bad';

/** 0~100 점수를 톤으로 */
export function scoreTone(total: number): ScoreTone {
  if (total >= 80) return 'good';
  if (total >= 60) return 'warn';
  return 'bad';
}

export function scoreCaption(total: number): string {
  if (total >= 80) return '코치 동작과 거의 같아요';
  if (total >= 60) return '흐름은 맞아요. 손볼 구간이 몇 군데 있어요';
  return '구간별로 차이가 커요. 아래에서 어디가 다른지 확인해 보세요';
}
