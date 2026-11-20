import type { DeviationDirection, FrameMapKind, Segment, SegmentVerdict, TimingLabel } from '@/types/analysis';

/** 화면에서 쓰는 색 톤. components/Badge 의 tone 과 같은 이름을 쓴다. */
export type Tone = 'neutral' | 'good' | 'bad' | 'warn' | 'slow' | 'fast';

export const VERDICT_LABEL: Record<SegmentVerdict, string> = {
  match: '일치',
  missing: '빠진 동작',
  extra: '추가 동작',
  substitute: '다른 동작',
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

/** "코치보다 ○○ 12.3°" 의 ○○ */
export const DIRECTION_LABEL: Record<DeviationDirection, string> = {
  up: '위로',
  down: '아래로',
  forward: '앞으로',
  backward: '뒤로',
  inward: '안쪽으로',
  outward: '바깥쪽으로',
  left: '왼쪽으로',
  right: '오른쪽으로',
};

/** 한쪽만 움직이는 재생 프레임에 붙이는 말. 둘 다 움직이면(together) 아무것도 보여주지 않는다 */
export const WAITING_LABEL: Record<Exclude<FrameMapKind, 'together'>, string> = {
  coachAhead: '코치만 움직이는 중',
  coachWaits: '나만 움직이는 중',
};

/** 타임라인 블록 하나를 어떤 색으로 칠할지. 일치 구간은 타이밍으로, 나머지는 판정으로 */
export function segmentTone(segment: Segment): Tone {
  if (segment.verdict !== 'match') return VERDICT_TONE[segment.verdict];
  if (segment.timing) return TIMING_TONE[segment.timing.label];
  return 'good';
}

/** 재생 시각(초)에 해당하는 구간 */
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
