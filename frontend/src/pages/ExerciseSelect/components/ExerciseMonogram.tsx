import type { ExerciseType } from '@/types/exercise';
import { cn } from '@/utils/cn';

const MONOGRAM: Record<ExerciseType, string> = {
  squat: 'SQ',
  deadlift: 'DL',
  bench_press: 'BP',
};

interface ExerciseMonogramProps {
  exercise: ExerciseType;
  selected?: boolean;
}

/** 종목을 나타내는 두 글자 마크. 아이콘 라이브러리 없이 종목을 구분한다. */
export function ExerciseMonogram({ exercise, selected = false }: ExerciseMonogramProps) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        'flex size-11 items-center justify-center rounded-[12px] text-[13px] font-bold tracking-wide',
        selected ? 'bg-ink text-white' : 'bg-page text-ink-muted',
      )}
    >
      {MONOGRAM[exercise]}
    </span>
  );
}
