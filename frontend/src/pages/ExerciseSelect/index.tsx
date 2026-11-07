import { useState } from 'react';
import { useNavigate } from 'react-router';
import { BottomBar } from '@/components/BottomBar';
import { Button } from '@/components/Button';
import { PageHeader } from '@/components/PageHeader';
import { SKIP_INPUT_GATES } from '@/constants/dev';
import { EXERCISE_BY_ID } from '@/constants/exercises';
import { ROUTES } from '@/constants/routes';
import { useAnalysis } from '@/context/useAnalysis';
import type { ExerciseType } from '@/types/exercise';
import { ExerciseList } from './components/ExerciseList';

/** 1단계: 어떤 운동을 비교할지 고른다. */
export function ExerciseSelectPage() {
  const navigate = useNavigate();
  const { exercise, result, setExercise } = useAnalysis();
  const [selected, setSelected] = useState<ExerciseType | null>(exercise);

  const handleNext = () => {
    const next = selected ?? (SKIP_INPUT_GATES ? 'squat' : null);
    if (!next) return;
    setExercise(next);
    navigate(ROUTES.upload);
  };

  return (
    <>
      <PageHeader />

      <main className="flex-1 px-5 pb-6">
        <div className="pt-4 pb-7">
          <h1 className="text-[26px] leading-tight font-bold tracking-tight text-ink">
            어떤 동작을
            <br />
            코치와 비교할까요?
          </h1>
          <p className="mt-3 text-[15px] leading-relaxed text-ink-muted">
            코치의 동작을 내 체형에 맞게 옮긴 다음, 내 동작과 구간별로 비교해 드려요.
          </p>
        </div>

        <ExerciseList value={selected} onChange={setSelected} />

        {result && (
          <button
            type="button"
            onClick={() => navigate(ROUTES.result)}
            className="mt-6 text-[14px] font-medium text-ink-muted underline underline-offset-4"
          >
            지난 {EXERCISE_BY_ID[result.exercise].name} 분석 결과 다시 보기
          </button>
        )}
      </main>

      <BottomBar>
        <Button size="lg" block disabled={!SKIP_INPUT_GATES && !selected} onClick={handleNext}>
          다음
        </Button>
      </BottomBar>
    </>
  );
}
