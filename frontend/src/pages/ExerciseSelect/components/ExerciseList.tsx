import { OptionCard } from '@/components/OptionCard';
import { EXERCISES } from '@/constants/exercises';
import type { ExerciseType } from '@/types/exercise';
import { ExerciseMonogram } from './ExerciseMonogram';

interface ExerciseListProps {
  value: ExerciseType | null;
  onChange: (exercise: ExerciseType) => void;
}

export function ExerciseList({ value, onChange }: ExerciseListProps) {
  return (
    <div role="radiogroup" aria-label="운동 종목" className="flex flex-col gap-3">
      {EXERCISES.map((exercise) => {
        const selected = value === exercise.id;
        return (
          <OptionCard
            key={exercise.id}
            selected={selected}
            onSelect={() => onChange(exercise.id)}
            title={exercise.name}
            description={exercise.description}
            leading={<ExerciseMonogram exercise={exercise.id} selected={selected} />}
          />
        );
      })}
    </div>
  );
}
