import { useMemo, useRef, type ChangeEvent } from 'react';
import { Badge } from '@/components/Badge';
import { OptionCard } from '@/components/OptionCard';
import { coachVideosFor } from '@/constants/coachVideos';
import { UPLOAD_LIMITS } from '@/constants/upload';
import type { CoachSelection } from '@/types/coach';
import type { ExerciseType } from '@/types/exercise';
import { formatBytes, formatSeconds } from '@/utils/format';
import { validateVideoFile } from '@/utils/video';

interface CoachVideoPickerProps {
  exercise: ExerciseType;
  value: CoachSelection | null;
  onChange: (selection: CoachSelection | null) => void;
}

/**
 * 코치 영상 선택.
 * 직접 올리는 쪽을 먼저 두고, 올리지 않으면 기본 제공 영상 중 하나를 쓴다.
 */
export function CoachVideoPicker({ exercise, value, onChange }: CoachVideoPickerProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const options = useMemo(() => coachVideosFor(exercise), [exercise]);

  const customFile = value?.kind === 'custom' ? value.file : null;
  const customError = customFile ? validateVideoFile(customFile) : null;

  const handleFile = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onChange({ kind: 'custom', file });
    e.target.value = '';
  };

  return (
    <div role="radiogroup" aria-label="코치 영상" className="flex flex-col gap-3">
      <input ref={inputRef} type="file" accept={UPLOAD_LIMITS.accept} className="hidden" onChange={handleFile} />

      <OptionCard
        selected={value?.kind === 'custom'}
        onSelect={() => inputRef.current?.click()}
        title={customFile ? customFile.name : '코치 영상 직접 올리기'}
        description={
          customFile
            ? `${formatBytes(customFile.size)} · 다시 누르면 다른 영상을 고를 수 있어요`
            : '따라 하고 싶은 영상이 따로 있다면 올려 주세요'
        }
        trailing={customError ? <Badge tone="bad">확인 필요</Badge> : undefined}
      />
      {customError && <p className="-mt-1 px-1 text-[13px] text-bad">{customError}</p>}

      <p className="pt-1 text-[13px] text-ink-muted">또는 기본 영상 중에서 고르기</p>

      {options.map((option) => (
        <OptionCard
          key={option.id}
          selected={value?.kind === 'default' && value.coachId === option.id}
          onSelect={() => onChange({ kind: 'default', coachId: option.id })}
          title={option.title}
          description={option.description}
          trailing={option.durationSec ? <Badge>{formatSeconds(option.durationSec, 0)}</Badge> : undefined}
        />
      ))}
    </div>
  );
}
