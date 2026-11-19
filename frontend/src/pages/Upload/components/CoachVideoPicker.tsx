import { useRef, type ChangeEvent } from 'react';
import { Badge } from '@/components/Badge';
import { OptionCard } from '@/components/OptionCard';
import { coachVideoFor } from '@/constants/coachVideos';
import { UPLOAD_LIMITS } from '@/constants/upload';
import type { CoachSelection } from '@/types/coach';
import type { ExerciseType } from '@/types/exercise';
import { formatBytes, formatSeconds } from '@/utils/format';
import type { VideoMetadata } from '@/utils/video';
import { VideoPreview } from './VideoPreview';

interface CoachVideoPickerProps {
  exercise: ExerciseType;
  value: CoachSelection | null;
  onChange: (selection: CoachSelection | null) => void;
  /** 직접 올린 코치 영상의 미리보기 · 검사 결과. 부모가 useVideoFile 로 만들어 넘긴다 */
  customPreviewUrl: string | null;
  customMetadata: VideoMetadata | null;
  customError: string | null;
  customLoading: boolean;
}

/**
 * 코치 영상 선택.
 * 직접 올리는 쪽을 먼저 두고, 올리지 않으면 기본 제공 영상을 쓴다.
 * 고른 쪽의 영상은 카드 바로 아래에서 재생해 볼 수 있다.
 */
export function CoachVideoPicker({
  exercise,
  value,
  onChange,
  customPreviewUrl,
  customMetadata,
  customError,
  customLoading,
}: CoachVideoPickerProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const defaultVideo = coachVideoFor(exercise);

  const customFile = value?.kind === 'custom' ? value.file : null;
  const defaultSelected = !!defaultVideo && value?.kind === 'default' && value.coachId === defaultVideo.id;

  const handleFile = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onChange({ kind: 'custom', file });
    e.target.value = '';
  };

  const customInfo = customFile
    ? [
        formatBytes(customFile.size),
        customMetadata && Number.isFinite(customMetadata.durationSec) && formatSeconds(customMetadata.durationSec),
        customMetadata && `${customMetadata.width}×${customMetadata.height}`,
      ]
        .filter(Boolean)
        .join(' · ')
    : null;

  return (
    <div role="radiogroup" aria-label="코치 영상" className="flex flex-col gap-3">
      <input ref={inputRef} type="file" accept={UPLOAD_LIMITS.accept} className="hidden" onChange={handleFile} />

      <OptionCard
        selected={value?.kind === 'custom'}
        onSelect={() => inputRef.current?.click()}
        title={customFile ? customFile.name : '코치 영상 직접 올리기'}
        description={
          customFile
            ? `${customInfo} · 다시 누르면 다른 영상을 고를 수 있어요`
            : defaultVideo
              ? '따라 하고 싶은 영상이 따로 있다면 올려 주세요'
              : '이 종목은 기본 영상이 아직 없어요. 따라 할 영상을 올려 주세요'
        }
        trailing={customError ? <Badge tone="bad">확인 필요</Badge> : undefined}
      />
      {customFile && (
        <div className="overflow-hidden rounded-card">
          <VideoPreview
            src={customPreviewUrl}
            landscape={!!customMetadata && customMetadata.width > customMetadata.height}
            loading={customLoading}
          />
        </div>
      )}
      {customError && <p className="-mt-1 px-1 text-[13px] text-bad">{customError}</p>}

      {defaultVideo && (
        <>
          <p className="pt-1 text-[13px] text-ink-muted">또는 기본 영상으로 비교하기</p>
          <OptionCard
            selected={defaultSelected}
            onSelect={() => onChange({ kind: 'default', coachId: defaultVideo.id })}
            title={defaultVideo.title}
            description={defaultVideo.description}
            trailing={
              defaultVideo.durationSec ? <Badge>{formatSeconds(defaultVideo.durationSec, 0)}</Badge> : undefined
            }
          />
          {defaultSelected && (
            <div className="overflow-hidden rounded-card">
              <VideoPreview src={defaultVideo.videoUrl} landscape />
            </div>
          )}
        </>
      )}
    </div>
  );
}
