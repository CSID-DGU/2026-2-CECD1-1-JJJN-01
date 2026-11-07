import { useRef, type ChangeEvent } from 'react';
import { Button } from '@/components/Button';
import { Spinner } from '@/components/Spinner';
import { UPLOAD_LIMITS } from '@/constants/upload';
import type { VideoMetadata } from '@/utils/video';
import { formatBytes, formatSeconds } from '@/utils/format';
import { cn } from '@/utils/cn';

interface VideoDropzoneProps {
  file: File | null;
  previewUrl: string | null;
  metadata: VideoMetadata | null;
  error: string | null;
  loading: boolean;
  onPick: (file: File) => void;
  onClear: () => void;
}

/**
 * 사용자 영상 선택 영역.
 * 상태(미리보기 URL·검사 결과)는 부모가 useVideoFile 로 만들어 넘기고, 여기서는 그리기만 한다.
 */
export function VideoDropzone({ file, previewUrl, metadata, error, loading, onPick, onClear }: VideoDropzoneProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const picked = e.target.files?.[0];
    if (picked) onPick(picked);
    e.target.value = ''; // 같은 파일을 다시 골라도 change 가 나도록
  };

  return (
    <div>
      <input ref={fileInputRef} type="file" accept={UPLOAD_LIMITS.accept} className="hidden" onChange={handleChange} />
      {/* 모바일 브라우저에서는 카메라가 바로 열린다. 데스크톱은 일반 파일 선택으로 동작 */}
      <input
        ref={cameraInputRef}
        type="file"
        accept="video/*"
        capture="environment"
        className="hidden"
        onChange={handleChange}
      />

      {!file ? (
        <div className="rounded-card border border-dashed border-ink-faint/60 p-5">
          <p className="text-[14px] leading-relaxed text-ink-muted">
            한 번의 동작이 담긴 영상을 올려 주세요.
            <br />
            {UPLOAD_LIMITS.maxDurationSec}초 이내, {formatBytes(UPLOAD_LIMITS.maxBytes)} 이하
          </p>
          <div className="mt-4 flex gap-2">
            <Button block onClick={() => fileInputRef.current?.click()}>
              영상 고르기
            </Button>
            <Button block variant="secondary" onClick={() => cameraInputRef.current?.click()}>
              지금 촬영하기
            </Button>
          </div>
        </div>
      ) : (
        <div className={cn('overflow-hidden rounded-card border', error ? 'border-bad' : 'border-line')}>
          <div className="relative aspect-[3/4] max-h-[360px] w-full bg-ink">
            {previewUrl && (
              <video
                src={previewUrl}
                className="size-full object-contain"
                controls
                muted
                playsInline
                preload="metadata"
              />
            )}
            {loading && (
              <div className="absolute inset-0 flex items-center justify-center bg-ink/40">
                <Spinner className="text-white" />
              </div>
            )}
          </div>
          <div className="flex items-center gap-3 px-4 py-3">
            <div className="min-w-0 flex-1">
              <div className="truncate text-[14px] font-medium text-ink">{file.name}</div>
              <div className="mt-0.5 text-[12px] text-ink-muted">
                {formatBytes(file.size)}
                {metadata && Number.isFinite(metadata.durationSec) && (
                  <>
                    {' · '}
                    {formatSeconds(metadata.durationSec)}
                    {' · '}
                    {metadata.width}×{metadata.height}
                  </>
                )}
              </div>
            </div>
            <Button variant="ghost" onClick={onClear}>
              다시 고르기
            </Button>
          </div>
          {error && <p className="border-t border-bad/30 bg-bad-soft px-4 py-2.5 text-[13px] text-bad">{error}</p>}
        </div>
      )}
    </div>
  );
}
