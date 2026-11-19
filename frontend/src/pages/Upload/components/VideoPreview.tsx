import { Spinner } from '@/components/Spinner';
import { cn } from '@/utils/cn';

interface VideoPreviewProps {
  /** 재생할 영상 주소 (object URL 또는 정적 파일). 없으면 빈 상자만 그린다 */
  src: string | null;
  /** 가로 영상이면 16:9, 아니면 세로(3:4) 상자에 담는다 */
  landscape?: boolean;
  /** 메타데이터 읽는 중 */
  loading?: boolean;
}

/** 고른 영상을 그 자리에서 재생해 보는 상자. 내 영상 · 코치 영상이 같이 쓴다. */
export function VideoPreview({ src, landscape = false, loading = false }: VideoPreviewProps) {
  return (
    <div className={cn('relative w-full bg-ink', landscape ? 'aspect-video' : 'aspect-[3/4] max-h-[360px]')}>
      {src && <video src={src} className="size-full object-contain" controls muted playsInline preload="metadata" />}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-ink/40">
          <Spinner className="text-white" />
        </div>
      )}
    </div>
  );
}
