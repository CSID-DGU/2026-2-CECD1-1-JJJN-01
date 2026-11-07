import { UPLOAD_LIMITS } from '@/constants/upload';
import { formatBytes } from './format';

export interface VideoMetadata {
  durationSec: number;
  width: number;
  height: number;
}

/** 브라우저에서 영상 길이·해상도만 읽는다. (디코딩 전체를 하지 않으므로 빠르다) */
export function readVideoMetadata(file: File): Promise<VideoMetadata> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const video = document.createElement('video');
    video.preload = 'metadata';
    video.muted = true;

    const cleanup = () => {
      URL.revokeObjectURL(url);
      video.removeAttribute('src');
      video.load();
    };

    video.onloadedmetadata = () => {
      const meta = {
        durationSec: video.duration,
        width: video.videoWidth,
        height: video.videoHeight,
      };
      cleanup();
      resolve(meta);
    };
    video.onerror = () => {
      cleanup();
      reject(new Error('영상을 읽을 수 없어요. 다른 파일을 골라 주세요.'));
    };
    video.src = url;
  });
}

/** 제한에 걸리면 사용자에게 보여줄 문장을, 괜찮으면 null 을 돌려준다. */
export function validateVideoFile(file: File, meta?: VideoMetadata): string | null {
  const acceptedTypes: readonly string[] = UPLOAD_LIMITS.acceptedMimeTypes;
  if (file.type && !acceptedTypes.includes(file.type)) {
    return 'mp4, mov, webm 영상만 올릴 수 있어요.';
  }
  if (file.size > UPLOAD_LIMITS.maxBytes) {
    return `영상이 ${formatBytes(UPLOAD_LIMITS.maxBytes)}를 넘어요. (지금 ${formatBytes(file.size)})`;
  }
  if (meta && Number.isFinite(meta.durationSec) && meta.durationSec > UPLOAD_LIMITS.maxDurationSec) {
    return `영상은 ${UPLOAD_LIMITS.maxDurationSec}초 이내여야 해요. (지금 ${Math.round(meta.durationSec)}초)`;
  }
  return null;
}
