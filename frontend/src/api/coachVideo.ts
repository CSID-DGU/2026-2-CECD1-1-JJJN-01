import type { CoachVideoOption } from '@/types/coach';
import { ApiError } from './client';

/**
 * 기본 코치 영상(프론트에 들어 있는 정적 파일)을 서버로 보낼 File 로 읽어 온다.
 * 백엔드는 코치 영상이 항상 업로드로 온다고 보므로, 기본 영상도 직접 올린 영상과 같은 길로 보낸다.
 */
export async function fetchCoachVideoFile(option: CoachVideoOption): Promise<File> {
  let res: Response;
  try {
    res = await fetch(option.videoUrl);
  } catch {
    throw new ApiError(0, '기본 코치 영상을 불러올 수 없어요. 네트워크를 확인해 주세요.');
  }
  if (!res.ok) throw new ApiError(res.status, '기본 코치 영상을 불러올 수 없어요. 잠시 후 다시 시도해 주세요.');
  const blob = await res.blob();
  // 백엔드는 확장자 · MIME 으로 형식을 검사한다. 기본 영상은 모두 mp4 다
  return new File([blob], `${option.id}.mp4`, { type: 'video/mp4' });
}
