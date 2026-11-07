import { useEffect, useMemo, useRef, useState } from 'react';
import { readVideoMetadata, validateVideoFile, type VideoMetadata } from '@/utils/video';

interface UseVideoFileResult {
  /** 미리보기용 object URL. 파일이 없으면 null */
  previewUrl: string | null;
  metadata: VideoMetadata | null;
  /** 제한에 걸렸을 때 보여줄 문장 */
  error: string | null;
  /** 메타데이터 읽는 중 */
  loading: boolean;
}

/** 어느 파일의 메타데이터인지 같이 들고 있어서, 파일이 바뀌면 이전 값이 새어 나가지 않는다 */
interface MetaState {
  file: File;
  metadata: VideoMetadata | null;
  readError: string | null;
}

/** 고른 영상 파일의 미리보기 URL·길이를 읽고 업로드 제한을 검사한다. */
export function useVideoFile(file: File | null): UseVideoFileResult {
  // object URL 은 파일마다 하나. 이전 파일의 URL 은 다음 파일로 바뀔 때 정리한다.
  // (cleanup 에서 지우면 StrictMode 의 가짜 unmount 때 살아 있는 URL 이 지워진다)
  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  const prevUrlRef = useRef<string | null>(null);
  useEffect(() => {
    if (prevUrlRef.current && prevUrlRef.current !== previewUrl) URL.revokeObjectURL(prevUrlRef.current);
    prevUrlRef.current = previewUrl;
  }, [previewUrl]);

  const [meta, setMeta] = useState<MetaState | null>(null);

  useEffect(() => {
    if (!file) return;
    let cancelled = false;
    readVideoMetadata(file)
      .then((metadata) => {
        if (!cancelled) setMeta({ file, metadata, readError: null });
      })
      .catch((e: unknown) => {
        if (!cancelled)
          setMeta({ file, metadata: null, readError: e instanceof Error ? e.message : '영상을 읽을 수 없어요.' });
      });
    return () => {
      cancelled = true;
    };
  }, [file]);

  if (!file) return { previewUrl: null, metadata: null, error: null, loading: false };

  const current = meta?.file === file ? meta : null;
  const metadata = current?.metadata ?? null;
  const error = validateVideoFile(file, metadata ?? undefined) ?? current?.readError ?? null;
  return { previewUrl, metadata, error, loading: current === null };
}
