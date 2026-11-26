import { useEffect, useRef, useState } from 'react';
import type { FramePlayer } from '@/hooks/useFramePlayer';
import { cn } from '@/utils/cn';

interface OverlayVideoProps {
  /** AnalysisResult.overlayUrl */
  src: string;
  /** 결과 화면의 재생 시계. 3D 뷰어와 같은 것을 받는다 */
  player: FramePlayer;
  fps: number;
  className?: string;
}

/** 영상 정보를 읽기 전 칸의 비율 (휴대폰 세로 영상) */
const DEFAULT_ASPECT = 9 / 16;

/** 재생 중 영상과 시계가 이만큼(초) 벌어지면 영상을 시계에 다시 맞춘다 */
const MAX_DRIFT_SEC = 0.1;

/**
 * 사용자 영상 위에 리타겟팅된 코치를 SMPL 메시로 겹친 영상.
 * 영상은 재생 순서로 만들어져 있어서 재생 프레임 k 가 영상 시각 [k, k + 1) / fps 다.
 * 멈춰 있을 땐 프레임마다 그 시각으로 옮기고, 재생 중엔 영상을 그대로 재생하다가 시계와 벌어지면 맞춘다.
 */
export function OverlayVideo({ src, player, fps, className }: OverlayVideoProps) {
  const ref = useRef<HTMLVideoElement>(null);
  // 칸의 폭을 영상 비율로 정한다 (세로 영상이면 좁게). 메타데이터를 읽으면 실제 비율로 바꾼다
  const [aspect, setAspect] = useState(DEFAULT_ASPECT);
  // 프레임 k 의 가운데 시각 (경계에 걸쳐 앞 프레임이 보이지 않게)
  const target = (player.frame + 0.5) / fps;
  const { isPlaying, speed } = player;

  useEffect(() => {
    const video = ref.current;
    if (!video) return;
    video.playbackRate = speed;
    if (isPlaying) video.play().catch(() => {});
    else video.pause();
  }, [isPlaying, speed]);

  useEffect(() => {
    const video = ref.current;
    if (!video) return;
    const drift = Math.abs(video.currentTime - target);
    if (drift > (isPlaying ? MAX_DRIFT_SEC : 0.5 / fps)) video.currentTime = target;
    // 끝까지 가서 멈춘 영상을 시계가 처음으로 돌려보낸 경우 (반복 재생)
    if (isPlaying && video.paused) video.play().catch(() => {});
  }, [target, isPlaying, fps]);

  return (
    <video
      ref={ref}
      src={src}
      muted
      playsInline
      preload="auto"
      aria-label="코치를 내 영상 위에 겹친 영상"
      onLoadedMetadata={(e) => {
        const { videoWidth, videoHeight } = e.currentTarget;
        if (videoWidth && videoHeight) setAspect(videoWidth / videoHeight);
      }}
      style={{ aspectRatio: aspect }}
      className={cn('rounded-card bg-ink object-contain', className)}
    />
  );
}
