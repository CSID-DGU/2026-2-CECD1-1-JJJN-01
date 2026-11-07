import { useCallback, useEffect, useRef, useState } from 'react';

interface UseFramePlayerOptions {
  frameCount: number;
  fps: number;
  /** 끝까지 가면 처음으로 (기본 true) */
  loop?: boolean;
}

export interface FramePlayer {
  frame: number;
  isPlaying: boolean;
  speed: number;
  play: () => void;
  pause: () => void;
  toggle: () => void;
  seek: (frame: number) => void;
  setSpeed: (speed: number) => void;
}

/**
 * 스켈레톤 시퀀스 재생 상태.
 * requestAnimationFrame 으로 시간을 재고, fps 에 맞춰 frame 을 올린다.
 */
export function useFramePlayer({ frameCount, fps, loop = true }: UseFramePlayerOptions): FramePlayer {
  const [frame, setFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  const frameRef = useRef(0);
  const accRef = useRef(0);
  const lastRef = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);

  const seek = useCallback(
    (next: number) => {
      const clamped = Math.min(Math.max(Math.round(next), 0), Math.max(frameCount - 1, 0));
      frameRef.current = clamped;
      setFrame(clamped);
    },
    [frameCount],
  );

  // frameCount 가 줄어들면 범위 안으로
  useEffect(() => {
    if (frameRef.current > frameCount - 1) seek(0);
  }, [frameCount, seek]);

  useEffect(() => {
    if (!isPlaying || frameCount <= 1) return;

    const frameDuration = 1000 / fps;

    const step = (now: number) => {
      if (lastRef.current === null) lastRef.current = now;
      accRef.current += (now - lastRef.current) * speed;
      lastRef.current = now;

      let advanced = 0;
      while (accRef.current >= frameDuration) {
        accRef.current -= frameDuration;
        advanced += 1;
      }

      if (advanced > 0) {
        let next = frameRef.current + advanced;
        if (next >= frameCount) {
          if (loop) next %= frameCount;
          else {
            next = frameCount - 1;
            frameRef.current = next;
            setFrame(next);
            setIsPlaying(false);
            return;
          }
        }
        frameRef.current = next;
        setFrame(next);
      }
      rafRef.current = requestAnimationFrame(step);
    };

    lastRef.current = null;
    accRef.current = 0;
    rafRef.current = requestAnimationFrame(step);

    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [isPlaying, fps, speed, frameCount, loop]);

  const play = useCallback(() => setIsPlaying(true), []);
  const pause = useCallback(() => setIsPlaying(false), []);
  const toggle = useCallback(() => setIsPlaying((p) => !p), []);

  return { frame, isPlaying, speed, play, pause, toggle, seek, setSpeed };
}
