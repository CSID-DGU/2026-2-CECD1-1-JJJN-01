import type { FramePlayer } from '@/hooks/useFramePlayer';

interface FrameControlsProps {
  player: FramePlayer;
  fps: number;
  frameCount: number;
}

const SPEEDS = [1, 0.5, 0.25];

/** 재생 · 멈춤 · 프레임 슬라이더 · 배속 */
export function FrameControls({ player, fps, frameCount }: FrameControlsProps) {
  const currentSec = player.frame / fps;
  const totalSec = Math.max(frameCount - 1, 0) / fps;

  const cycleSpeed = () => {
    const i = SPEEDS.indexOf(player.speed);
    player.setSpeed(SPEEDS[(i + 1) % SPEEDS.length]);
  };

  return (
    <div className="flex items-center gap-3">
      <button
        type="button"
        onClick={player.toggle}
        aria-label={player.isPlaying ? '멈춤' : '재생'}
        className="flex size-10 shrink-0 items-center justify-center rounded-full bg-ink text-white"
      >
        {player.isPlaying ? (
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
            <rect x="3" y="2" width="4" height="12" rx="1" />
            <rect x="9" y="2" width="4" height="12" rx="1" />
          </svg>
        ) : (
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
            <path d="M4 2.5v11a1 1 0 0 0 1.5.87l9-5.5a1 1 0 0 0 0-1.74l-9-5.5A1 1 0 0 0 4 2.5z" />
          </svg>
        )}
      </button>

      <input
        type="range"
        min={0}
        max={Math.max(frameCount - 1, 0)}
        step={1}
        value={player.frame}
        onChange={(e) => {
          player.pause();
          player.seek(Number(e.target.value));
        }}
        aria-label="프레임"
        className="h-1.5 min-w-0 flex-1 cursor-pointer accent-ink"
      />

      <span className="w-[86px] shrink-0 text-right text-[12px] text-ink-muted tabular-nums">
        {currentSec.toFixed(2)} / {totalSec.toFixed(2)}초
      </span>

      <button
        type="button"
        onClick={cycleSpeed}
        aria-label="재생 속도"
        className="h-8 w-12 shrink-0 rounded-full border border-line text-[12px] font-semibold text-ink"
      >
        {player.speed}×
      </button>
    </div>
  );
}
