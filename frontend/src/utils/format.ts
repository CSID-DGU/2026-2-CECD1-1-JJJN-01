export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${Math.round(kb)}KB`;
  const mb = kb / 1024;
  if (mb < 1024) return `${mb.toFixed(mb < 10 ? 1 : 0)}MB`;
  return `${(mb / 1024).toFixed(1)}GB`;
}

/** 1.3 → "1.3초", 12 → "12초" */
export function formatSeconds(sec: number, digits = 1): string {
  const rounded = Number(sec.toFixed(digits));
  return `${rounded}초`;
}

/** [1.3, 1.8] → "1.3~1.8초" */
export function formatRange(range: [number, number]): string {
  const [a, b] = range;
  return `${Number(a.toFixed(2))}~${Number(b.toFixed(2))}초`;
}

/** 1.5 → "×1.5" */
export function formatRatio(ratio: number): string {
  return `×${ratio.toFixed(2).replace(/\.?0+$/, '')}`;
}

/** 15.9 → "15.9°" */
export function formatDegrees(deg: number): string {
  return `${Math.abs(deg).toFixed(1)}°`;
}
