/**
 * 업로드 제한. 백엔드와 같은 값으로 맞춘다.
 * TODO: 백엔드 합의 후 확정
 */
export const UPLOAD_LIMITS = {
  /** 100MB */
  maxBytes: 100 * 1024 * 1024,
  /** 초 */
  maxDurationSec: 30,
  /** 브라우저 file input accept 값 */
  accept: 'video/mp4,video/quicktime,video/webm',
  acceptedMimeTypes: ['video/mp4', 'video/quicktime', 'video/webm'],
} as const;

/** 분석 상태 폴링 간격 (ms) */
export const JOB_POLL_INTERVAL_MS = 1500;
