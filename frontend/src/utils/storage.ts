/**
 * sessionStorage 를 JSON 으로 감싼 얇은 래퍼.
 * 새로고침해도 종목·결과가 남도록 쓴다. (File 객체는 저장하지 않는다)
 * 실패해도 앱이 멈추지 않도록 모든 접근을 try/catch 로 감싼다.
 */

const PREFIX = 'posture-coach:';

export function readStorage<T>(key: string): T | null {
  try {
    const raw = sessionStorage.getItem(PREFIX + key);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}

export function writeStorage<T>(key: string, value: T): void {
  try {
    sessionStorage.setItem(PREFIX + key, JSON.stringify(value));
  } catch {
    /* 용량 초과 등은 무시 */
  }
}

export function removeStorage(key: string): void {
  try {
    sessionStorage.removeItem(PREFIX + key);
  } catch {
    /* ignore */
  }
}
