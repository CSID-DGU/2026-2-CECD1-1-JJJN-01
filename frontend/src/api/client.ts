/**
 * fetch 래퍼. 컴포넌트는 이 파일을 직접 쓰지 않고 api/analysis.ts 같은 모듈을 통해 호출한다.
 */

const rawBase = import.meta.env.VITE_API_BASE_URL?.trim() ?? '';

/** 끝의 슬래시를 뗀 베이스 URL. 비어 있으면 mock 모드 */
export const API_BASE_URL = rawBase.replace(/\/+$/, '');

export const USE_MOCK = import.meta.env.VITE_USE_MOCK === 'true' || API_BASE_URL === '';

export class ApiError extends Error {
  status: number;
  body?: unknown;

  constructor(status: number, message: string, body?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.body = body;
  }
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${path.startsWith('/') ? path : `/${path}`}`;

  let res: Response;
  try {
    res = await fetch(url, init);
  } catch {
    throw new ApiError(0, '서버에 연결할 수 없어요. 네트워크를 확인해 주세요.');
  }

  const contentType = res.headers.get('content-type') ?? '';
  const isJson = contentType.includes('application/json');
  const body: unknown = isJson ? await res.json().catch(() => undefined) : await res.text().catch(() => undefined);

  if (!res.ok) {
    const message =
      (isJson && body && typeof body === 'object' && 'message' in body && typeof body.message === 'string'
        ? body.message
        : null) ?? `요청에 실패했어요. (${res.status})`;
    throw new ApiError(res.status, message, body);
  }

  return body as T;
}
