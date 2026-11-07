/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** 백엔드 주소. 비어 있으면 mock 모드 */
  readonly VITE_API_BASE_URL?: string;
  /** 'true' 면 mock 모드 */
  readonly VITE_USE_MOCK?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
