/**
 * 개발 플래그: 입력 검사 건너뛰기.
 * true 면 종목을 안 골라도 "다음", 영상을 안 올려도 "분석 시작" 이 눌려서 화면을 끝까지 확인할 수 있다.
 * (종목은 스쿼트, 영상은 빈 파일로 대신한다)
 *
 * mock 모드일 때만 켜진다: api/client.ts 의 USE_MOCK 과 같은 조건이다
 * (VITE_USE_MOCK=true, 또는 VITE_API_BASE_URL 이 비어 있음). 실제 백엔드에 붙이면 자동으로 꺼져서
 * 빈 파일이 서버로 가지 않는다. constants/ 가 api/ 를 가져오지 않도록 env 를 직접 읽는다.
 */
const apiBase = import.meta.env.VITE_API_BASE_URL?.trim() ?? '';

export const SKIP_INPUT_GATES = import.meta.env.VITE_USE_MOCK === 'true' || apiBase === '';
