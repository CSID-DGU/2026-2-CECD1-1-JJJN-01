/** 조건부 class 이름을 합친다. falsy 값은 버린다. */
export function cn(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(' ');
}
