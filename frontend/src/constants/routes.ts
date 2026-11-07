export const ROUTES = {
  home: '/',
  upload: '/upload',
  analyzing: '/analyzing',
  result: '/result',
} as const;

export type RoutePath = (typeof ROUTES)[keyof typeof ROUTES];
