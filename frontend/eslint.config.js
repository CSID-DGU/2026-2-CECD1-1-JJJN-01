import js from '@eslint/js';
import prettier from 'eslint-config-prettier';
import reactHooks from 'eslint-plugin-react-hooks';
import reactRefresh from 'eslint-plugin-react-refresh';
import globals from 'globals';
import tseslint from 'typescript-eslint';

/**
 * ESLint flat config.
 * 포맷은 Prettier 가 맡으므로 (eslint-config-prettier) 스타일 규칙은 여기 두지 않는다.
 */
export default tseslint.config(
  { ignores: ['dist', 'node_modules'] },

  js.configs.recommended,
  ...tseslint.configs.recommended,
  reactHooks.configs.flat.recommended,
  reactRefresh.configs.vite,

  {
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      ecmaVersion: 2022,
      globals: globals.browser,
    },
    rules: {
      // 타입만 쓰는 import 는 `import type` 으로 (verbatimModuleSyntax 와 맞춤)
      '@typescript-eslint/consistent-type-imports': ['error', { fixStyle: 'inline-type-imports' }],
      // `_` 로 시작하는 건 일부러 안 쓰는 변수
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_', varsIgnorePattern: '^_' }],
    },
  },

  // vite.config.ts 등 Node 쪽 파일
  {
    files: ['*.config.{js,ts}'],
    languageOptions: { globals: globals.node },
  },

  // 반드시 마지막: Prettier 와 겹치는 규칙을 끈다
  prettier,
);
