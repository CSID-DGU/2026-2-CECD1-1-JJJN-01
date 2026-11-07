import type { ButtonHTMLAttributes, ReactNode } from 'react';
import { cn } from '@/utils/cn';
import { Spinner } from './Spinner';

type Variant = 'primary' | 'secondary' | 'ghost';
type Size = 'md' | 'lg';

interface ButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'children'> {
  variant?: Variant;
  size?: Size;
  /** 가로로 꽉 채우기 */
  block?: boolean;
  loading?: boolean;
  children: ReactNode;
}

const VARIANT_CLASS: Record<Variant, string> = {
  primary: 'bg-ink text-white hover:bg-ink/90 disabled:bg-ink/25 disabled:text-white',
  secondary: 'bg-surface text-ink border border-line hover:bg-page disabled:text-ink-faint',
  ghost: 'bg-transparent text-ink-muted hover:bg-page disabled:text-ink-faint',
};

const SIZE_CLASS: Record<Size, string> = {
  md: 'h-11 px-4 text-[15px]',
  lg: 'h-13 px-5 text-base',
};

export function Button({
  variant = 'primary',
  size = 'md',
  block = false,
  loading = false,
  disabled,
  className,
  children,
  ...rest
}: ButtonProps) {
  return (
    <button
      type="button"
      disabled={disabled || loading}
      className={cn(
        'inline-flex items-center justify-center gap-2 rounded-control font-semibold transition-colors',
        'disabled:cursor-not-allowed',
        VARIANT_CLASS[variant],
        SIZE_CLASS[size],
        block && 'w-full',
        className,
      )}
      {...rest}
    >
      {loading && <Spinner size={18} className="text-current" />}
      {children}
    </button>
  );
}
