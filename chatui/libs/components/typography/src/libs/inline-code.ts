import { cn } from '@memgpt/utils';
import { ClassValue } from 'clsx/clsx';

export const cnInlineCode = (...inputs: ClassValue[]) =>
	cn('rounded bg-muted px-[0.3rem] py-[0.2rem] font-mono text-sm font-semibold', inputs);
