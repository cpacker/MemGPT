import { cn } from '@memgpt/utils';
import { ClassValue } from 'clsx/clsx';

export const cnBlockquote = (...inputs: ClassValue[]) => cn('mt-6 border-l-2 pl-6 italic', inputs);
