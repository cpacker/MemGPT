import { ClassValue } from 'clsx/clsx';
import { cn } from '@memgpt/utils';

export const cnBlockquote = (...inputs: ClassValue[]) => cn('mt-6 border-l-2 pl-6 italic', inputs);
