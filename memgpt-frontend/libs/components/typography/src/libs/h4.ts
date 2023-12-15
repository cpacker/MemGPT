import { ClassValue } from 'clsx/clsx';
import { cn } from '@memgpt/utils';

export const cnH4 = (...inputs: ClassValue[]) => cn('scroll-m-20 text-xl font-semibold tracking-tight', inputs);
