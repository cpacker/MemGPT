import { cn } from '@memgpt/utils';
import { ClassValue } from 'clsx/clsx';

export const cnH3 = (...inputs: ClassValue[]) => cn('scroll-m-20 text-2xl font-semibold tracking-tight', inputs);
