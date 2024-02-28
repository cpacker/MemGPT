import { cn } from '@memgpt/utils';
import { ClassValue } from 'clsx/clsx';

export const cnSmall = (...inputs: ClassValue[]) => cn('text-sm font-medium leading-none', inputs);
