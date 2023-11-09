import { ClassValue } from 'clsx/clsx';
import { cn } from '@memgpt/utils';

export const cnSmall = (...inputs: ClassValue[]) => cn('text-sm font-medium leading-none', inputs);
