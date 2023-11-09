import { ClassValue } from 'clsx/clsx';
import { cn } from '@memgpt/utils';

export const cnLarge = (...inputs: ClassValue[]) => cn('text-lg font-semibold', inputs);
