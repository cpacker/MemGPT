import { cn } from '@memgpt/utils';
import { ClassValue } from 'clsx/clsx';

export const cnLarge = (...inputs: ClassValue[]) => cn('text-lg font-semibold', inputs);
