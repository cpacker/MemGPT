import { cn } from '@memgpt/utils';
import { ClassValue } from 'clsx/clsx';

export const cnP = (...inputs: ClassValue[]) => cn('leading-7 [&:not(:first-child)]:mt-6', inputs);
