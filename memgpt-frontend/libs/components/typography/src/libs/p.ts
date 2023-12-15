import { ClassValue } from 'clsx/clsx';
import { cn } from '@memgpt/utils';

export const cnP = (...inputs: ClassValue[]) => cn('leading-7 [&:not(:first-child)]:mt-6', inputs);
