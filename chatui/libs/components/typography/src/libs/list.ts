import { cn } from '@memgpt/utils';
import { ClassValue } from 'clsx/clsx';

export const cnList = (...inputs: ClassValue[]) => cn('my-6 ml-6 list-disc [&>li]:mt-2', inputs);
