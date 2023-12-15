import { ClassValue } from 'clsx/clsx';
import { cn } from '@memgpt/utils';

export const cnList = (...inputs: ClassValue[]) => cn('my-6 ml-6 list-disc [&>li]:mt-2', inputs);
