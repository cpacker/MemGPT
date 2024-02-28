import { cn } from '@memgpt/utils';
import { ClassValue } from 'clsx/clsx';

export const cnMuted = (...inputs: ClassValue[]) => cn('text-sm text-muted-foreground', inputs);
