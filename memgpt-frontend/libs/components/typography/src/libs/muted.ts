import { ClassValue } from 'clsx/clsx';
import { cn } from '@memgpt/utils';

export const cnMuted = (...inputs: ClassValue[]) => cn('text-sm text-muted-foreground', inputs);
