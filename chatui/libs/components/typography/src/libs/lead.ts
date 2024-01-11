import { cn } from '@memgpt/utils';
import { ClassValue } from 'clsx/clsx';

export const cnLead = (...inputs: ClassValue[]) => cn('text-xl text-muted-foreground', inputs);
