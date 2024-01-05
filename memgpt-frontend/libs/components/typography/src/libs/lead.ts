import { ClassValue } from 'clsx/clsx';
import { cn } from '@memgpt/utils';

export const cnLead = (...inputs: ClassValue[]) => cn('text-xl text-muted-foreground', inputs);
