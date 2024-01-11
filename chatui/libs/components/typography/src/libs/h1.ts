import { cn } from '@memgpt/utils';
import { ClassValue } from 'clsx/clsx';

export const cnH1 = (...inputs: ClassValue[]) =>
	cn('scroll-m-20 text-4xl font-extrabold tracking-tight lg:text-5xl', inputs);
